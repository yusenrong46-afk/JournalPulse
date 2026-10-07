"""Deletion retires logical creation IDs and fences authenticated in-flight writes."""

from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient

from journalpulse.api import create_app
from journalpulse.persistence import SQLiteRepository
from test_conversations_api import USER_A, chat_settings, start
from test_journals_api import HEADERS, OWNER, save, settings


@pytest.mark.parametrize("whole_account", [False, True])
def test_deleted_journal_request_cannot_restore_text(tmp_path, whole_account):
    configured = settings(tmp_path)
    with TestClient(create_app(settings=configured)) as client:
        request = {"client_request_id": str(uuid4()), "text": "Fictional writing to erase."}
        saved = save(client, **request)
        assert client.post("/v1/journal/entries", headers=HEADERS, json=request).json() == saved
        endpoint = "/v1/account/data" if whole_account else f"/v1/journal/entries/{saved['id']}"
        assert client.delete(endpoint, headers=HEADERS).status_code in (200, 204)
        revision = client.get("/v1/account/data-revision", headers=HEADERS).json()["revision"]
        current_headers = {**HEADERS, "X-JournalPulse-Data-Revision": str(revision)}
        retry = client.post("/v1/journal/entries", headers=current_headers, json=request)
        assert retry.status_code == 409
        assert client.get("/v1/export", headers=HEADERS).json()["journal_entries"] == []
        fresh = {**request, "client_request_id": str(uuid4())}
        assert client.post("/v1/journal/entries", headers=current_headers, json=fresh).status_code == 201


@pytest.mark.parametrize("whole_account", [False, True])
def test_deleted_chat_id_cannot_become_a_target_for_old_commands(tmp_path, whole_account):
    configured = chat_settings(tmp_path)
    headers = {"X-JournalPulse-User": USER_A}
    with TestClient(create_app(settings=configured)) as client:
        cid = str(uuid4())
        start(client, client_request_id=cid, llm_consent=False)
        endpoint = "/v1/account/data" if whole_account else f"/v1/conversations/{cid}"
        assert client.delete(endpoint, headers=headers).status_code in (200, 204)
        revision = client.get("/v1/account/data-revision", headers=headers).json()["revision"]
        current_headers = {**headers, "X-JournalPulse-Data-Revision": str(revision)}
        replacement = client.post("/v1/conversations", headers=current_headers, json={
            "client_request_id": cid, "llm_consent": False,
        })
        assert replacement.status_code == 409
        stale = client.post(f"/v1/conversations/{cid}/preference", headers=current_headers, json={
            "client_request_id": str(uuid4()), "expected_revision": 0, "preference": "listen",
        })
        assert stale.status_code == 404
        assert client.get("/v1/export", headers=headers).json()["conversations"] == []
        fresh = client.post("/v1/conversations", headers=current_headers,
                            json={"client_request_id": str(uuid4()), "llm_consent": False})
        assert fresh.status_code == 201
        assert fresh.json()["id"] != cid


def test_never_committed_save_cannot_cross_account_erasure(tmp_path):
    configured = settings(tmp_path)

    class DelayedRepository(SQLiteRepository):
        before_save = None

        def save_journal_entry(self, entry):
            if self.before_save:
                callback, self.before_save = self.before_save, None
                callback()
            return super().save_journal_entry(entry)

    repo = DelayedRepository(configured.database_path)
    with TestClient(create_app(settings=configured, repository_factory=lambda _: repo)) as client:
        # Erasure occurs after authentication, before this ID has ever committed.
        def erase():
            assert client.delete("/v1/account/data", headers=HEADERS).status_code == 200

        repo.before_save = erase
        request = {"client_request_id": str(uuid4()), "text": "Fictional delayed writing."}
        rejected = client.post("/v1/journal/entries", headers=HEADERS, json=request)
        assert rejected.status_code == 409
        assert repo.list_journal_entries(UUID(OWNER)) == []
        # A new user action after erasure belongs to the new account data revision.
        revision = client.get("/v1/account/data-revision", headers=HEADERS).json()["revision"]
        assert client.post("/v1/journal/entries", headers={
            **HEADERS, "X-JournalPulse-Data-Revision": str(revision),
        }, json={**request, "client_request_id": str(uuid4())}).status_code == 201


@pytest.mark.parametrize("whole_account", [False, True])
def test_deleted_legacy_reflection_request_cannot_restore_summary(tmp_path, whole_account):
    from test_research_beta_api import reflection_payload
    from test_research_beta_api import settings as reflection_settings

    with TestClient(create_app(settings=reflection_settings(tmp_path))) as client:
        request = reflection_payload(client_request_id=str(uuid4()), retain_text=True)
        response = client.post("/v1/reflections", headers=HEADERS, json=request)
        assert response.status_code == 201
        saved = response.json()
        endpoint = "/v1/account/data" if whole_account else f"/v1/reflections/{saved['id']}"
        assert client.delete(endpoint, headers=HEADERS).status_code in (200, 204)
        revision = client.get("/v1/account/data-revision", headers=HEADERS).json()["revision"]
        assert client.post("/v1/reflections", headers={
            **HEADERS, "X-JournalPulse-Data-Revision": str(revision),
        }, json=request).status_code == 409
        assert client.get("/v1/export", headers=HEADERS).json()["reflections"] == []


def test_deleted_activity_id_cannot_be_reoffered_on_a_fresh_chat(tmp_path):
    from test_conversations_api import say

    with TestClient(create_app(settings=chat_settings(tmp_path))) as client:
        first = start(client, llm_consent=False)
        card = say(client, first["id"], "Fictional routine moment.", goal="settle").json()["conversation"]
        request = {
            "client_request_id": str(uuid4()), "expected_conversation_revision": card["revision"],
            "resource_id": card["card"]["actions"][0]["id"],
        }
        offered = client.post(
            f"/v1/conversations/{first['id']}/activity-sessions", headers=HEADERS, json=request,
        )
        assert offered.status_code == 201
        assert client.delete(f"/v1/conversations/{first['id']}", headers=HEADERS).status_code == 204
        fresh = start(client, llm_consent=False)
        assert say(client, fresh["id"], "Fictional routine moment.", goal="settle").status_code == 200
        replay = client.post(
            f"/v1/conversations/{fresh['id']}/activity-sessions", headers=HEADERS, json=request,
        )
        assert replay.status_code == 409
        delayed = client.post(
            f"/v1/activity-sessions/{request['client_request_id']}/commands", headers=HEADERS,
            json={"client_request_id": str(uuid4()), "expected_revision": 0,
                  "expected_conversation_revision": card["revision"], "command": "start"},
        )
        assert delayed.status_code == 404


def test_tombstones_hold_only_identity_and_survive_empty_account_erasure(tmp_path):
    configured = settings(tmp_path)
    repo = SQLiteRepository(configured.database_path)
    with TestClient(create_app(settings=configured)) as client:
        saved = save(client)
        assert client.delete("/v1/account/data", headers=HEADERS).status_code == 200
        assert client.delete("/v1/account/data", headers=HEADERS).status_code == 200
    with repo.connect() as connection:
        columns = [row["name"] for row in connection.execute("PRAGMA table_info(deleted_object_ids)")]
        assert columns == ["object_kind", "object_id", "user_id"]
        markers = [tuple(row) for row in connection.execute("SELECT * FROM deleted_object_ids")]
        assert markers == [("journal_entries", saved["id"], OWNER)]
    assert repo.get_erasure_revision(UUID(OWNER)) == 2


def test_delayed_accept_after_source_delete_cannot_restore_derived_summary(tmp_path):
    from journalpulse.domain import Conversation
    from test_conversations_api import ScriptedClient, say

    configured = chat_settings(tmp_path)

    class DelayedAcceptance(SQLiteRepository):
        before_accept = None

        def accept_conversation(self, *args, **kwargs):
            callback, self.before_accept = self.before_accept, None
            if callback:
                callback()
            return super().accept_conversation(*args, **kwargs)

    repo = DelayedAcceptance(configured.database_path)
    with TestClient(create_app(settings=configured, repository_factory=lambda _: repo,
                              conversation_client=ScriptedClient(offers=[False, True]))) as client:
        source = save(client)
        original = start(client, source_entry_id=source["id"])
        assert say(client, original["id"], "A fictional difficult day.").status_code == 200
        card = say(client, original["id"], "I would like to understand.", goal="understand")
        assert card.status_code == 200
        current = card.json()["conversation"]

        def delete_source():
            assert client.delete(f"/v1/journal/entries/{source['id']}", headers=HEADERS).status_code == 204
            # Old acceptance read the card/summary already. Even an attempted
            # replacement with its same revision cannot establish a new target.
            with pytest.raises(ValueError, match="deleted"):
                repo.create_conversation(Conversation.model_validate(original).model_copy(update={
                    "source_entry_id": None, "source_entry_created_at": None,
                }))

        repo.before_accept = delete_source
        accepted = client.post(f"/v1/conversations/{original['id']}/accept", headers=HEADERS, json={
            "client_request_id": str(uuid4()),
            "action_id": current["card"]["decision_preview"]["action_id"],
            "expected_revision": current["revision"],
            "self_report": {"valence": 0, "arousal": .5, "agency": .5, "emotion_tags": []},
        })
        assert accepted.status_code == 404
        exported = repo.export_user_data(UUID(OWNER))
        assert all(exported[key] == [] for key in (
            "journal_entries", "conversations", "conversation_messages", "reflections",
        ))


def test_supabase_request_binding_signs_revision_without_mutating_shared_repository(tmp_path):
    import json

    import httpx

    from journalpulse.erasure import bind_repository
    from journalpulse.journal_models import JournalEntry
    from journalpulse.persistence import SupabaseRepository
    from test_supabase_repository import settings as supabase_settings

    entry = JournalEntry(user_id=UUID(OWNER), text="Fictional entry.")
    payloads = []

    def handler(request):
        payloads.append(json.loads(json.loads(request.content)["payload"]))
        return httpx.Response(200, json=entry.model_dump(mode="json"))

    repo = SupabaseRepository(supabase_settings(tmp_path), "fixture-token",
                              client=httpx.Client(transport=httpx.MockTransport(handler)))
    request_a = bind_repository(repo, UUID(OWNER), 4)
    request_b = bind_repository(repo, UUID(OWNER), 5)
    request_a.save_journal_entry(entry)
    request_b.save_journal_entry(entry)
    request_a.save_journal_entry(entry)
    repo.save_journal_entry(entry)
    assert [item.get("expected_erasure_revision") for item in payloads] == [4, 5, 4, None]


def test_unavailable_account_revision_rpc_fails_closed(tmp_path):
    import httpx

    from journalpulse.persistence import StorageUnavailable, SupabaseRepository
    from test_supabase_repository import settings as supabase_settings

    repo = SupabaseRepository(supabase_settings(tmp_path), "fixture-token",
                              client=httpx.Client(transport=httpx.MockTransport(
                                  lambda _: httpx.Response(404, json={"message": "RPC missing"})
                              )))
    with pytest.raises(StorageUnavailable, match="erasure guard"):
        repo.get_erasure_revision(UUID(OWNER))


@pytest.mark.parametrize("prior_erases,send_revision", [(0, False), (0, True), (1, True)])
def test_request_already_waiting_in_server_auth_cannot_save_after_erasure(
    tmp_path, monkeypatch, prior_erases, send_revision,
):
    import threading
    from concurrent.futures import ThreadPoolExecutor

    import journalpulse.api as api

    entered, release = threading.Event(), threading.Event()
    original_auth = api.resolve_auth

    def delayed_auth(*args, **kwargs):
        auth = original_auth(*args, **kwargs)
        if kwargs.get("authorization") == "Bearer synthetic-delayed-auth":
            entered.set()
            assert release.wait(5)
        return auth

    monkeypatch.setattr(api, "resolve_auth", delayed_auth)
    with (
        TestClient(create_app(settings=settings(tmp_path))) as client,
        ThreadPoolExecutor(max_workers=1) as pool,
    ):
        for _ in range(prior_erases):
            assert client.delete("/v1/account/data", headers=HEADERS).status_code == 200
        headers = {**HEADERS, "Authorization": "Bearer synthetic-delayed-auth"}
        if send_revision:
            headers["X-JournalPulse-Data-Revision"] = str(prior_erases)
        body = {"client_request_id": str(uuid4()), "text": "Fictional writing submitted before erasure."}
        pending = pool.submit(client.post, "/v1/journal/entries", headers=headers, json=body)
        try:
            assert entered.wait(5)
            assert client.delete("/v1/account/data", headers=HEADERS).status_code == 200
            assert client.get("/v1/export", headers=HEADERS).json()["journal_entries"] == []
        finally:
            release.set()
        resumed = pending.result(timeout=5)
        assert resumed.status_code == 409
        assert client.get("/v1/export", headers=HEADERS).json()["journal_entries"] == []


@pytest.mark.parametrize(
    "revision", ["bad", "-1", "1.2", "9007199254740992", "99999999999999999999999999999999"],
)
def test_invalid_account_revision_header_is_rejected_before_writing(tmp_path, revision):
    with TestClient(create_app(settings=settings(tmp_path))) as client:
        refused = client.post("/v1/journal/entries", headers={
            **HEADERS, "X-JournalPulse-Data-Revision": revision,
        }, json={"text": "Fictional writing."})
        assert refused.status_code == 422
        assert client.get("/v1/export", headers=HEADERS).json()["journal_entries"] == []


def test_revision_preflight_is_owner_scoped_uncached_and_allows_new_post_erase_writing(tmp_path):
    from test_journals_api import OTHER

    with TestClient(create_app(settings=settings(tmp_path))) as client:
        before = client.get("/v1/account/data-revision", headers=HEADERS)
        assert before.status_code == 200
        assert before.json() == {"revision": 0}
        assert before.headers["cache-control"] == "no-store"
        assert client.delete("/v1/account/data", headers=HEADERS).status_code == 200
        assert client.get("/v1/account/data-revision", headers=HEADERS).json() == {"revision": 1}
        assert client.get("/v1/account/data-revision", headers={
            "X-JournalPulse-User": OTHER,
        }).json() == {"revision": 0}
        body = {"client_request_id": str(uuid4()), "text": "Fictional new post-erase writing."}
        assert client.post("/v1/journal/entries", headers=HEADERS, json=body).status_code == 409
        headers = {**HEADERS, "X-JournalPulse-Data-Revision": "1"}
        saved = client.post("/v1/journal/entries", headers=headers, json=body)
        assert saved.status_code == 201
        assert client.post("/v1/journal/entries", headers=headers, json=body).json() == saved.json()
        # Deletion remains available even to an old client after earlier clears.
        assert client.delete("/v1/account/data", headers=HEADERS).status_code == 200
        assert client.post("/v1/journal/entries", headers=headers, json=body).status_code == 409
        assert client.get("/v1/export", headers=HEADERS).json()["journal_entries"] == []


def test_cors_allows_client_observed_revision_header(tmp_path):
    with TestClient(create_app(settings=settings(tmp_path))) as client:
        response = client.options("/v1/journal/entries", headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "authorization,content-type,x-journalpulse-data-revision",
        })
        assert response.status_code == 200
        assert "x-journalpulse-data-revision" in response.headers["access-control-allow-headers"].lower()
