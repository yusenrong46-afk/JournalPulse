"""Reproduce lost-request identity and signed usage-limit boundary failures."""

import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import httpx
import pytest
from fastapi.testclient import TestClient

from journalpulse.api import create_app
from journalpulse.persistence import SQLiteRepository, StorageUnavailable
from journalpulse.signing import sign_text
from test_conversations_api import USER_A, ScriptedClient, chat_settings, choose_goal, start
from test_supabase_repository import KEY, USER_ID, repository


@pytest.mark.parametrize("changed", [
    {"retain_text": False}, {"llm_consent": False}, {"locale": "US"},
])
def test_creation_id_cannot_reuse_different_privacy_settings(tmp_path: Path, changed: dict):
    with TestClient(create_app(settings=chat_settings(tmp_path))) as client:
        request_id = str(uuid4())
        original = start(client, client_request_id=request_id, retain_text=True)
        response = client.post("/v1/conversations", headers={"X-JournalPulse-User": USER_A}, json={
            "client_request_id": request_id, "retain_text": True, "llm_consent": True,
            "locale": "CA", **changed,
        })
        assert response.status_code == 409
        current = client.get(f"/v1/conversations/{request_id}", headers={"X-JournalPulse-User": USER_A})
        assert current.json()["conversation"] == original


@pytest.mark.parametrize("changed", [
    {"text": "Different writing."}, {"mood_score": 4},
    {"goal": "settle"}, {"confirmed_feelings": ["calm"]},
])
def test_turn_id_cannot_reuse_changed_text_or_reported_inputs(tmp_path: Path, changed: dict):
    model = ScriptedClient([False])
    with TestClient(create_app(settings=chat_settings(tmp_path), conversation_client=model)) as client:
        chat = start(client, retain_text=True)
        body = {"client_message_id": str(uuid4()), "text": "The meeting was difficult.", "mood_score": 2}
        url = f"/v1/conversations/{chat['id']}/messages"
        headers = {"X-JournalPulse-User": USER_A}
        first = client.post(url, headers=headers, json=body)
        assert first.status_code == 200
        assert client.post(url, headers=headers, json=body).status_code == 200
        conflicting = client.post(url, headers=headers, json={**body, **changed})
        assert conflicting.status_code == 409
        assert len(client.get(f"/v1/conversations/{chat['id']}", headers=headers).json()["messages"]) == 2
    assert len(model.calls) == 1


def test_purged_turn_replay_is_refused_without_generation(tmp_path: Path):
    model = ScriptedClient([False])
    with TestClient(create_app(settings=chat_settings(tmp_path), conversation_client=model)) as client:
        chat = start(client, retain_text=False)
        headers = {"X-JournalPulse-User": USER_A}
        body = {"client_message_id": str(uuid4()), "text": "The meeting was difficult."}
        url = f"/v1/conversations/{chat['id']}/messages"
        assert client.post(url, headers=headers, json=body).status_code == 200
        assert client.post(f"/v1/conversations/{chat['id']}/close", headers=headers).status_code == 200
        replay = client.post(url, headers=headers, json=body)
        assert replay.status_code == 409
        assert "no longer retained" in replay.json()["detail"]
    assert len(model.calls) == 1


def test_shared_limit_parameters_are_signed_instead_of_caller_controlled(tmp_path: Path):
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/rest/v1/rpc/jp_consume_rate_limit_v2"
        args = json.loads(request.content)
        body = json.loads(args["payload"])
        assert args["signature"] == sign_text(args["payload"], KEY)
        assert body["purpose"] == "consume_rate_limit"
        assert body["user_id"] == str(USER_ID)
        assert (body["bucket"], body["max_events"], body["window_seconds"]) == ("generation", 3, 60)
        return httpx.Response(200, json={"allowed": False, "retry_after": 15})

    assert repository(tmp_path, handler).consume_rate_limit(
        USER_ID, "generation", limit=3, window_seconds=60, now=datetime.now(UTC),
    ) == (False, 15)


def test_missing_signed_limit_migration_fails_closed(tmp_path: Path):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"code": "PGRST202", "message": "Function not found"})

    with pytest.raises(StorageUnavailable):
        repository(tmp_path, handler).consume_rate_limit(
            USER_ID, "generation", limit=20, window_seconds=60, now=datetime.now(UTC),
        )


def test_lost_commit_response_does_not_claim_nothing_was_saved(tmp_path: Path):
    class LostResponseRepository(SQLiteRepository):
        lose_response = True

        def save_journal_entry(self, entry):
            saved = super().save_journal_entry(entry)
            if self.lose_response:
                self.lose_response = False
                raise StorageUnavailable("Response was lost after commit")
            return saved

    settings = chat_settings(tmp_path)
    storage = LostResponseRepository(settings.database_path)
    with TestClient(create_app(settings=settings, repository_factory=lambda _: storage)) as client:
        body = {"client_request_id": str(uuid4()), "text": "Fictional writing."}
        headers = {"X-JournalPulse-User": USER_A}
        lost = client.post("/v1/journal/entries", headers=headers, json=body)
        assert lost.status_code == 503
        assert "could not confirm" in lost.json()["detail"]
        retry = client.post("/v1/journal/entries", headers=headers, json=body)
        assert retry.status_code == 201
        assert retry.json()["id"] == body["client_request_id"]
        assert len(client.get("/v1/journal/entries", headers=headers).json()["items"]) == 1


def test_acceptance_id_cannot_link_another_conversations_reflection(tmp_path: Path):
    with TestClient(create_app(settings=chat_settings(tmp_path))) as client:
        first, second = start(client), start(client)
        one = choose_goal(client, first["id"])
        two = choose_goal(client, second["id"])
        headers = {"X-JournalPulse-User": USER_A}
        request_id = str(uuid4())
        body = {"client_request_id": request_id, "action_id": one["conversation"]["card"]["actions"][0]["id"]}
        saved = client.post(f"/v1/conversations/{first['id']}/accept", headers=headers, json=body)
        assert saved.status_code == 201
        response = client.post(f"/v1/conversations/{second['id']}/accept", headers=headers, json={
            **body, "action_id": two["conversation"]["card"]["actions"][0]["id"],
        })
        assert response.status_code == 409
        current = client.get(f"/v1/conversations/{second['id']}", headers=headers).json()["conversation"]
        assert current["status"] == "open"
        assert current["reflection_id"] is None


def test_acceptance_retry_rejects_a_different_action(tmp_path: Path):
    with TestClient(create_app(settings=chat_settings(tmp_path))) as client:
        chat = start(client)
        turn = choose_goal(client, chat["id"])
        actions = turn["conversation"]["card"]["actions"]
        assert len(actions) > 1
        headers = {"X-JournalPulse-User": USER_A}
        body = {"client_request_id": str(uuid4()), "action_id": actions[0]["id"]}
        url = f"/v1/conversations/{chat['id']}/accept"
        assert client.post(url, headers=headers, json=body).status_code == 201
        assert client.post(url, headers=headers, json=body).status_code == 201
        changed = client.post(url, headers=headers, json={**body, "action_id": actions[1]["id"]})
        assert changed.status_code == 409


def test_acceptance_retry_rejects_changed_effective_self_report(tmp_path: Path):
    with TestClient(create_app(settings=chat_settings(tmp_path))) as client:
        chat = start(client)
        turn = choose_goal(client, chat["id"])
        headers = {"X-JournalPulse-User": USER_A}
        state = {"valence": -.2, "arousal": .4, "agency": .5, "emotion_tags": ["tired"]}
        body = {
            "client_request_id": str(uuid4()), "action_id": turn["conversation"]["card"]["actions"][0]["id"],
            "self_report": state,
        }
        url = f"/v1/conversations/{chat['id']}/accept"
        assert client.post(url, headers=headers, json=body).status_code == 201
        assert client.post(url, headers=headers, json=body).status_code == 201
        assert client.post(url, headers=headers, json={
            **body, "self_report": {**state, "valence": .5},
        }).status_code == 409
