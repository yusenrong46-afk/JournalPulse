"""Lifecycle invariants under interleaving: a slow reply must never outrun a close,
delete, accept, or a reply committed by another server instance."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

import httpx
import pytest
from fastapi.testclient import TestClient

from journalpulse.api import create_app
from journalpulse.config import Settings
from journalpulse.domain import ModelRun
from journalpulse.intelligence import ConversationCompletion
from journalpulse.persistence import SQLiteRepository, StorageUnavailable

USER = "00000000-0000-4000-8000-000000000001"
HEADERS = {"X-JournalPulse-User": USER}


def chat_settings(tmp_path: Path, **overrides: object) -> Settings:
    root = Path(__file__).resolve().parents[1]
    configured = Settings(
        environment="test",
        database_path=tmp_path / "beta.db",
        resource_catalog_path=root / "assets" / "resources" / "catalog.json",
        openrouter_api_key="test-only-key",
        openrouter_model="openai/gpt-6-luna",
        openrouter_base_url="https://openrouter.ai/api/v1",
        openrouter_zdr=True,
        openrouter_timeout_seconds=2,
        supabase_url=None,
        supabase_anon_key=None,
        raw_text_retention_default=False,
        analysis_rate_limit_per_minute=50,
    )
    if overrides:
        configured = Settings(**{**configured.__dict__, **overrides})
    return configured


def completion(
    reply: str = "Tell me more.", *, offer: bool = False, feelings: tuple[str, ...] = ()
) -> ConversationCompletion:
    return ConversationCompletion(
        reply=reply,
        offer_action=offer,
        resource_intent="reflect",
        card_reason="A reviewed option." if offer else "",
        summary="A hard day.",
        feelings=feelings,
        model_run=ModelRun(model="openai/gpt-6-luna", latency_ms=5, schema_valid=True),
    )


class GatedClient:
    """Replies immediately, except for the turn that is told to wait for a release."""

    def __init__(self) -> None:
        self.calls = 0
        self.started = threading.Event()
        self.release = threading.Event()
        self.block_next = False

    def complete(self, messages: list[dict[str, str]]) -> ConversationCompletion:
        self.calls += 1
        if self.block_next:
            self.block_next = False
            self.started.set()
            assert self.release.wait(10), "test never released the model"
        return completion(f"reply {self.calls}", feelings=("tired",))


def start(client: TestClient, **overrides: object) -> dict:
    body = {"llm_consent": True, "retain_text": False, "locale": "CA", **overrides}
    response = client.post("/v1/conversations", json=body, headers=HEADERS)
    assert response.status_code == 201, response.text
    return response.json()


def say(client: TestClient, conversation_id: str, text: str, **extra: object) -> httpx.Response:
    body = {"client_message_id": str(uuid4()), "text": text, **extra}
    return client.post(f"/v1/conversations/{conversation_id}/messages", json=body, headers=HEADERS)


def detail(client: TestClient, conversation_id: str) -> dict:
    response = client.get(f"/v1/conversations/{conversation_id}", headers=HEADERS)
    assert response.status_code == 200, response.text
    return response.json()


def slow_turn(client: TestClient, model: GatedClient, conversation_id: str, text: str):
    model.block_next = True
    pool = ThreadPoolExecutor(max_workers=1)
    future = pool.submit(say, client, conversation_id, text)
    assert model.started.wait(10), "slow turn never reached the model"
    return pool, future


def test_a_reply_that_arrives_after_close_is_rejected_and_restores_nothing(tmp_path: Path):
    model = GatedClient()
    app = create_app(settings=chat_settings(tmp_path), conversation_client=model)
    with TestClient(app) as client, TestClient(app) as other:
        chat = start(client)
        assert say(client, chat["id"], "First thing on my mind.").status_code == 200
        pool, pending = slow_turn(client, model, chat["id"], "Something private and slow.")
        closed = other.post(f"/v1/conversations/{chat['id']}/close", headers=HEADERS)
        assert closed.status_code == 200
        model.release.set()
        late = pending.result(timeout=10)
        pool.shutdown()
        assert late.status_code == 409
        assert late.json()["detail"] == "This conversation is closed."
        stored = detail(client, chat["id"])
        assert stored["conversation"]["status"] == "closed"
        assert len(stored["messages"]) == 2
        assert all(message["content"] is None for message in stored["messages"])


def test_a_reply_that_arrives_after_delete_recreates_nothing(tmp_path: Path):
    model = GatedClient()
    app = create_app(settings=chat_settings(tmp_path), conversation_client=model)
    with TestClient(app) as client, TestClient(app) as other:
        chat = start(client)
        pool, pending = slow_turn(client, model, chat["id"], "Delete me while you think.")
        assert other.delete(f"/v1/conversations/{chat['id']}", headers=HEADERS).status_code == 204
        model.release.set()
        late = pending.result(timeout=10)
        pool.shutdown()
        assert late.status_code == 404
        assert client.get(f"/v1/conversations/{chat['id']}", headers=HEADERS).status_code == 404
        exported = client.get("/v1/export", headers=HEADERS).json()
        assert exported["conversations"] == []
        assert exported["conversation_messages"] == []


def test_a_reply_that_arrives_after_accept_cannot_reopen_the_chat(tmp_path: Path):
    model = GatedClient()
    app = create_app(settings=chat_settings(tmp_path), conversation_client=model)
    with TestClient(app) as client, TestClient(app) as other:
        chat = start(client)
        say(client, chat["id"], "A long week.")
        card = say(client, chat["id"], "I'd like to calm down.", goal="settle", confirmed_feelings=["tired"])
        action = card.json()["conversation"]["card"]["decision_preview"]["action_id"]
        pool, pending = slow_turn(client, model, chat["id"], "One more thought.")
        accepted = other.post(
            f"/v1/conversations/{chat['id']}/accept",
            json={"client_request_id": str(uuid4()), "action_id": action},
            headers=HEADERS,
        )
        assert accepted.status_code == 201, accepted.text
        model.release.set()
        late = pending.result(timeout=10)
        pool.shutdown()
        assert late.status_code == 409
        stored = detail(client, chat["id"])
        assert stored["conversation"]["status"] == "closed"
        assert stored["conversation"]["reflection_id"] == accepted.json()["id"]
        assert len(client.get("/v1/reflections", headers=HEADERS).json()["items"]) == 1


def test_a_second_instance_cannot_overwrite_a_newer_turn(tmp_path: Path):
    """Two app instances share the database but not the in-process lock."""
    slow_model = GatedClient()
    fast_model = GatedClient()
    first = create_app(settings=chat_settings(tmp_path), conversation_client=slow_model)
    second = create_app(settings=chat_settings(tmp_path), conversation_client=fast_model)
    with TestClient(first) as slow_client, TestClient(second) as fast_client:
        chat = start(slow_client)
        pool, pending = slow_turn(slow_client, slow_model, chat["id"], "I started first.")
        winner = say(fast_client, chat["id"], "I finished first.")
        assert winner.status_code == 200
        assert winner.json()["conversation"]["revision"] == 1
        slow_model.release.set()
        stale = pending.result(timeout=10)
        pool.shutdown()
        assert stale.status_code == 409
        assert "was not saved" in stale.json()["detail"]
        stored = detail(fast_client, chat["id"])
        assert stored["conversation"]["status"] == "open"
        assert stored["conversation"]["revision"] == 1
        assert [message["content"] for message in stored["messages"]] == ["I finished first.", "reply 1"]
        again = say(slow_client, chat["id"], "Trying again.")
        assert again.status_code == 200
        assert again.json()["conversation"]["revision"] == 2


def test_repeated_close_and_stale_sweeps_never_regress_status(tmp_path: Path):
    model = GatedClient()
    app = create_app(settings=chat_settings(tmp_path), conversation_client=model)
    with TestClient(app) as client:
        chat = start(client)
        say(client, chat["id"], "Hello.")
        first = client.post(f"/v1/conversations/{chat['id']}/close", headers=HEADERS).json()
        second = client.post(f"/v1/conversations/{chat['id']}/close", headers=HEADERS).json()
        assert first["status"] == second["status"] == "closed"
        assert second["revision"] == first["revision"]
        blocked = say(client, chat["id"], "Can I keep going?")
        assert blocked.status_code == 409
        assert detail(client, chat["id"])["conversation"]["status"] == "closed"


def accept_ready(client: TestClient, **start_overrides: object) -> tuple[dict, str]:
    chat = start(client, **start_overrides)
    say(client, chat["id"], "Work was a lot.", mood_score=2)
    card = say(client, chat["id"], "Calm down.", goal="settle", confirmed_feelings=["tired", "anxious"])
    assert card.status_code == 200, card.text
    return chat, card.json()["conversation"]["card"]["decision_preview"]["action_id"]


def test_accept_is_idempotent_for_one_request_and_exclusive_across_requests(tmp_path: Path):
    app = create_app(settings=chat_settings(tmp_path), conversation_client=GatedClient())
    with TestClient(app) as client:
        chat, action = accept_ready(client)
        request_id = str(uuid4())
        body = {"client_request_id": request_id, "action_id": action}
        first = client.post(f"/v1/conversations/{chat['id']}/accept", json=body, headers=HEADERS)
        retry = client.post(f"/v1/conversations/{chat['id']}/accept", json=body, headers=HEADERS)
        assert first.status_code == retry.status_code == 201
        assert first.json() == retry.json()
        other = client.post(
            f"/v1/conversations/{chat['id']}/accept",
            json={"client_request_id": str(uuid4()), "action_id": action},
            headers=HEADERS,
        )
        assert other.status_code == 409
        assert other.json()["detail"] == "This conversation already has a saved choice."
        assert len(client.get("/v1/reflections", headers=HEADERS).json()["items"]) == 1


def test_concurrent_accepts_from_two_instances_save_exactly_one(tmp_path: Path):
    first = create_app(settings=chat_settings(tmp_path), conversation_client=GatedClient())
    second = create_app(settings=chat_settings(tmp_path), conversation_client=GatedClient())
    with TestClient(first) as one, TestClient(second) as two:
        chat, action = accept_ready(one)
        barrier = threading.Barrier(2)

        def accept(client: TestClient) -> httpx.Response:
            barrier.wait()
            return client.post(
                f"/v1/conversations/{chat['id']}/accept",
                json={"client_request_id": str(uuid4()), "action_id": action},
                headers=HEADERS,
            )

        with ThreadPoolExecutor(max_workers=2) as pool:
            results = [
                future.result(timeout=10) for future in [pool.submit(accept, one), pool.submit(accept, two)]
            ]
        assert sorted(result.status_code for result in results) == [201, 409]
        assert len(one.get("/v1/reflections", headers=HEADERS).json()["items"]) == 1
        winner = next(result.json() for result in results if result.status_code == 201)
        assert detail(one, chat["id"])["conversation"]["reflection_id"] == winner["id"]


def test_a_failure_midway_through_accept_leaves_no_partial_state(tmp_path: Path, monkeypatch):
    app = create_app(settings=chat_settings(tmp_path), conversation_client=GatedClient())
    with TestClient(app, raise_server_exceptions=False) as client:
        chat, action = accept_ready(client)
        original = SQLiteRepository._close_locked

        def explode(self, *args, **kwargs):
            raise RuntimeError("injected failure after the reflection insert")

        monkeypatch.setattr(SQLiteRepository, "_close_locked", explode)
        body = {"client_request_id": str(uuid4()), "action_id": action}
        failed = client.post(f"/v1/conversations/{chat['id']}/accept", json=body, headers=HEADERS)
        assert failed.status_code == 500
        assert client.get("/v1/reflections", headers=HEADERS).json()["items"] == []
        stored = detail(client, chat["id"])
        assert stored["conversation"]["status"] == "open"
        assert stored["conversation"]["reflection_id"] is None
        assert all(message["content"] for message in stored["messages"])
        monkeypatch.setattr(SQLiteRepository, "_close_locked", original)
        retried = client.post(f"/v1/conversations/{chat['id']}/accept", json=body, headers=HEADERS)
        assert retried.status_code == 201


@pytest.mark.parametrize(("retain", "kept"), [(False, False), (True, True)])
def test_accept_purges_or_keeps_text_as_the_person_chose(tmp_path: Path, retain: bool, kept: bool):
    app = create_app(settings=chat_settings(tmp_path), conversation_client=GatedClient())
    with TestClient(app) as client:
        chat, action = accept_ready(client, retain_text=retain)
        accepted = client.post(
            f"/v1/conversations/{chat['id']}/accept",
            json={"client_request_id": str(uuid4()), "action_id": action},
            headers=HEADERS,
        )
        assert accepted.status_code == 201
        contents = [message["content"] for message in detail(client, chat["id"])["messages"]]
        assert all(contents) if kept else not any(contents)


def test_confirmed_feelings_and_mood_survive_reload_and_decide_the_saved_state(tmp_path: Path):
    app = create_app(settings=chat_settings(tmp_path), conversation_client=GatedClient())
    with TestClient(app) as client:
        chat, action = accept_ready(client)
        reloaded = detail(client, chat["id"])["conversation"]
        assert reloaded["feelings"] == ["tired"]
        assert reloaded["confirmed_feelings"] == ["tired", "anxious"]
        assert reloaded["reported_mood"] == 2
        forged = {"valence": 0.9, "arousal": 0.1, "agency": 1, "emotion_tags": ["happy"], "confidence": 1}
        saved = client.post(
            f"/v1/conversations/{chat['id']}/accept",
            json={"client_request_id": str(uuid4()), "action_id": action, "self_report": forged},
            headers=HEADERS,
        ).json()
        assert saved["self_report_input"] == {"feelings": ["tired", "anxious"], "mood_score": 2}
        assert saved["state"]["emotion_tags"] == ["tired", "anxious"]
        assert saved["state"]["derivation"] == "feeling-buttons-v1"
        assert saved["state"]["confidence"] is None
        assert saved["state"]["valence"] < 0


def test_a_later_mood_tap_does_not_replace_the_first(tmp_path: Path):
    app = create_app(settings=chat_settings(tmp_path), conversation_client=GatedClient())
    with TestClient(app) as client:
        chat = start(client)
        say(client, chat["id"], "Rough.", mood_score=1)
        say(client, chat["id"], "Actually great.", mood_score=5)
        assert detail(client, chat["id"])["conversation"]["reported_mood"] == 1


def test_unknown_confirmed_feelings_are_rejected(tmp_path: Path):
    app = create_app(settings=chat_settings(tmp_path), conversation_client=GatedClient())
    with TestClient(app) as client:
        chat = start(client)
        response = say(client, chat["id"], "Calm.", goal="settle", confirmed_feelings=["depressed"])
        assert response.status_code == 422


def test_support_mode_never_reuses_an_earlier_ordinary_reply(tmp_path: Path):
    model = GatedClient()
    app = create_app(settings=chat_settings(tmp_path), conversation_client=model)
    with TestClient(app) as client:
        chat = start(client)
        first = say(client, chat["id"], "Long day at work.").json()
        say(client, chat["id"], "Still tired.")
        entered = say(client, chat["id"], "I want to kill myself tonight.").json()
        follow_up = say(client, chat["id"], "Can we just talk normally?").json()
        goal = say(client, chat["id"], "Calm down.", goal="settle").json()
        assert model.calls == 2
        support_text = entered["conversation"]["safety"]["support_message"]
        assert support_text.startswith("If you may act on thoughts of suicide")
        for turn in (entered, follow_up, goal):
            assert turn["assistant_message"]["content"] == support_text
            assert turn["assistant_message"]["content"] != first["assistant_message"]["content"]
            assert turn["assistant_message"]["model_run"]["model"] == "safety-router"
            assert turn["conversation"]["safety_mode"] == "support"
        card = goal["conversation"]["card"]
        assert card["resource_intent"] == "pause"
        assert all(item["resource_type"] == "support" for item in card["actions"])
        ordinary = client.post(
            f"/v1/conversations/{chat['id']}/accept",
            json={"client_request_id": str(uuid4()), "action_id": "site_nhs_breathing"},
            headers=HEADERS,
        )
        assert ordinary.status_code == 422


def test_rate_limit_is_shared_by_every_instance(tmp_path: Path):
    tight = chat_settings(tmp_path, analysis_rate_limit_per_minute=2)
    one = create_app(settings=tight, conversation_client=GatedClient())
    two = create_app(settings=tight, conversation_client=GatedClient())
    with TestClient(one) as first, TestClient(two) as second:
        chat = start(first)
        assert say(first, chat["id"], "One.").status_code == 200
        assert say(second, chat["id"], "Two.").status_code == 200
        refused = say(second, chat["id"], "Three.")
        assert refused.status_code == 429
        assert 1 <= int(refused.headers["Retry-After"]) <= 60


def test_rate_limit_window_boundary(tmp_path: Path):
    repository = SQLiteRepository(tmp_path / "limit.db")
    user = uuid4()
    start_at = datetime(2026, 9, 28, 12, 0, tzinfo=UTC)
    for second in range(3):
        allowed, _ = repository.consume_rate_limit(
            user, "generation", limit=3, window_seconds=60, now=start_at + timedelta(seconds=second)
        )
        assert allowed
    allowed, retry_after = repository.consume_rate_limit(
        user, "generation", limit=3, window_seconds=60, now=start_at + timedelta(seconds=30)
    )
    assert (allowed, retry_after) == (False, 30)
    allowed, _ = repository.consume_rate_limit(
        user, "generation", limit=3, window_seconds=60, now=start_at + timedelta(seconds=60)
    )
    assert allowed
    other, _ = repository.consume_rate_limit(uuid4(), "generation", limit=3, window_seconds=60, now=start_at)
    assert other


def test_rate_limit_fails_closed_when_the_shared_counter_is_unavailable(tmp_path: Path, monkeypatch):
    def unavailable(self, *args, **kwargs):
        raise StorageUnavailable("counter down")

    monkeypatch.setattr(SQLiteRepository, "consume_rate_limit", unavailable)
    model = GatedClient()
    app = create_app(settings=chat_settings(tmp_path), conversation_client=model)
    with TestClient(app) as client:
        chat = start(client)
        refused = say(client, chat["id"], "Hello.")
        assert refused.status_code == 503
        assert model.calls == 0


def test_scheduled_purge_closes_idle_chats_and_keeps_retained_text(tmp_path: Path):
    clock = {"now": datetime(2026, 9, 28, 9, 0, tzinfo=UTC)}
    app = create_app(
        settings=chat_settings(tmp_path),
        conversation_client=GatedClient(),
        clock=lambda: clock["now"],
    )
    with TestClient(app) as client:
        temporary = start(client)
        say(client, temporary["id"], "Temporary words.")
        kept = start(client, retain_text=True)
        say(client, kept["id"], "Words to keep.")
    repository = SQLiteRepository(tmp_path / "beta.db")
    assert repository.purge_expired_conversations(now=clock["now"] + timedelta(hours=23)) == {
        "closed": 0,
        "purged_messages": 0,
    }
    result = repository.purge_expired_conversations(now=clock["now"] + timedelta(hours=25))
    assert result == {"closed": 2, "purged_messages": 2}
    user = uuid4().__class__(USER)
    temporary_messages = repository.list_messages(user, uuid4().__class__(temporary["id"]))
    kept_messages = repository.list_messages(user, uuid4().__class__(kept["id"]))
    assert all(message.content is None for message in temporary_messages)
    assert all(message.content for message in kept_messages)
