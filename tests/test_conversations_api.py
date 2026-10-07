import json
import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from journalpulse.api import create_app
from journalpulse.config import Settings
from journalpulse.domain import ModelRun
from journalpulse.intelligence import (
    ConversationCompletion,
    ConversationProviderError,
    UnsupportedProviderResponse,
    deterministic_reflection,
)

USER_A = "00000000-0000-4000-8000-000000000001"
USER_B = "00000000-0000-4000-8000-000000000002"


def chat_settings(tmp_path: Path, **overrides: object) -> Settings:
    root = Path(__file__).resolve().parents[1]
    configured = Settings(
        environment="test",
        database_path=tmp_path / "beta.db",
        resource_catalog_path=root / "assets" / "resources" / "catalog.json",
        openrouter_api_key="test-only-key",
        openrouter_model="openai/gpt-5.4-mini",
        openrouter_base_url="https://openrouter.ai/api/v1",
        openrouter_zdr=True,
        openrouter_timeout_seconds=2,
        supabase_url=None,
        supabase_anon_key=None,
        raw_text_retention_default=False,
        analysis_rate_limit_per_minute=20,
    )
    if overrides:
        configured = Settings(**{**configured.__dict__, **overrides})
    return configured


@pytest.mark.parametrize("enabled,expected", [(False, "unavailable"), (True, "configured")])
def test_discovery_capability_never_calls_a_provider(tmp_path: Path, enabled: bool, expected: str):
    model = ScriptedClient()
    configured = chat_settings(tmp_path, search_feature_enabled=enabled, search_api_key="test-only")
    with TestClient(create_app(settings=configured, conversation_client=model)) as client:
        response = client.get("/v1/capabilities")
        assert response.status_code == 200
        assert response.json() == {"discovery": expected}
        assert response.headers["cache-control"] == "no-store"
        assert model.calls == []


def test_reviewed_activity_browse_obeys_constraints_without_ai(tmp_path: Path):
    from journalpulse.activity_resources import ActivityConstraints, activity_resource_matches_constraints

    with TestClient(create_app(settings=chat_settings(tmp_path))) as client:
        response = client.get(
            "/v1/activity-resources?time_minutes=2&no_audio=true&no_video=true&seated=true&avoid_breath_focus=true",
            headers={"X-JournalPulse-User": USER_A},
        )
        assert response.status_code == 200, response.text
        items = response.json()["items"]
        assert items
        constraints = ActivityConstraints(
            time_minutes=2, no_audio=True, no_video=True, seated=True, avoid_breath_focus=True,
        )
        assert all(activity_resource_matches_constraints(item, constraints) for item in items)
        assert all(item["resource_type"] != "support" for item in items)


def test_closed_choice_detail_recovers_only_the_owned_reflection(tmp_path: Path):
    with TestClient(create_app(settings=chat_settings(tmp_path, llm_feature_enabled=False))) as client:
        conversation = start(client, llm_consent=False)
        turn = choose_goal(client, conversation["id"])
        chosen = turn["conversation"]["card"]["actions"][0]["id"]
        saved = client.post(
            f"/v1/conversations/{conversation['id']}/accept",
            headers={"X-JournalPulse-User": USER_A},
            json={"client_request_id": str(uuid4()), "action_id": chosen},
        )
        assert saved.status_code == 201, saved.text
        own = client.get(
            f"/v1/conversations/{conversation['id']}", headers={"X-JournalPulse-User": USER_A},
        )
        assert own.json()["conversation"]["status"] == "closed"
        assert own.json()["accepted_reflection"] == saved.json()
        outsider = client.get(
            f"/v1/conversations/{conversation['id']}", headers={"X-JournalPulse-User": USER_B},
        )
        assert outsider.status_code == 404


def completion(*, offer: bool) -> ConversationCompletion:
    return ConversationCompletion(
        reply="That still sounds unsettled. What would make the next hour a little easier?",
        offer_action=offer,
        resource_intent="reflect",
        card_reason="A short reviewed pause matches what you described." if offer else "",
        summary="The meeting is still unresolved.",
        model_run=ModelRun(
            model="openai/gpt-6-luna",
            provider="openrouter",
            latency_ms=12,
            schema_valid=True,
            prompt_version="2026-09-24.1",
        ),
    )


class ScriptedClient:
    def __init__(self, offers: list[bool] | None = None) -> None:
        self.offers = offers or [False, True]
        self.calls: list[list[dict[str, str]]] = []

    def complete(self, messages: list[dict[str, str]]) -> ConversationCompletion:
        self.calls.append(messages)
        offer = self.offers[min(len(self.calls) - 1, len(self.offers) - 1)]
        return completion(offer=offer)


def start(client: TestClient, user: str = USER_A, **overrides: object) -> dict:
    payload = {"llm_consent": True, "retain_text": False, "locale": "CA"}
    payload.update(overrides)
    response = client.post("/v1/conversations", json=payload, headers={"X-JournalPulse-User": user})
    assert response.status_code == 201, response.text
    return response.json()


def say(
    client: TestClient,
    conversation_id: str,
    text: str,
    user: str = USER_A,
    message_id: str | None = None,
    goal: str | None = None,
):
    body: dict[str, object] = {"client_message_id": message_id or str(uuid4()), "text": text}
    if goal is not None:
        body["goal"] = goal
    return client.post(
        f"/v1/conversations/{conversation_id}/messages",
        headers={"X-JournalPulse-User": user},
        json=body,
    )


def choose_goal(client: TestClient, conversation_id: str, goal: str = "understand", user: str = USER_A):
    response = say(client, conversation_id, f"I'd like help to {goal}.", user=user, goal=goal)
    assert response.status_code == 200, response.text
    return response.json()


def test_conversation_accepts_a_catalog_card_and_checks_in(tmp_path: Path):
    model = ScriptedClient()
    app = create_app(settings=chat_settings(tmp_path), conversation_client=model)
    with TestClient(app) as client:
        conversation = start(client)
        first = say(client, conversation["id"], "The meeting is still replaying in my head.")
        assert first.status_code == 200, first.text
        assert first.json()["conversation"]["card"] is None
        assert first.json()["conversation"]["ready_for_action"] is False
        second = say(client, conversation["id"], "I think I could try one small thing.")
        assert second.status_code == 200, second.text
        assert second.json()["conversation"]["ready_for_action"] is True
        assert second.json()["conversation"]["card"] is None
        turn = choose_goal(client, conversation["id"], "understand")
        assert len(model.calls) == 2
        card = turn["conversation"]["card"]
        assert turn["assistant_message"]["model_run"]["model"] == "luna-guided"
        assert card["goal"] == "understand"
        assert card["resource_intent"] == "read"
        assert card["actions"]
        assert len(card["actions"]) <= 3
        action_id = card["decision_preview"]["action_id"]
        saved = client.post(
            f"/v1/conversations/{conversation['id']}/accept",
            headers={"X-JournalPulse-User": USER_A},
            json={
                "action_id": action_id,
                "decision": {"policy_name": "forged-policy", "action_id": "forged"},
                "model_run": {"model": "forged-model"},
                "self_report": {
                    "valence": -0.2,
                    "arousal": 0.4,
                    "agency": 0.55,
                    "emotion_tags": ["tense"],
                    "confidence": 0.42,
                },
            },
        )
        assert saved.status_code == 201, saved.text
        record = saved.json()
        assert record["context"]["source"] == "conversation"
        assert record["context"]["conversation_id"] == conversation["id"]
        assert record["text"] is None
        assert record["state"]["confidence"] == 0.42
        assert record["decision"]["policy_name"] == "fixed-baseline"
        assert record["decision"]["action_id"] == action_id
        assert record["decision"]["selection_source"] == "policy_accepted"
        assert record["target"]["goal"] == "understand"
        assert record["model_run"]["model"] == "openai/gpt-6-luna"
        assert record["reflection"]["reflection_question"] == "What changed after you tried it?"
        outcome = client.post(
            "/v1/outcomes",
            headers={"X-JournalPulse-User": USER_A},
            json={
                "decision_id": record["decision"]["decision_id"],
                "completed": True,
                "post_state": {
                    "valence": 0.1,
                    "arousal": 0.3,
                    "agency": 0.6,
                    "emotion_tags": ["steadier"],
                    "confidence": 0.5,
                },
                "helpfulness": 4,
            },
        )
        assert outcome.status_code == 201, outcome.text
        detail = client.get(
            f"/v1/conversations/{conversation['id']}",
            headers={"X-JournalPulse-User": USER_A},
        )
        assert detail.status_code == 200
        assert detail.json()["conversation"]["status"] == "closed"
        assert all(message["content"] is None for message in detail.json()["messages"])


def test_provider_failures_persist_nothing(tmp_path: Path):
    class Exploding:
        def __init__(self, error: Exception) -> None:
            self.error = error
            self.calls = 0

        def complete(self, messages: list[dict[str, str]]) -> ConversationCompletion:
            del messages
            self.calls += 1
            raise self.error

    for error, status in (
        (UnsupportedProviderResponse("bad shape"), 502),
        (ConversationProviderError("The model reply was cut off. Nothing was saved."), 502),
        (
            ConversationProviderError(
                "The model reply did not match the conversation schema. Nothing was saved."
            ),
            502,
        ),
    ):
        model = Exploding(error)
        app = create_app(settings=chat_settings(tmp_path), conversation_client=model)
        with TestClient(app) as client:
            conversation = start(client)
            response = say(client, conversation["id"], "Can we talk this through?")
            assert response.status_code == status
            detail = client.get(
                f"/v1/conversations/{conversation['id']}",
                headers={"X-JournalPulse-User": USER_A},
            )
            assert detail.json()["messages"] == []
            assert model.calls == 1


def test_support_mode_never_calls_the_model_and_stays_there(tmp_path: Path):
    model = ScriptedClient()
    app = create_app(settings=chat_settings(tmp_path), conversation_client=model)
    sentences = (
        "I have a suicide plan",
        "I might act on these thoughts",
        "I am not suicidal. I plan to kill myself tonight.",
        "I don't want to die, but I will overdose tonight.",
    )
    with TestClient(app) as client:
        for sentence in sentences:
            conversation = start(client)
            response = say(client, conversation["id"], sentence)
            assert response.status_code == 200, response.text
            body = response.json()
            assert body["conversation"]["safety_mode"] == "support"
            assert "9" in body["assistant_message"]["content"]
            assert body["conversation"]["card"]["decision_preview"]["policy_name"] == "safety-router"
            follow = say(client, conversation["id"], "I am still here.")
            assert follow.status_code == 200
            assert follow.json()["conversation"]["safety_mode"] == "support"
        assert model.calls == []


def test_unsafe_feeling_without_a_time_suffix_bypasses_the_model(tmp_path: Path):
    model = ScriptedClient(offers=[True])
    app = create_app(settings=chat_settings(tmp_path), conversation_client=model)
    with TestClient(app) as client:
        for sentence in ("I don't feel safe.", "I don’t feel safe."):
            conversation = start(client)
            response = say(client, conversation["id"], sentence)
            assert response.status_code == 200, response.text
            body = response.json()
            assert body["conversation"]["safety_mode"] == "support"
            assert body["conversation"]["ready_for_action"] is False
            assert body["assistant_message"]["model_run"]["model"] == "safety-router"
            assert model.calls == []


def test_analysis_and_conversation_share_one_limiter(tmp_path: Path):
    class Counting:
        def __init__(self) -> None:
            self.calls = 0

        def analyze(self, text: str, context: dict[str, str]):
            del context
            self.calls += 1
            return deterministic_reflection(text)

        def complete(self, messages: list[dict[str, str]]) -> ConversationCompletion:
            del messages
            self.calls += 1
            return completion(offer=False)

    model = Counting()
    app = create_app(
        settings=chat_settings(tmp_path, analysis_rate_limit_per_minute=2),
        intelligence_client=model,
        conversation_client=model,
    )
    with TestClient(app) as client:
        conversation = start(client)
        headers = {"X-JournalPulse-User": USER_A}
        analysis = client.post(
            "/v1/reflections/analyze",
            headers=headers,
            json={"text": "The meeting is still on my mind.", "llm_consent": True, "locale": "CA"},
        )
        assert analysis.status_code == 200, analysis.text
        turn = say(client, conversation["id"], "I keep replaying one sentence.")
        assert turn.status_code == 200, turn.text
        blocked = say(client, conversation["id"], "One more thought.")
        assert blocked.status_code == 429
        assert model.calls == 2


def test_retried_message_id_does_not_call_the_model_again(tmp_path: Path):
    model = ScriptedClient([False])
    app = create_app(settings=chat_settings(tmp_path), conversation_client=model)
    with TestClient(app) as client:
        conversation = start(client)
        message_id = str(uuid4())
        first = say(client, conversation["id"], "The same thought, sent once.", message_id=message_id)
        second = say(client, conversation["id"], "The same thought, sent once.", message_id=message_id)
        assert first.status_code == 200
        assert second.status_code == 200
        assert first.json()["assistant_message"]["content"] == second.json()["assistant_message"]["content"]
        assert len(model.calls) == 1


def test_a_second_turn_in_progress_is_rejected(tmp_path: Path):
    started = threading.Event()
    release = threading.Event()

    class Blocking:
        def complete(self, messages: list[dict[str, str]]) -> ConversationCompletion:
            del messages
            started.set()
            assert release.wait(3)
            return completion(offer=False)

    app = create_app(settings=chat_settings(tmp_path), conversation_client=Blocking())
    with TestClient(app) as client:
        conversation = start(client)
        results: dict[str, object] = {}

        def run(name: str) -> None:
            results[name] = say(client, conversation["id"], f"Thought from {name}.")

        first = threading.Thread(target=run, args=("first",))
        first.start()
        assert started.wait(3)
        second = threading.Thread(target=run, args=("second",))
        second.start()
        second.join(3)
        release.set()
        first.join(3)
        statuses = sorted(item.status_code for item in results.values())
        assert statuses == [200, 409]


def test_twenty_first_user_message_is_rejected_without_a_model_call(tmp_path: Path):
    model = ScriptedClient([False])
    app = create_app(settings=chat_settings(tmp_path), conversation_client=model)
    with TestClient(app) as client:
        conversation = start(client)
        for index in range(20):
            response = say(client, conversation["id"], f"Note number {index}.")
            assert response.status_code == 200, response.text
        extra = say(client, conversation["id"], "This one should not be sent.")
        assert extra.status_code == 409
        assert len(model.calls) == 20


def test_guided_luna_runs_without_consent_or_a_model(tmp_path: Path, monkeypatch):
    def forbidden(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("conversation client must not be constructed")

    monkeypatch.setattr(
        "journalpulse.conversations.OpenRouterConversationClient",
        forbidden,
    )
    private_line = "My sister's wedding speech is tomorrow and I'm exhausted and anxious."
    with TestClient(create_app(settings=chat_settings(tmp_path))) as client:
        conversation = start(client, llm_consent=False)
        assert conversation["mode"] == "guided"
        assert conversation["llm_consent"] is False
        first = say(client, conversation["id"], private_line)
        assert first.status_code == 200, first.text
        body = first.json()
        assert body["assistant_message"]["model_run"]["model"] == "luna-guided"
        assert body["conversation"]["feelings"] == ["tired", "anxious"]
        assert body["conversation"]["ready_for_action"] is False
        say(client, conversation["id"], "Mostly in my chest.")
        third = say(client, conversation["id"], "I just want it to go well.")
        assert third.json()["conversation"]["ready_for_action"] is False
        requested = say(client, conversation["id"], "I'd like to find one small thing.")
        assert requested.json()["conversation"]["ready_for_action"] is True
        turn = choose_goal(client, conversation["id"], "settle")
        card = turn["conversation"]["card"]
        assert card["goal"] == "settle"
        assert all(item["resource_type"] != "support" for item in card["actions"])
        saved = client.post(
            f"/v1/conversations/{conversation['id']}/accept",
            headers={"X-JournalPulse-User": USER_A},
            json={
                "action_id": card["decision_preview"]["action_id"],
                "self_report": {
                    "valence": -0.4,
                    "arousal": 0.7,
                    "agency": 0.4,
                    "emotion_tags": ["tired", "anxious"],
                    "confidence": 0.6,
                },
            },
        )
        assert saved.status_code == 201, saved.text
        record = saved.json()
        assert record["target"]["goal"] == "settle"
        assert "wedding" not in json.dumps(record)
    disabled = chat_settings(tmp_path, openrouter_api_key=None, llm_feature_enabled=False)
    with TestClient(create_app(settings=disabled)) as client:
        conversation = start(client, llm_consent=True)
        assert conversation["mode"] == "guided"


def test_a_goal_turn_still_goes_through_the_safety_gate(tmp_path: Path):
    model = ScriptedClient([False])
    app = create_app(settings=chat_settings(tmp_path), conversation_client=model)
    with TestClient(app) as client:
        conversation = start(client)
        response = say(
            client,
            conversation["id"],
            "I want to kill myself tonight.",
            goal="settle",
        )
        assert response.status_code == 200, response.text
        updated = response.json()["conversation"]
        assert updated["safety_mode"] == "support"
        assert updated["card"]["resource_intent"] == "pause"
        assert model.calls == []


@pytest.mark.parametrize("disabled_setting", ["llm_feature_enabled", "openrouter_zdr"])
def test_existing_unlinked_ai_chat_respects_disabled_ai_without_saving_a_turn(
    tmp_path: Path, disabled_setting: str,
):
    model = ScriptedClient([False])
    with TestClient(create_app(settings=chat_settings(tmp_path), conversation_client=model)) as client:
        chat = start(client)
        assert say(client, chat["id"], "A fictional first message.").status_code == 200
        before = client.get(
            f"/v1/conversations/{chat['id']}", headers={"X-JournalPulse-User": USER_A},
        ).json()
    disabled = chat_settings(tmp_path, **{disabled_setting: False})
    with TestClient(create_app(settings=disabled, conversation_client=model)) as client:
        rejected = say(client, chat["id"], "Do not send this while AI is disabled.")
        assert rejected.status_code == 409
        assert "AI help is unavailable" in rejected.json()["detail"]
        after = client.get(
            f"/v1/conversations/{chat['id']}", headers={"X-JournalPulse-User": USER_A},
        ).json()
        assert after == before
    assert len(model.calls) == 1


def test_accept_rejects_unknown_actions_and_a_missing_card(tmp_path: Path):
    model = ScriptedClient([False, True])
    app = create_app(settings=chat_settings(tmp_path), conversation_client=model)
    self_report = {
        "valence": 0,
        "arousal": 0.4,
        "agency": 0.4,
        "emotion_tags": [],
        "confidence": 0.2,
    }
    with TestClient(app) as client:
        conversation = start(client)
        say(client, conversation["id"], "Not ready for an action yet.")
        missing = client.post(
            f"/v1/conversations/{conversation['id']}/accept",
            headers={"X-JournalPulse-User": USER_A},
            json={"action_id": "mindful_breathing_ucla", "self_report": self_report},
        )
        assert missing.status_code == 409
        say(client, conversation["id"], "Maybe one small thing now.")
        choose_goal(client, conversation["id"])
        rejected = client.post(
            f"/v1/conversations/{conversation['id']}/accept",
            headers={"X-JournalPulse-User": USER_A},
            json={"action_id": "not-in-the-catalog", "self_report": self_report},
        )
        assert rejected.status_code == 422
        history = client.get("/v1/reflections", headers={"X-JournalPulse-User": USER_A})
        assert history.json()["items"] == []


def test_retention_keeps_text_only_when_requested_and_sweep_closes_stale_chats(tmp_path: Path):
    model = ScriptedClient([False])
    kept = create_app(settings=chat_settings(tmp_path), conversation_client=model)
    with TestClient(kept) as client:
        conversation = start(client, retain_text=True)
        say(client, conversation["id"], "Please keep this wording.")
        closed = client.post(
            f"/v1/conversations/{conversation['id']}/close",
            headers={"X-JournalPulse-User": USER_A},
        )
        assert closed.status_code == 200
        detail = client.get(
            f"/v1/conversations/{conversation['id']}",
            headers={"X-JournalPulse-User": USER_A},
        )
        assert any(message["content"] == "Please keep this wording." for message in detail.json()["messages"])

    current = {"moment": datetime(2026, 9, 1, tzinfo=UTC)}
    swept = create_app(
        settings=chat_settings(tmp_path / "sweep"),
        conversation_client=ScriptedClient([False]),
        clock=lambda: current["moment"],
    )
    with TestClient(swept) as client:
        conversation = start(client, retain_text=False)
        say(client, conversation["id"], "This should expire.")
        current["moment"] = current["moment"] + timedelta(hours=25)
        detail = client.get(
            f"/v1/conversations/{conversation['id']}",
            headers={"X-JournalPulse-User": USER_A},
        )
        assert detail.json()["conversation"]["status"] == "closed"
        assert all(message["content"] is None for message in detail.json()["messages"])


def test_conversations_are_private_deletable_and_exported(tmp_path: Path):
    model = ScriptedClient([True])
    app = create_app(settings=chat_settings(tmp_path), conversation_client=model)
    self_report = {
        "valence": -0.1,
        "arousal": 0.3,
        "agency": 0.6,
        "emotion_tags": [],
        "confidence": 0.3,
    }
    with TestClient(app) as client:
        conversation = start(client)
        say(client, conversation["id"], "A private note for user A.")
        said = choose_goal(client, conversation["id"])
        action_id = said["conversation"]["card"]["decision_preview"]["action_id"]
        saved = client.post(
            f"/v1/conversations/{conversation['id']}/accept",
            headers={"X-JournalPulse-User": USER_A},
            json={"action_id": action_id, "self_report": self_report},
        )
        assert saved.status_code == 201, saved.text
        decision_id = saved.json()["decision"]["decision_id"]
        outcome = client.post(
            "/v1/outcomes",
            headers={"X-JournalPulse-User": USER_A},
            json={"decision_id": decision_id, "completed": True},
        )
        assert outcome.status_code == 201
        other = {"X-JournalPulse-User": USER_B}
        for method, path in (
            ("GET", f"/v1/conversations/{conversation['id']}"),
            ("DELETE", f"/v1/conversations/{conversation['id']}"),
        ):
            response = client.request(method, path, headers=other)
            assert response.status_code == 404
        posted = say(client, conversation["id"], "Trying another person's chat.", user=USER_B)
        assert posted.status_code == 404
        accepted = client.post(
            f"/v1/conversations/{conversation['id']}/accept",
            headers=other,
            json={"action_id": action_id, "self_report": self_report},
        )
        assert accepted.status_code == 404
        closed = client.post(f"/v1/conversations/{conversation['id']}/close", headers=other)
        assert closed.status_code == 404

        exported = client.get("/v1/export", headers={"X-JournalPulse-User": USER_A})
        assert len(exported.json()["conversations"]) == 1
        assert len(exported.json()["conversation_messages"]) == 4
        removed = client.delete(
            f"/v1/conversations/{conversation['id']}",
            headers={"X-JournalPulse-User": USER_A},
        )
        assert removed.status_code == 204
        assert client.get("/v1/reflections", headers={"X-JournalPulse-User": USER_A}).json()["items"] == []
        assert client.get("/v1/outcomes", headers={"X-JournalPulse-User": USER_A}).json()["items"] == []

        another = start(client)
        say(client, another["id"], "Count this conversation.")
        deleted = client.delete("/v1/account/data", headers={"X-JournalPulse-User": USER_A})
        assert deleted.status_code == 200
        assert deleted.json()["deleted_records"] == 3


def test_model_history_contains_only_the_current_conversation(tmp_path: Path):
    model = ScriptedClient([False])
    app = create_app(settings=chat_settings(tmp_path), conversation_client=model)
    with TestClient(app) as client:
        first = start(client)
        second = start(client)
        say(client, first["id"], "Only the first conversation says apple.")
        say(client, second["id"], "Only the second conversation says orange.")
        flattened = [" ".join(message["content"] for message in call) for call in model.calls]
        assert any("apple" in item for item in flattened)
        assert any("orange" in item for item in flattened)
        assert all("apple" not in item or "orange" not in item for item in flattened)
