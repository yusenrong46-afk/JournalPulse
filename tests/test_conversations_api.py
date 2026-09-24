import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

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
):
    return client.post(
        f"/v1/conversations/{conversation_id}/messages",
        headers={"X-JournalPulse-User": user},
        json={"client_message_id": message_id or str(uuid4()), "text": text},
    )


def test_conversation_accepts_a_catalog_card_and_checks_in(tmp_path: Path):
    model = ScriptedClient()
    app = create_app(settings=chat_settings(tmp_path), conversation_client=model)
    with TestClient(app) as client:
        conversation = start(client)
        first = say(client, conversation["id"], "The meeting is still replaying in my head.")
        assert first.status_code == 200, first.text
        assert first.json()["conversation"]["card"] is None
        second = say(client, conversation["id"], "I think I could try one small thing.")
        assert second.status_code == 200, second.text
        card = second.json()["conversation"]["card"]
        assert card["resource_intent"] == "reflect"
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


def test_talk_refuses_to_start_without_consent_or_a_configured_model(tmp_path: Path, monkeypatch):
    def forbidden(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("conversation client must not be constructed")

    monkeypatch.setattr(
        "journalpulse.conversations.OpenRouterConversationClient",
        forbidden,
    )
    with TestClient(create_app(settings=chat_settings(tmp_path))) as client:
        refused = client.post(
            "/v1/conversations",
            headers={"X-JournalPulse-User": USER_A},
            json={"llm_consent": False, "retain_text": False, "locale": "CA"},
        )
        assert refused.status_code == 409
        assert "private AI analysis" in refused.json()["detail"]
    disabled = chat_settings(tmp_path, openrouter_api_key=None, llm_feature_enabled=False)
    with TestClient(create_app(settings=disabled)) as client:
        refused = client.post(
            "/v1/conversations",
            headers={"X-JournalPulse-User": USER_A},
            json={"llm_consent": True, "retain_text": False, "locale": "CA"},
        )
        assert refused.status_code == 409


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
        said = say(client, conversation["id"], "A private note for user A.")
        action_id = said.json()["conversation"]["card"]["decision_preview"]["action_id"]
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
        assert len(exported.json()["conversation_messages"]) == 2
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
