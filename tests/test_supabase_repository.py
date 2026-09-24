import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

import httpx

from journalpulse.config import Settings
from journalpulse.domain import (
    AffectiveState,
    ModelRun,
    PolicyDecision,
    ReflectionCopy,
    ReflectionRecord,
    SafetyMode,
    SafetyResult,
    TargetState,
)
from journalpulse.persistence import SupabaseRepository

USER_ID = UUID("00000000-0000-4000-8000-000000000007")


def settings(tmp_path: Path) -> Settings:
    return Settings(
        environment="test",
        database_path=tmp_path / "unused.db",
        resource_catalog_path=Path("unused.json"),
        openrouter_api_key=None,
        openrouter_model="openai/gpt-5.4-mini",
        openrouter_base_url="https://openrouter.ai/api/v1",
        openrouter_zdr=True,
        openrouter_timeout_seconds=2,
        supabase_url="https://project.supabase.co",
        supabase_anon_key="public-anon-key",
        raw_text_retention_default=False,
    )


def record() -> ReflectionRecord:
    state = AffectiveState(
        valence=-0.2,
        arousal=0.6,
        agency=0.4,
        emotion_tags=["frustration"],
        confidence=1,
    )
    return ReflectionRecord(
        user_id=USER_ID,
        created_at=datetime(2026, 7, 12, tzinfo=UTC),
        text=None,
        text_retained=False,
        state=state,
        target=TargetState(goal="settle", arousal=0.3, agency=0.7),
        reflection=ReflectionCopy(
            summary="A summary.",
            interpretation="An interpretation.",
            reflection_question="What changed?",
        ),
        safety=SafetyResult(
            mode=SafetyMode.NORMAL,
            locale="CA",
            exploration_allowed=True,
        ),
        decision=PolicyDecision(
            action_id="approved-resource",
            propensity=1,
            policy_name="fixed-baseline",
            policy_version="1.0.0",
            safe_action_ids=["approved-resource"],
            context_snapshot={},
            explanation="Deterministic baseline.",
        ),
        model_run=ModelRun(model="fallback", latency_ms=0, schema_valid=True),
    )


def test_supabase_adapter_forwards_user_jwt_and_writes_normalized_audit_rows(tmp_path: Path):
    requests: list[tuple[str, str, dict]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"] == "Bearer user-session"
        body = json.loads(request.content or b"{}")
        requests.append((request.method, request.url.path, body))
        return httpx.Response(200, json=body["payload"])

    repository = SupabaseRepository(
        settings(tmp_path),
        "user-session",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    repository.save_reflection(record())
    assert [path for _, path, _ in requests] == ["/rest/v1/rpc/save_reflection_bundle"]
    payload = requests[0][2]["payload"]
    assert payload["user_id"] == str(USER_ID)
    decision_payload = payload["decision"]
    assert decision_payload["selection_source"] == "policy"
    assert decision_payload["eligible_for_ope"] is True


def test_conversation_turn_and_close_use_authenticated_functions(tmp_path: Path):
    requests: list[tuple[str, str, dict]] = []
    stored: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content or b"{}")
        requests.append((request.method, request.url.path, body))
        if request.url.path.endswith("/save_conversation_turn"):
            stored.update(body["payload"]["conversation"])
            return httpx.Response(200, json=body["payload"])
        if request.method == "GET" and request.url.path.endswith("/conversations"):
            return httpx.Response(200, json=[{"record": stored}])
        if request.url.path.endswith("/close_conversation"):
            closed = {**stored, "status": "closed"}
            return httpx.Response(200, json=closed)
        return httpx.Response(200, json=body)

    from datetime import UTC, datetime
    from uuid import uuid4

    from journalpulse.domain import Conversation, ConversationMessage, MessageRole, SafetyMode

    repository = SupabaseRepository(
        settings(tmp_path),
        "user-session",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    conversation = Conversation(
        id=uuid4(),
        user_id=USER_ID,
        created_at=datetime(2026, 9, 24, tzinfo=UTC),
        updated_at=datetime(2026, 9, 24, tzinfo=UTC),
        llm_consent=True,
        retain_text=False,
        locale="CA",
        prompt_version="2026-09-24.1",
    )
    user_message = ConversationMessage(
        conversation_id=conversation.id,
        client_message_id=uuid4(),
        role=MessageRole.USER,
        content="Hello",
        created_at=datetime(2026, 9, 24, tzinfo=UTC),
        safety_mode=SafetyMode.NORMAL,
    )
    assistant_message = ConversationMessage(
        conversation_id=conversation.id,
        role=MessageRole.ASSISTANT,
        content="I hear you.",
        created_at=datetime(2026, 9, 24, 0, 0, 1, tzinfo=UTC),
        safety_mode=SafetyMode.NORMAL,
        model_run=ModelRun(model="openai/gpt-5.6-luna", latency_ms=5, schema_valid=True),
    )
    repository.save_turn(conversation, user_message, assistant_message)
    repository.close_conversation(
        USER_ID,
        conversation.id,
        purge=True,
        now=datetime(2026, 9, 24, 0, 5, tzinfo=UTC),
    )
    paths = [path for _, path, _ in requests]
    assert paths[0] == "/rest/v1/rpc/save_conversation_turn"
    assert paths[1] == "/rest/v1/conversations"
    turn_payload = requests[0][2]["payload"]
    assert turn_payload["user_message"]["content"] == "Hello"
    assert turn_payload["assistant_message"]["content"] == "I hear you."
    assert turn_payload["conversation"]["user_id"] == str(USER_ID)
    assert paths[-1] == "/rest/v1/rpc/close_conversation"
    assert requests[-1][2] == {"conversation_id": str(conversation.id), "purge": True}


def test_bulk_delete_uses_authenticated_database_function(tmp_path: Path):
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/rest/v1/rpc/delete_my_journalpulse_data"
        return httpx.Response(200, json=7)

    repository = SupabaseRepository(
        settings(tmp_path),
        "user-session",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    assert repository.delete_user_data(USER_ID) == 7
