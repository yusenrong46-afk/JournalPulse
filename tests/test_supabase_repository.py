import json
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import parse_qs
from uuid import UUID, uuid4

import httpx
import pytest

from journalpulse.config import Settings
from journalpulse.domain import (
    AffectiveState,
    Conversation,
    ConversationMessage,
    MessageRole,
    ModelRun,
    PolicyDecision,
    ReflectionCopy,
    ReflectionRecord,
    SafetyMode,
    SafetyResult,
    TargetState,
)
from journalpulse.persistence import (
    ConversationAlreadyAccepted,
    ConversationClosed,
    ConversationNotFound,
    ConversationStale,
    DuplicateOutcomeError,
    StorageUnavailable,
    SupabaseRepository,
    supabase_readiness,
)
from journalpulse.signing import sign_text

USER_ID = UUID("00000000-0000-4000-8000-000000000007")
KEY = "unit-test-signing-key-0123456789abcdef"


def settings(tmp_path: Path, **overrides: object) -> Settings:
    configured = Settings(
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
        write_signing_key=KEY,
    )
    if overrides:
        configured = Settings(**{**configured.__dict__, **overrides})
    return configured


def repository(tmp_path: Path, handler, **overrides: object) -> SupabaseRepository:
    return SupabaseRepository(
        settings(tmp_path, **overrides),
        "user-session",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )


def record() -> ReflectionRecord:
    return ReflectionRecord(
        user_id=USER_ID,
        created_at=datetime(2026, 7, 12, tzinfo=UTC),
        state=AffectiveState(valence=-0.2, arousal=0.6, agency=0.4, emotion_tags=["tired"]),
        target=TargetState(goal="settle"),
        reflection=ReflectionCopy(summary="A summary.", interpretation="Calm.", reflection_question="What?"),
        safety=SafetyResult(mode=SafetyMode.NORMAL, locale="CA", exploration_allowed=True),
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


def conversation() -> Conversation:
    return Conversation(
        id=uuid4(),
        user_id=USER_ID,
        created_at=datetime(2026, 9, 24, tzinfo=UTC),
        updated_at=datetime(2026, 9, 24, tzinfo=UTC),
        llm_consent=True,
        locale="CA",
        prompt_version="2026-09-27.2",
    )


def turn(chat: Conversation) -> tuple[ConversationMessage, ConversationMessage]:
    return (
        ConversationMessage(
            conversation_id=chat.id,
            client_message_id=uuid4(),
            role=MessageRole.USER,
            content="Hello",
            safety_mode=SafetyMode.NORMAL,
        ),
        ConversationMessage(
            conversation_id=chat.id,
            role=MessageRole.ASSISTANT,
            content="I hear you.",
            safety_mode=SafetyMode.NORMAL,
        ),
    )


def test_provenance_writes_are_signed_over_the_exact_payload(tmp_path: Path):
    seen: list[tuple[str, dict]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"] == "Bearer user-session"
        body = json.loads(request.content)
        seen.append((request.url.path, body))
        return httpx.Response(200, json=json.loads(body["payload"])["reflection"])

    repository(tmp_path, handler).save_reflection(record())
    path, body = seen[0]
    assert path == "/rest/v1/rpc/jp_save_reflection"
    assert body["signature"] == sign_text(body["payload"], KEY)
    envelope = json.loads(body["payload"])
    assert envelope["purpose"] == "save_reflection"
    assert envelope["user_id"] == str(USER_ID)
    assert envelope["reflection"]["decision"]["policy_name"] == "fixed-baseline"


def test_writes_refuse_to_run_without_a_signing_key(tmp_path: Path):
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("no request may be sent unsigned")

    with pytest.raises(StorageUnavailable):
        repository(tmp_path, handler, write_signing_key=None).save_reflection(record())


def test_turn_commit_carries_the_expected_revision(tmp_path: Path):
    seen: list[dict] = []
    chat = conversation()
    user, assistant = turn(chat)

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append(json.loads(body["payload"]))
        envelope = seen[-1]
        committed = {**envelope["conversation"], "revision": envelope["expected_revision"] + 1}
        return httpx.Response(
            200,
            json={
                "conversation": committed,
                "user_message": envelope["user_message"],
                "assistant_message": envelope["assistant_message"],
            },
        )

    stored, _, _ = repository(tmp_path, handler).commit_turn(chat, user, assistant, expected_revision=4)
    assert seen[0]["purpose"] == "commit_turn"
    assert seen[0]["expected_revision"] == 4
    assert stored.revision == 5


@pytest.mark.parametrize(
    ("status", "message", "error"),
    [
        (409, "Conversation changed", ConversationStale),
        (409, "Conversation is closed", ConversationClosed),
        (409, "Conversation already accepted", ConversationAlreadyAccepted),
        (404, "Conversation not found", ConversationNotFound),
        (409, "An outcome already exists for this decision", DuplicateOutcomeError),
        (503, "connection refused", StorageUnavailable),
    ],
)
def test_database_refusals_become_typed_errors(tmp_path: Path, status: int, message: str, error: type):
    chat = conversation()
    user, assistant = turn(chat)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json={"code": f"PT{status}", "message": message})

    with pytest.raises(error):
        repository(tmp_path, handler).commit_turn(chat, user, assistant, expected_revision=0)


def test_unreachable_database_is_storage_unavailable(tmp_path: Path):
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down")

    with pytest.raises(StorageUnavailable):
        repository(tmp_path, handler).get_conversation(USER_ID, uuid4())


def test_close_and_delete_use_the_lifecycle_functions(tmp_path: Path):
    chat = conversation()
    seen: list[tuple[str, dict]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append((request.url.path, body))
        if request.url.path.endswith("jp_close_conversation"):
            return httpx.Response(
                200, json={**chat.model_dump(mode="json"), "status": "closed", "revision": 3}
            )
        return httpx.Response(200, json=True)

    repo = repository(tmp_path, handler)
    closed = repo.close_conversation(USER_ID, chat.id)
    assert closed is not None and closed.status == "closed" and closed.revision == 3
    assert repo.delete_conversation(USER_ID, chat.id) is True
    assert seen == [
        ("/rest/v1/rpc/jp_close_conversation", {"p_conversation_id": str(chat.id)}),
        ("/rest/v1/rpc/jp_delete_conversation", {"p_conversation_id": str(chat.id)}),
    ]


def test_export_reads_every_page_of_every_table(tmp_path: Path):
    sizes = {
        "reflections": 1203,
        "outcomes": 501,
        "conversations": 500,
        "conversation_messages": 1777,
        "policy_decisions": 1203,
        "model_runs": 3,
        "safety_events": 0,
        "affective_observations": 1203,
    }
    requests: dict[str, int] = {name: 0 for name in sizes}
    sample = record().model_dump(mode="json")
    chat = conversation().model_dump(mode="json")
    message = turn(conversation())[0].model_dump(mode="json")

    def handler(request: httpx.Request) -> httpx.Response:
        table = request.url.path.rsplit("/", 1)[-1]
        query = parse_qs(request.url.query.decode())
        assert query["user_id"] == [f"eq.{USER_ID}"]
        assert query["order"][0].endswith(",id.asc") or query["order"][0].endswith(",id.desc")
        limit, offset = int(query["limit"][0]), int(query["offset"][0])
        requests[table] += 1
        count = max(0, min(limit, sizes[table] - offset))
        if table in {"reflections", "outcomes"}:
            if table == "reflections":
                rows = [{"record": sample} for _ in range(count)]
            else:
                rows = [
                    {"record": {"user_id": str(USER_ID), "decision_id": str(uuid4()), "completed": True}}
                    for _ in range(count)
                ]
        elif table == "conversations":
            rows = [{"record": chat, "revision": 2} for _ in range(count)]
        elif table == "conversation_messages":
            rows = [{"record": message} for _ in range(count)]
        else:
            rows = [{"id": offset + index} for index in range(count)]
        return httpx.Response(200, json=rows)

    exported = repository(tmp_path, handler).export_user_data(USER_ID)
    for table, size in sizes.items():
        assert len(exported[table]) == size, table
    assert requests["reflections"] == 3
    assert requests["conversations"] == 2
    assert requests["safety_events"] == 1
    assert exported["conversations"][0]["revision"] == 2


def test_readiness_distinguishes_schema_signing_and_reachability(tmp_path: Path):
    def ready(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["probe"].startswith("readiness:")
        assert body["signature"] == sign_text(body["probe"], KEY)
        assert request.headers["authorization"] == "Bearer public-anon-key"
        return httpx.Response(
            200, json={"schema": "phase-a-1", "signing": "valid", "retention_job": "scheduled"}
        )

    checks = supabase_readiness(settings(tmp_path), client=httpx.Client(transport=httpx.MockTransport(ready)))
    assert checks == {
        "database": "reachable",
        "schema": "schema_ready",
        "signing": "valid",
        "retention_job": "scheduled",
    }

    def missing(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"message": "function not found"})

    checks = supabase_readiness(
        settings(tmp_path), client=httpx.Client(transport=httpx.MockTransport(missing))
    )
    assert checks == {"database": "reachable", "schema": "schema_missing"}

    def down(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down")

    checks = supabase_readiness(settings(tmp_path), client=httpx.Client(transport=httpx.MockTransport(down)))
    assert checks == {"database": "unreachable"}

    def mismatched(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json={"schema": "phase-a-1", "signing": "invalid", "retention_job": "scheduled"}
        )

    unsigned = supabase_readiness(
        settings(tmp_path, write_signing_key=None),
        client=httpx.Client(transport=httpx.MockTransport(mismatched)),
    )
    assert unsigned["signing"] == "not_configured"
    assert supabase_readiness(settings(tmp_path, supabase_url=None)) == {"database": "not_configured"}


def test_bulk_delete_uses_authenticated_database_function(tmp_path: Path):
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/rest/v1/rpc/delete_my_journalpulse_data"
        return httpx.Response(200, json=7)

    assert repository(tmp_path, handler).delete_user_data(USER_ID) == 7
