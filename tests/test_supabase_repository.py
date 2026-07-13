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
        requests.append((request.method, request.url.path, json.loads(request.content or b"{}")))
        return httpx.Response(201, json=[{}])

    repository = SupabaseRepository(
        settings(tmp_path),
        "user-session",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    repository.save_reflection(record())
    assert [path for _, path, _ in requests] == [
        "/rest/v1/reflections",
        "/rest/v1/affective_observations",
        "/rest/v1/policy_decisions",
        "/rest/v1/model_runs",
        "/rest/v1/safety_events",
    ]
    assert all(payload["user_id"] == str(USER_ID) for _, _, payload in requests)


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
