from pathlib import Path

import httpx
import pytest
from fastapi import HTTPException

from journalpulse.auth import resolve_auth
from journalpulse.config import Settings

USER_ID = "00000000-0000-4000-8000-000000000009"


def configured(tmp_path: Path, *, environment: str = "test", supabase: bool = False) -> Settings:
    return Settings(
        environment=environment,
        database_path=tmp_path / "test.db",
        resource_catalog_path=Path(__file__).resolve().parents[1]
        / "assets"
        / "resources"
        / "catalog.json",
        openrouter_api_key=None,
        openrouter_model="openai/gpt-5.4-mini",
        openrouter_base_url="https://openrouter.ai/api/v1",
        openrouter_zdr=True,
        openrouter_timeout_seconds=2,
        supabase_url="https://project.supabase.co" if supabase else None,
        supabase_anon_key="public-anon-key" if supabase else None,
        raw_text_retention_default=False,
    )


def test_production_refuses_development_identity_header(tmp_path: Path):
    with pytest.raises(HTTPException) as error:
        resolve_auth(
            configured(tmp_path, environment="production"),
            authorization=None,
            development_user=USER_ID,
        )
    assert error.value.status_code == 503


def test_supabase_session_is_verified_before_user_id_is_trusted(tmp_path: Path):
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"] == "Bearer signed-session"
        return httpx.Response(200, json={"id": USER_ID})

    auth = resolve_auth(
        configured(tmp_path, supabase=True),
        authorization="Bearer signed-session",
        development_user="00000000-0000-4000-8000-000000000001",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    assert str(auth.user_id) == USER_ID
    assert auth.access_token == "signed-session"


def test_supabase_rejects_missing_or_invalid_session(tmp_path: Path):
    with pytest.raises(HTTPException) as missing:
        resolve_auth(
            configured(tmp_path, supabase=True), authorization=None, development_user=None
        )
    assert missing.value.status_code == 401

    client = httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(401)))
    with pytest.raises(HTTPException) as invalid:
        resolve_auth(
            configured(tmp_path, supabase=True),
            authorization="Bearer expired",
            development_user=None,
            client=client,
        )
    assert invalid.value.status_code == 401


def test_migration_enables_rls_and_owner_policy_for_every_user_table():
    migration = (
        Path(__file__).resolve().parents[1]
        / "supabase"
        / "migrations"
        / "202607120001_research_beta.sql"
    ).read_text(encoding="utf-8")
    user_tables = {
        "profiles",
        "consents",
        "reflections",
        "affective_observations",
        "policy_decisions",
        "outcomes",
        "episodic_memories",
        "model_runs",
        "safety_events",
    }
    for table in user_tables:
        assert f"alter table public.{table} enable row level security" in migration
        assert f"'{table}'" in migration
    assert "auth.uid() = user_id" in migration
    assert "security invoker" in migration
    assert "delete_my_journalpulse_data" in migration
    assert "references public.policy_decisions(id) on delete cascade" in migration
