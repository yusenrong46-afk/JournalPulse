from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from journalpulse.api import create_app
from journalpulse.config import Settings

ROOT = Path(__file__).resolve().parents[1]
READY_DATABASE = {
    "database": "reachable",
    "schema": "schema_ready",
    "signing": "valid",
    "retention_job": "scheduled",
}


def settings(tmp_path: Path, **overrides: object) -> Settings:
    configured = Settings(
        environment="production",
        database_path=tmp_path / "unused.db",
        resource_catalog_path=ROOT / "assets" / "resources" / "catalog.json",
        openrouter_api_key="configured",
        openrouter_model="openai/gpt-6-luna",
        openrouter_base_url="https://openrouter.ai/api/v1",
        openrouter_zdr=True,
        openrouter_timeout_seconds=2,
        supabase_url="https://project.supabase.co",
        supabase_anon_key="public-anon-key",
        raw_text_retention_default=False,
        cors_origins=("https://journalpulse.vercel.app",),
        write_signing_key="k" * 40,
    )
    if overrides:
        configured = Settings(**{**configured.__dict__, **overrides})
    return configured


def ready(app_settings: Settings, probe: dict[str, str]):
    calls: list[int] = []

    def database_probe() -> dict[str, str]:
        calls.append(1)
        return probe

    with TestClient(create_app(settings=app_settings, database_probe=database_probe)) as client:
        response = client.get("/ready")
        client.get("/ready")
    return response, calls


def test_ready_reports_each_layer_and_never_calls_the_model(tmp_path: Path):
    response, calls = ready(settings(tmp_path), READY_DATABASE)
    assert response.status_code == 200
    assert response.json() == {
        "status": "ready",
        "checks": {
            "configuration": "ready",
            "resources": "ready",
            "llm": "configured:not_probed",
            **READY_DATABASE,
        },
    }
    assert len(calls) == 1, "the database probe is cached between requests"


@pytest.mark.parametrize(
    "probe",
    [
        {"database": "unreachable"},
        {"database": "reachable", "schema": "schema_missing"},
        {**READY_DATABASE, "signing": "invalid"},
        {**READY_DATABASE, "signing": "missing"},
        {**READY_DATABASE, "schema": "schema_outdated:none"},
    ],
)
def test_ready_is_not_ready_when_the_database_layer_is_not(tmp_path: Path, probe: dict[str, str]):
    response, _ = ready(settings(tmp_path), probe)
    assert response.status_code == 503
    assert response.json()["status"] == "not_ready"
    for key, value in probe.items():
        assert response.json()["checks"][key] == value


def test_an_unscheduled_retention_job_is_reported_but_not_fatal(tmp_path: Path):
    response, _ = ready(settings(tmp_path), {**READY_DATABASE, "retention_job": "not_scheduled"})
    assert response.status_code == 200
    assert response.json()["checks"]["retention_job"] == "not_scheduled"


def test_production_without_a_signing_key_is_not_ready(tmp_path: Path):
    response, _ = ready(settings(tmp_path, write_signing_key=None), READY_DATABASE)
    assert response.status_code == 503
    assert response.json()["checks"]["configuration"] == "not_ready:signing_key_missing"


def test_disabled_ai_is_explicit_and_still_ready(tmp_path: Path):
    response, _ = ready(
        settings(tmp_path, llm_feature_enabled=False, openrouter_api_key=None), READY_DATABASE
    )
    assert response.status_code == 200
    assert response.json()["checks"]["llm"] == "disabled"


def test_local_sqlite_is_labelled_and_needs_no_database_probe(tmp_path: Path):
    local = settings(
        tmp_path, environment="local", supabase_url=None, supabase_anon_key=None, cors_origins=()
    )
    response, calls = ready(local, READY_DATABASE)
    assert response.json()["checks"]["database"] == "local_sqlite"
    assert calls == []
