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


@pytest.fixture(autouse=True)
def backend_only_unless_explicitly_configured(monkeypatch):
    monkeypatch.delenv("JOURNALPULSE_WEB_DIST", raising=False)


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


@pytest.mark.parametrize("state", ["missing", "never_succeeded", "failed", "overdue", "unknown", "healthy"])
def test_retention_diagnostics_do_not_disable_the_app(tmp_path: Path, state: str):
    response, _ = ready(settings(tmp_path), {**READY_DATABASE, "retention_state": state})
    assert response.status_code == 200
    assert response.json()["checks"]["retention_state"] == state


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


@pytest.mark.parametrize("chat_model", ["", "   "])
def test_missing_chat_model_cannot_report_ai_ready(tmp_path: Path, chat_model: str):
    response, _ = ready(settings(tmp_path, chat_model=chat_model), READY_DATABASE)
    assert response.status_code == 503
    checks = response.json()["checks"]
    assert checks["configuration"] == "not_ready:chat_model_missing"
    assert checks["llm"] == "not_configured"


def test_disabled_ai_does_not_require_a_chat_model(tmp_path: Path):
    response, _ = ready(settings(tmp_path, llm_feature_enabled=False, chat_model=""), READY_DATABASE)
    assert response.status_code == 200
    assert response.json()["checks"]["llm"] == "disabled"


@pytest.mark.parametrize("export_kind", ["missing", "file", "empty_directory"])
def test_explicit_static_export_must_include_a_homepage(tmp_path: Path, monkeypatch, export_kind: str):
    export = tmp_path / "web-out"
    if export_kind == "file":
        export.write_text("not an export directory")
    elif export_kind == "empty_directory":
        export.mkdir()
    monkeypatch.setenv("JOURNALPULSE_WEB_DIST", str(export))
    response, _ = ready(settings(tmp_path), READY_DATABASE)
    assert response.status_code == 503
    assert response.json()["checks"]["web"] == "not_ready:export_missing"


def test_valid_configured_static_export_is_ready_and_served(tmp_path: Path, monkeypatch):
    export = tmp_path / "web-out"
    export.mkdir()
    (export / "index.html").write_text("<h1>JournalPulse</h1>")
    monkeypatch.setenv("JOURNALPULSE_WEB_DIST", str(export))
    with TestClient(create_app(settings=settings(tmp_path), database_probe=lambda: READY_DATABASE)) as client:
        response = client.get("/ready")
        assert response.status_code == 200
        assert response.json()["checks"]["web"] == "ready"
        assert client.get("/").text == "<h1>JournalPulse</h1>"


def test_backend_only_deployment_needs_no_static_export(tmp_path: Path):
    response, _ = ready(settings(tmp_path), READY_DATABASE)
    assert response.status_code == 200
    assert "web" not in response.json()["checks"]


def test_local_sqlite_is_labelled_and_needs_no_database_probe(tmp_path: Path):
    local = settings(
        tmp_path, environment="local", supabase_url=None, supabase_anon_key=None, cors_origins=()
    )
    response, calls = ready(local, READY_DATABASE)
    assert response.json()["checks"]["database"] == "local_sqlite"
    assert calls == []
