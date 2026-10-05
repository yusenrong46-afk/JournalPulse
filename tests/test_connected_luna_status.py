"""Privacy settings must agree with readiness and the frontend's availability hint."""

from pathlib import Path

from fastapi.testclient import TestClient

from journalpulse.api import create_app
from journalpulse.config import Settings


def test_disabled_retention_policy_does_not_advertise_private_ai(tmp_path: Path):
    settings = Settings(
        environment="test", database_path=tmp_path / "status.db",
        resource_catalog_path=Path(__file__).resolve().parents[1] / "assets/resources/catalog.json",
        openrouter_api_key="test-only-key", openrouter_model="test-model",
        openrouter_base_url="https://openrouter.ai/api/v1", openrouter_zdr=False,
        openrouter_timeout_seconds=1, supabase_url=None, supabase_anon_key=None,
        raw_text_retention_default=False,
    )
    with TestClient(create_app(settings=settings)) as client:
        ready = client.get("/ready")
        assert ready.status_code == 503
        assert ready.json()["checks"]["llm"] == "not_configured"
        status = client.get("/v1/system/status", headers={
            "X-JournalPulse-User": "00000000-0000-4000-8000-000000000001",
        })
        assert status.status_code == 200
        assert status.json()["analysis_mode"] == "local_fallback"
