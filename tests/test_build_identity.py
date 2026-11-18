"""Build identity: /health and /v1/system/status report the deployed Git commit."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from journalpulse.api import create_app
from journalpulse.build_info import COMMIT_VARIABLES, build_identity
from journalpulse.config import Settings

CATALOG = Path(__file__).resolve().parents[1] / "assets/resources/catalog.json"
COMMIT = "0123456789abcdef0123456789abcdef01234567"


@pytest.fixture(autouse=True)
def no_host_commit(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in COMMIT_VARIABLES:
        monkeypatch.delenv(name, raising=False)


def app(tmp_path: Path):
    return create_app(settings=Settings(
        environment="test", database_path=tmp_path / "build.db", resource_catalog_path=CATALOG,
        openrouter_api_key=None, openrouter_model="test-double", openrouter_base_url="https://provider.example",
        openrouter_zdr=True, openrouter_timeout_seconds=1, raw_text_retention_default=False,
        supabase_url=None, supabase_anon_key=None,
        write_signing_key="test-only-build-signing-key-0123456789abcdef",
    ))


def test_unknown_without_a_host_variable() -> None:
    assert build_identity().model_dump() == {"commit": "unknown", "source": "none"}


@pytest.mark.parametrize("name", COMMIT_VARIABLES)
def test_each_host_variable_is_read(monkeypatch: pytest.MonkeyPatch, name: str) -> None:
    monkeypatch.setenv(name, COMMIT.upper())
    assert build_identity().model_dump() == {"commit": COMMIT, "source": name}


def test_first_variable_wins_and_invalid_values_are_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("JOURNALPULSE_GIT_COMMIT", "not a commit; rm -rf /")
    monkeypatch.setenv("VERCEL_GIT_COMMIT_SHA", COMMIT[:7])
    monkeypatch.setenv("GITHUB_SHA", COMMIT)
    assert build_identity().model_dump() == {"commit": COMMIT[:7], "source": "VERCEL_GIT_COMMIT_SHA"}


def test_health_and_system_status_report_the_commit(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("VERCEL_GIT_COMMIT_SHA", COMMIT)
    with TestClient(app(tmp_path)) as client:
        health = client.get("/health")
        assert health.json() == {"status": "ok"}
        assert health.headers["X-JournalPulse-Commit"] == COMMIT
        status = client.get("/v1/system/status")
        assert status.status_code == 200, status.text
        assert status.json()["build"] == {"commit": COMMIT, "source": "VERCEL_GIT_COMMIT_SHA"}


def test_health_says_unknown_rather_than_guessing(tmp_path: Path) -> None:
    with TestClient(app(tmp_path)) as client:
        assert client.get("/health").headers["X-JournalPulse-Commit"] == "unknown"
