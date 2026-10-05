from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path

import pytest

from journalpulse import config


def test_load_settings_reads_project_dotenv_without_overriding_environment(
    monkeypatch, tmp_path: Path
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "JOURNALPULSE_LLM_API_KEY=dotenv-key\nJOURNALPULSE_LLM_MODEL=dotenv-model\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    monkeypatch.setenv("JOURNALPULSE_LLM_MODEL", "environment-model")
    monkeypatch.delenv("JOURNALPULSE_LLM_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    try:
        settings = config.load_settings()

        assert settings.openrouter_api_key == "dotenv-key"
        assert settings.openrouter_model == "environment-model"
    finally:
        os.environ.pop("JOURNALPULSE_LLM_API_KEY", None)


def test_load_settings_accepts_standard_openrouter_key(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    monkeypatch.delenv("JOURNALPULSE_LLM_API_KEY", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "standard-key")

    assert config.load_settings().openrouter_api_key == "standard-key"


def test_blank_database_path_uses_safe_project_default(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    monkeypatch.setenv("JOURNALPULSE_DB_PATH", "")

    assert config.load_settings().database_path == tmp_path / "artifacts" / "research_beta.db"


def test_production_configuration_rejects_local_origins_and_missing_dependencies(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    monkeypatch.setenv("JOURNALPULSE_ENV", "production")
    monkeypatch.setenv("JOURNALPULSE_CORS_ORIGINS", "http://localhost:3000")
    monkeypatch.delenv("JOURNALPULSE_LLM_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_ANON_KEY", raising=False)
    monkeypatch.delenv("RENDER_EXTERNAL_URL", raising=False)
    monkeypatch.delenv("VERCEL_PROJECT_PRODUCTION_URL", raising=False)
    monkeypatch.delenv("VERCEL_BRANCH_URL", raising=False)
    monkeypatch.delenv("VERCEL_URL", raising=False)

    settings = config.load_settings()

    assert set(settings.configuration_issues) == {
        "supabase_missing",
        "llm_missing",
        "production_cors_invalid",
    }
    valid = replace(
        settings,
        openrouter_api_key="rotated-secret",
        supabase_url="https://project.supabase.co",
        supabase_anon_key="public-anon-key",
        cors_origins=("https://journalpulse.example",),
    )
    assert valid.configuration_issues == ["signing_key_missing"]
    signed = replace(valid, write_signing_key="k" * 32)
    assert signed.configuration_issues == []


def _ready_production(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    monkeypatch.setenv("JOURNALPULSE_ENV", "production")
    monkeypatch.delenv("JOURNALPULSE_CORS_ORIGINS", raising=False)
    monkeypatch.delenv("RENDER_EXTERNAL_URL", raising=False)
    monkeypatch.delenv("VERCEL_PROJECT_PRODUCTION_URL", raising=False)
    monkeypatch.delenv("VERCEL_BRANCH_URL", raising=False)
    monkeypatch.delenv("VERCEL_URL", raising=False)
    monkeypatch.setenv("JOURNALPULSE_LLM_API_KEY", "present")
    monkeypatch.setenv("SUPABASE_URL", "https://project.supabase.co")
    monkeypatch.setenv("SUPABASE_ANON_KEY", "public-anon-key")
    monkeypatch.setenv("JOURNALPULSE_WRITE_SIGNING_KEY", "k" * 32)


def test_production_uses_render_origin_when_cors_is_unset(monkeypatch, tmp_path: Path) -> None:
    _ready_production(monkeypatch, tmp_path)
    monkeypatch.setenv("RENDER_EXTERNAL_URL", "https://journalpulse-api.onrender.com/")

    settings = config.load_settings()

    assert settings.cors_origins == ("https://journalpulse-api.onrender.com",)
    assert settings.configuration_issues == []


def test_production_uses_vercel_origins_when_cors_is_unset(monkeypatch, tmp_path: Path) -> None:
    _ready_production(monkeypatch, tmp_path)
    monkeypatch.setenv("VERCEL_PROJECT_PRODUCTION_URL", "journalpulse.vercel.app")
    monkeypatch.setenv("VERCEL_URL", "https://journalpulse-abc.vercel.app/")

    settings = config.load_settings()

    assert settings.cors_origins == (
        "https://journalpulse.vercel.app",
        "https://journalpulse-abc.vercel.app",
    )
    assert settings.configuration_issues == []


@pytest.mark.parametrize("origin", [
    "*", "https://*.example.com", "http://journalpulse.example", "https://localhost",
    "https://127.0.0.2", "https://[::1]", "https://0.0.0.0", "https://app.localhost",
    "https://journalpulse.example/path", "https://journalpulse.example?", "https://journalpulse.example#",
    "https://user:password@journalpulse.example", "https://journalpulse.example:bad",
    "https://journalpulse.example:70000", "https://journalpulse.example:",
    "https://journal pulse.example", "https://journalpulse.example\\other",
])
def test_production_rejects_unsafe_or_malformed_cors_origins(monkeypatch, tmp_path: Path, origin: str):
    _ready_production(monkeypatch, tmp_path)
    monkeypatch.setenv("JOURNALPULSE_CORS_ORIGINS", origin)
    assert "production_cors_invalid" in config.load_settings().configuration_issues


def test_local_development_still_accepts_http_origins(monkeypatch, tmp_path: Path):
    _ready_production(monkeypatch, tmp_path)
    monkeypatch.setenv("JOURNALPULSE_ENV", "local")
    monkeypatch.setenv("JOURNALPULSE_CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000")
    settings = config.load_settings()
    assert settings.configuration_issues == []
    assert settings.cors_origins == ("http://localhost:3000", "http://127.0.0.1:3000")


def test_production_accepts_explicit_https_custom_port_origins(monkeypatch, tmp_path: Path):
    _ready_production(monkeypatch, tmp_path)
    monkeypatch.setenv("JOURNALPULSE_CORS_ORIGINS", "https://journalpulse.example:8443")
    assert config.load_settings().configuration_issues == []


@pytest.mark.parametrize("value", [0.0, -1.0, float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize(
    ("field", "issue"),
    [("openrouter_timeout_seconds", "llm_timeout_invalid"), ("chat_timeout_seconds", "chat_timeout_invalid")],
)
def test_invalid_provider_deadlines_are_not_ready(
    monkeypatch, tmp_path: Path, field: str, issue: str, value: float,
):
    _ready_production(monkeypatch, tmp_path)
    monkeypatch.setenv("VERCEL_URL", "journalpulse-preview.vercel.app")
    settings = replace(config.load_settings(), **{field: value})
    assert issue in settings.configuration_issues


@pytest.mark.parametrize("value", [0.1, 45.0, 95.0])
def test_positive_finite_provider_deadlines_remain_valid(monkeypatch, tmp_path: Path, value: float):
    _ready_production(monkeypatch, tmp_path)
    monkeypatch.setenv("VERCEL_URL", "journalpulse-preview.vercel.app")
    settings = replace(config.load_settings(), openrouter_timeout_seconds=value, chat_timeout_seconds=value)
    assert settings.configuration_issues == []


@pytest.mark.parametrize("value", [95.1, 120.0])
def test_chat_timeout_that_cannot_fit_the_function_limit_is_not_ready(
    monkeypatch, tmp_path: Path, value: float,
):
    # One attempt plus its final in-flight read must fit the shared per-turn budget,
    # which itself leaves room under Vercel's 120s maxDuration for auth and the commit.
    _ready_production(monkeypatch, tmp_path)
    monkeypatch.setenv("VERCEL_URL", "journalpulse-preview.vercel.app")
    settings = replace(config.load_settings(), chat_timeout_seconds=value)
    assert "chat_timeout_exceeds_budget" in settings.configuration_issues
