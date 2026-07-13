from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path

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
    assert valid.configuration_issues == []
