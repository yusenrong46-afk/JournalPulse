from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _origins(name: str) -> tuple[str, ...]:
    value = os.getenv(name, "http://localhost:3000,http://127.0.0.1:3000")
    return tuple(origin.strip().rstrip("/") for origin in value.split(",") if origin.strip())


@dataclass(frozen=True)
class Settings:
    environment: str
    database_path: Path
    resource_catalog_path: Path
    openrouter_api_key: str | None
    openrouter_model: str
    openrouter_base_url: str
    openrouter_zdr: bool
    openrouter_timeout_seconds: float
    supabase_url: str | None
    supabase_anon_key: str | None
    raw_text_retention_default: bool
    llm_feature_enabled: bool = True
    memory_feature_enabled: bool = False
    adaptive_policy_enabled: bool = False
    cors_origins: tuple[str, ...] = ("http://localhost:3000", "http://127.0.0.1:3000")
    max_request_bytes: int = 64_000
    analysis_rate_limit_per_minute: int = 20
    openrouter_max_attempts: int = 2

    @property
    def openrouter_enabled(self) -> bool:
        return bool(self.llm_feature_enabled and self.openrouter_api_key and self.openrouter_model)

    @property
    def supabase_enabled(self) -> bool:
        return bool(self.supabase_url and self.supabase_anon_key)

    @property
    def configuration_issues(self) -> list[str]:
        issues: list[str] = []
        if self.max_request_bytes < 5_000:
            issues.append("request_limit_too_small")
        if self.analysis_rate_limit_per_minute < 1:
            issues.append("analysis_rate_limit_invalid")
        if not 1 <= self.openrouter_max_attempts <= 3:
            issues.append("openrouter_attempts_invalid")
        if self.environment == "production":
            if not self.supabase_enabled:
                issues.append("supabase_missing")
            if self.llm_feature_enabled and not self.openrouter_enabled:
                issues.append("llm_missing")
            local_origins = any(
                "localhost" in item or "127.0.0.1" in item for item in self.cors_origins
            )
            if not self.cors_origins or local_origins:
                issues.append("production_cors_invalid")
        return issues


def load_settings() -> Settings:
    # Environment variables remain authoritative; .env only fills missing local values.
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    database_value = os.getenv("JOURNALPULSE_DB_PATH", "").strip()
    database_path = (
        Path(database_value)
        if database_value
        else PROJECT_ROOT / "artifacts" / "research_beta.db"
    )
    return Settings(
        environment=os.getenv("JOURNALPULSE_ENV", "local").strip().lower(),
        database_path=database_path.expanduser(),
        resource_catalog_path=PROJECT_ROOT / "assets" / "resources" / "catalog.json",
        openrouter_api_key=(
            os.getenv("JOURNALPULSE_LLM_API_KEY") or os.getenv("OPENROUTER_API_KEY") or None
        ),
        openrouter_model=os.getenv("JOURNALPULSE_LLM_MODEL", "openai/gpt-5.4-mini").strip(),
        openrouter_base_url=os.getenv("JOURNALPULSE_LLM_BASE_URL", "https://openrouter.ai/api/v1").rstrip(
            "/"
        ),
        openrouter_zdr=_flag("JOURNALPULSE_LLM_ZDR", True),
        openrouter_timeout_seconds=float(os.getenv("JOURNALPULSE_LLM_TIMEOUT_SECONDS", "20")),
        supabase_url=os.getenv("SUPABASE_URL") or None,
        supabase_anon_key=os.getenv("SUPABASE_ANON_KEY") or None,
        raw_text_retention_default=_flag("JOURNALPULSE_SAVE_RAW_TEXT_DEFAULT", False),
        llm_feature_enabled=_flag("JOURNALPULSE_LLM_ENABLED", True),
        memory_feature_enabled=_flag("JOURNALPULSE_MEMORY_ENABLED", False),
        adaptive_policy_enabled=_flag("JOURNALPULSE_ADAPTIVE_POLICY_ENABLED", False),
        cors_origins=_origins("JOURNALPULSE_CORS_ORIGINS"),
        max_request_bytes=int(os.getenv("JOURNALPULSE_MAX_REQUEST_BYTES", "64000")),
        analysis_rate_limit_per_minute=int(
            os.getenv("JOURNALPULSE_ANALYSIS_RATE_LIMIT_PER_MINUTE", "20")
        ),
        openrouter_max_attempts=int(os.getenv("JOURNALPULSE_LLM_MAX_ATTEMPTS", "2")),
    )
