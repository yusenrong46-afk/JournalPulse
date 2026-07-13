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

    @property
    def openrouter_enabled(self) -> bool:
        return bool(self.llm_feature_enabled and self.openrouter_api_key and self.openrouter_model)

    @property
    def supabase_enabled(self) -> bool:
        return bool(self.supabase_url and self.supabase_anon_key)


def load_settings() -> Settings:
    # Environment variables remain authoritative; .env only fills missing local values.
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    return Settings(
        environment=os.getenv("JOURNALPULSE_ENV", "local").strip().lower(),
        database_path=Path(
            os.getenv("JOURNALPULSE_DB_PATH", str(PROJECT_ROOT / "artifacts" / "research_beta.db"))
        ).expanduser(),
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
    )
