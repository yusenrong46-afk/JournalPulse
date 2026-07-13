from __future__ import annotations

from journalpulse.config import load_settings
from journalpulse.intelligence import OpenRouterReflectionClient


def main() -> None:
    settings = load_settings()
    if not settings.openrouter_enabled:
        raise SystemExit(
            "OpenRouter is not configured. Add a newly rotated key to "
            "JOURNALPULSE_LLM_API_KEY in .env."
        )

    result = OpenRouterReflectionClient(settings).analyze(
        "I finished a difficult task and feel relieved, but I am still a little restless.",
        {"activity": "working", "location": "home"},
    )
    if result.model_run.used_fallback or not result.model_run.schema_valid:
        raise SystemExit("OpenRouter did not return a valid structured response.")

    print("OpenRouter live verification passed")
    print(f"model: {result.model_run.model}")
    print(f"latency_ms: {result.model_run.latency_ms}")
    print(f"tags: {', '.join(result.state.emotion_tags)}")


if __name__ == "__main__":
    main()
