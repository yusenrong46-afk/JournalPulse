import json
from pathlib import Path

import httpx

from journalpulse.config import Settings
from journalpulse.intelligence import OpenRouterReflectionClient


def settings(tmp_path: Path) -> Settings:
    root = Path(__file__).resolve().parents[1]
    return Settings(
        environment="test",
        database_path=tmp_path / "unused.db",
        resource_catalog_path=root / "assets" / "resources" / "catalog.json",
        openrouter_api_key="test-only-key",
        openrouter_model="openai/gpt-5.4-mini",
        openrouter_base_url="https://openrouter.ai/api/v1",
        openrouter_zdr=True,
        openrouter_timeout_seconds=2,
        supabase_url=None,
        supabase_anon_key=None,
        raw_text_retention_default=False,
    )


def test_openrouter_request_enforces_zdr_and_json_schema(tmp_path: Path):
    observed = {}

    def handler(request: httpx.Request) -> httpx.Response:
        observed.update(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "valence": -0.4,
                                    "arousal": 0.7,
                                    "agency": 0.35,
                                    "emotion_tags": ["frustration", "work_stress"],
                                    "confidence": 0.83,
                                    "uncertainty": "The desired outcome is not explicit.",
                                    "summary": "The meeting still feels unresolved.",
                                    "interpretation": (
                                        "The language points to frustration and reduced agency."
                                    ),
                                    "reflection_question": (
                                        "What outcome would make the meeting feel complete?"
                                    ),
                                    "resource_intent": "reflect",
                                }
                            )
                        }
                    }
                ],
                "usage": {"prompt_tokens": 90, "completion_tokens": 80},
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    result = OpenRouterReflectionClient(settings(tmp_path), client=client).analyze(
        "The meeting is still bothering me.", {"activity": "work"}
    )
    assert observed["provider"] == {"zdr": True}
    assert observed["response_format"]["type"] == "json_schema"
    assert observed["response_format"]["json_schema"]["strict"] is True
    assert result.state.valence == -0.4
    assert result.model_run.schema_valid is True
    assert result.model_run.prompt_tokens == 90


def test_openrouter_client_refuses_non_zdr_configuration(tmp_path: Path):
    configured = settings(tmp_path)
    configured = Settings(**{**configured.__dict__, "openrouter_zdr": False})
    try:
        OpenRouterReflectionClient(configured)
    except ValueError as exc:
        assert "zero-data-retention" in str(exc)
    else:
        raise AssertionError("non-ZDR configuration must be rejected")
