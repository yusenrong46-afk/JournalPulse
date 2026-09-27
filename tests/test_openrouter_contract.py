import json
from pathlib import Path

import httpx
import pytest

from journalpulse.config import Settings
from journalpulse.intelligence import (
    OpenRouterReflectionClient,
    UnsupportedProviderResponse,
    safe_analyze,
)


def settings(tmp_path: Path) -> Settings:
    root = Path(__file__).resolve().parents[1]
    return Settings(
        environment="test",
        database_path=tmp_path / "unused.db",
        resource_catalog_path=root / "assets" / "resources" / "catalog.json",
        openrouter_api_key="test-only-key",
        openrouter_model="openai/gpt-6-luna",
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
    user_content = observed["messages"][1]["content"]
    assert isinstance(user_content, str)
    assert not isinstance(user_content, dict)
    assert "The meeting is still bothering me." in user_content
    assert "temperature" not in observed
    assert observed["model"] == "openai/gpt-6-luna"
    assert observed["reasoning"] == {"effort": "medium"}
    assert observed["include_reasoning"] is False
    assert observed["max_tokens"] == 4000
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


def test_openrouter_retries_transient_failure_then_validates_schema(tmp_path: Path):
    calls = 0
    delays: list[float] = []

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(503, json={"error": "temporarily unavailable"})
        return httpx.Response(
            200,
            json={
                "model": "resolved-model",
                "provider": "zdr-provider",
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "valence": 0.2,
                                    "arousal": 0.4,
                                    "agency": 0.7,
                                    "emotion_tags": ["relief"],
                                    "confidence": 0.8,
                                    "uncertainty": None,
                                    "summary": "The difficult part appears to be over.",
                                    "interpretation": "Relief and remaining activation coexist.",
                                    "reflection_question": "What would help the activation settle?",
                                    "resource_intent": "pause",
                                }
                            )
                        }
                    }
                ],
            },
        )

    result = OpenRouterReflectionClient(
        settings(tmp_path),
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        sleeper=delays.append,
    ).analyze("I finished the difficult task.", {})

    assert calls == 2
    assert delays == [0.15]
    assert result.model_run.model == "resolved-model"
    assert result.model_run.provider == "zdr-provider"


def _valid_reflection_json() -> str:
    return json.dumps(
        {
            "valence": -0.4,
            "arousal": 0.7,
            "agency": 0.35,
            "emotion_tags": ["frustration"],
            "confidence": 0.83,
            "uncertainty": None,
            "summary": "The meeting still feels unresolved.",
            "interpretation": "The language points to frustration and reduced agency.",
            "reflection_question": "What outcome would make the meeting feel complete?",
            "resource_intent": "reflect",
        }
    )


def test_openrouter_parses_text_part_array_content(tmp_path: Path):
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": [{"type": "text", "text": _valid_reflection_json()}]
                        }
                    }
                ]
            },
        )

    result = OpenRouterReflectionClient(
        settings(tmp_path),
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    ).analyze("The meeting is still bothering me.", {})
    assert result.state.confidence == 0.83
    assert result.model_run.schema_valid is True
    assert result.model_run.used_fallback is False


def test_unsupported_provider_content_is_not_treated_as_successful_output(tmp_path: Path):
    shapes = (
        {"unexpected": True},
        [{"type": "image_url", "image_url": {"url": "https://example.invalid/x"}}],
        None,
    )
    for shape in shapes:
        def handler(_: httpx.Request, payload: object = shape) -> httpx.Response:
            return httpx.Response(200, json={"choices": [{"message": {"content": payload}}]})

        client = OpenRouterReflectionClient(
            settings(tmp_path),
            client=httpx.Client(transport=httpx.MockTransport(handler)),
        )
        with pytest.raises(UnsupportedProviderResponse):
            client.analyze("The meeting is still bothering me.", {})
        with pytest.raises(UnsupportedProviderResponse):
            safe_analyze(
                settings(tmp_path),
                text="The meeting is still bothering me.",
                context={},
                consent=True,
                self_report=None,
                client=client,
            )
