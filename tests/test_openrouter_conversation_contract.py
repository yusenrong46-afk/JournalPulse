import json
from pathlib import Path

import httpx
import pytest

from journalpulse.config import Settings
from journalpulse.domain import FEELINGS
from journalpulse.intelligence import (
    CONVERSATION_JSON_SCHEMA,
    CONVERSATION_PROMPT_VERSION,
    ConversationProviderError,
    OpenRouterConversationClient,
    UnsupportedProviderResponse,
)

ALLOWED_BODY_KEYS = {
    "model",
    "provider",
    "max_tokens",
    "include_reasoning",
    "reasoning",
    "response_format",
    "messages",
}


def settings(tmp_path: Path, **overrides: object) -> Settings:
    root = Path(__file__).resolve().parents[1]
    configured = Settings(
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
    if overrides:
        configured = Settings(**{**configured.__dict__, **overrides})
    return configured


def _payload(**overrides: object) -> dict:
    body = {
        "reply": "That sounds heavy. What part is still loudest?",
        "offer_action": False,
        "resource_intent": "reflect",
        "card_reason": "",
        "summary": "A hard moment is still present.",
        "feelings": [],
    }
    body.update(overrides)
    return body


def _response(content: object, finish_reason: str = "stop") -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "model": "openai/gpt-6-luna",
            "provider": "openrouter",
            "choices": [{"finish_reason": finish_reason, "message": {"content": content}}],
            "usage": {"prompt_tokens": 40, "completion_tokens": 30},
        },
    )


def test_luna_request_uses_only_documented_parameters(tmp_path: Path):
    observed: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        observed.update(json.loads(request.content))
        return _response(json.dumps(_payload()))

    OpenRouterConversationClient(
        settings(tmp_path),
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    ).complete(
        [
            {"role": "user", "content": "Only this conversation mentions the late train."},
            {"role": "assistant", "content": "The delay is still with you."},
            {"role": "user", "content": "Yes."},
        ]
    )
    assert observed["model"] == "openai/gpt-6-luna"
    assert observed["provider"] == {"zdr": True}
    assert "temperature" not in observed
    assert set(observed) == ALLOWED_BODY_KEYS
    assert observed["reasoning"] == {"effort": "medium"}
    assert observed["include_reasoning"] is False
    assert observed["response_format"]["json_schema"]["strict"] is True
    assert observed["max_tokens"] == 4000
    contents = [message["content"] for message in observed["messages"]]
    assert all(isinstance(content, str) for content in contents)
    assert contents[0]
    assert "late train" in contents[1]
    assert "another conversation" not in " ".join(contents)


def test_http_200_upstream_rate_limit_is_retried(tmp_path: Path):
    calls = 0
    delays: list[float] = []

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(
                200,
                json={"error": {"message": "rate-limited upstream", "code": 429}},
            )
        return _response(json.dumps(_payload()))

    result = OpenRouterConversationClient(
        settings(tmp_path),
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        sleeper=delays.append,
    ).complete([{"role": "user", "content": "Hello."}])
    assert calls == 2
    assert delays == [1.5]
    assert result.reply.startswith("That sounds heavy")


def test_upstream_rate_limit_raises_instead_of_crashing(tmp_path: Path):
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"error": {"message": "rate-limited upstream", "code": 429}},
        )

    with pytest.raises(ConversationProviderError, match="rate-limited upstream") as caught:
        OpenRouterConversationClient(
            settings(tmp_path),
            client=httpx.Client(transport=httpx.MockTransport(handler)),
            sleeper=lambda _delay: None,
        ).complete([{"role": "user", "content": "Hello."}])
    assert caught.value.status_code == 429


def test_conversation_client_refuses_missing_key_or_zdr(tmp_path: Path):
    with pytest.raises(ValueError, match="not configured"):
        OpenRouterConversationClient(settings(tmp_path, openrouter_api_key=None))
    with pytest.raises(ValueError, match="zero-data-retention"):
        OpenRouterConversationClient(settings(tmp_path, openrouter_zdr=False))


def test_truncated_or_invalid_schema_is_a_provider_error(tmp_path: Path):
    def length(_: httpx.Request) -> httpx.Response:
        return _response(json.dumps(_payload()), finish_reason="length")

    with pytest.raises(ConversationProviderError, match="cut off"):
        OpenRouterConversationClient(
            settings(tmp_path),
            client=httpx.Client(transport=httpx.MockTransport(length)),
        ).complete([{"role": "user", "content": "Hello."}])

    def invalid(_: httpx.Request) -> httpx.Response:
        return _response(json.dumps(_payload(offer_action=True, card_reason="")))

    with pytest.raises(ConversationProviderError, match="schema"):
        OpenRouterConversationClient(
            settings(tmp_path),
            client=httpx.Client(transport=httpx.MockTransport(invalid)),
        ).complete([{"role": "user", "content": "Hello."}])


def test_unsupported_conversation_content_is_not_parsed_as_text(tmp_path: Path):
    def handler(_: httpx.Request) -> httpx.Response:
        return _response({"unexpected": True})

    with pytest.raises(UnsupportedProviderResponse):
        OpenRouterConversationClient(
            settings(tmp_path),
            client=httpx.Client(transport=httpx.MockTransport(handler)),
        ).complete([{"role": "user", "content": "Hello."}])


def test_text_part_arrays_are_accepted(tmp_path: Path):
    def handler(_: httpx.Request) -> httpx.Response:
        return _response([{"type": "text", "text": json.dumps(_payload(offer_action=False))}])

    result = OpenRouterConversationClient(
        settings(tmp_path),
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    ).complete([{"role": "user", "content": "Hello."}])
    assert result.reply.startswith("That sounds heavy")
    assert result.model_run.prompt_version == CONVERSATION_PROMPT_VERSION
    assert result.model_run.schema_valid is True


def test_suggested_feelings_come_only_from_the_allowed_list(tmp_path: Path):
    def known(_: httpx.Request) -> httpx.Response:
        return _response(json.dumps(_payload(feelings=["tired", "anxious", "tired"])))

    result = OpenRouterConversationClient(
        settings(tmp_path),
        client=httpx.Client(transport=httpx.MockTransport(known)),
    ).complete([{"role": "user", "content": "Hello."}])
    assert result.feelings == ("tired", "anxious")
    schema = CONVERSATION_JSON_SCHEMA["schema"]
    assert "feelings" in schema["required"]
    assert set(schema["properties"]["feelings"]["items"]["enum"]) == set(FEELINGS)

    def invented(_: httpx.Request) -> httpx.Response:
        return _response(json.dumps(_payload(feelings=["depressed"])))

    with pytest.raises(ConversationProviderError, match="schema"):
        OpenRouterConversationClient(
            settings(tmp_path),
            client=httpx.Client(transport=httpx.MockTransport(invented)),
        ).complete([{"role": "user", "content": "Hello."}])
