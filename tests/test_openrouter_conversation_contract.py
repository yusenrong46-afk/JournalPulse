import json
import logging
from pathlib import Path

import httpx
import pytest

from journalpulse.config import (
    CHAT_PROVIDER_BUDGET_SECONDS,
    STREAM_READ_GRACE_SECONDS,
    Settings,
)
from journalpulse.domain import FEELINGS
from journalpulse.intelligence import (
    CONVERSATION_JSON_SCHEMA,
    CONVERSATION_PROMPT_VERSION,
    CompletionDiagnostic,
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


def test_luna_client_cannot_be_created_when_ai_feature_is_disabled(tmp_path: Path):
    with pytest.raises(ValueError, match="not configured"):
        OpenRouterConversationClient(settings(tmp_path, llm_feature_enabled=False))


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
    assert observed["provider"] == {"zdr": True, "require_parameters": True}
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

    with pytest.raises(ConversationProviderError, match="temporarily unavailable") as caught:
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


@pytest.mark.parametrize("payload", [
    [], {}, {"choices": []}, {"choices": [None]}, {"choices": ["reply"]},
    {"choices": [{"message": None}]},
    {"choices": [{"message": {"content": json.dumps(_payload())}}], "usage": None},
    {"choices": [{"message": {"content": json.dumps(_payload())}}], "usage": {"prompt_tokens": -1}},
])
def test_malformed_provider_envelope_is_a_controlled_error(tmp_path: Path, payload: object):
    with pytest.raises(ConversationProviderError):
        OpenRouterConversationClient(
            settings(tmp_path),
            client=httpx.Client(transport=httpx.MockTransport(
                lambda _: httpx.Response(200, json=payload),
            )),
        ).complete([{"role": "user", "content": "Hello."}])


@pytest.mark.parametrize("changed", [
    {"offer_action": "yes", "card_reason": "An unrequested action."},
    {"offer_action": 1, "card_reason": "An unrequested action."},
    {"resource_intent": "invented"}, {"unexpected": "untrusted field"},
    {"reply": " \n "}, {"summary": " "},
])
def test_schema_valid_means_exact_output_contract(tmp_path: Path, changed: dict):
    with pytest.raises(ConversationProviderError, match="schema"):
        OpenRouterConversationClient(
            settings(tmp_path),
            client=httpx.Client(transport=httpx.MockTransport(
                lambda _: _response(json.dumps(_payload(**changed))),
            )),
        ).complete([{"role": "user", "content": "Hello."}])


@pytest.mark.parametrize("missing", ["card_reason", "feelings"])
def test_provider_must_return_all_required_schema_fields(tmp_path: Path, missing: str):
    output = _payload()
    del output[missing]
    with pytest.raises(ConversationProviderError, match="schema"):
        OpenRouterConversationClient(
            settings(tmp_path),
            client=httpx.Client(transport=httpx.MockTransport(
                lambda _: _response(json.dumps(output)),
            )),
        ).complete([{"role": "user", "content": "Hello."}])


@pytest.mark.parametrize("status", [200, 401, 429, 500])
def test_provider_diagnostics_are_not_exposed_to_the_user(tmp_path: Path, status: int):
    with pytest.raises(ConversationProviderError) as caught:
        OpenRouterConversationClient(
            settings(tmp_path, openrouter_max_attempts=1),
            client=httpx.Client(transport=httpx.MockTransport(
                lambda _: httpx.Response(status, json={
                    "error": {"code": 429, "message": "PRIVATE_PROVIDER_DIAGNOSTIC"},
                }),
            )),
        ).complete([{"role": "user", "content": "Hello."}])
    assert "PRIVATE_PROVIDER_DIAGNOSTIC" not in str(caught.value)


def test_provider_response_size_is_bounded(tmp_path: Path):
    valid_json = json.dumps({"choices": [{"message": {"content": json.dumps(_payload())}}]}).encode()
    with pytest.raises(ConversationProviderError):
        OpenRouterConversationClient(
            settings(tmp_path, openrouter_max_attempts=1),
            client=httpx.Client(transport=httpx.MockTransport(
                lambda _: httpx.Response(200, content=valid_json + b" " * 256_001),
            )),
        ).complete([{"role": "user", "content": "Hello."}])


def test_provider_stream_obeys_total_deadline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    # Advancing a controlled clock reproduces slow trickling without a slow test.
    clock = iter([0.0, 3.0])
    monkeypatch.setattr("journalpulse.intelligence.time.monotonic", lambda: next(clock))
    with pytest.raises(ConversationProviderError, match="respond in time"):
        OpenRouterConversationClient(
            settings(tmp_path, openrouter_max_attempts=1, chat_timeout_seconds=2),
            client=httpx.Client(transport=httpx.MockTransport(
                lambda _: _response(json.dumps(_payload())),
            )),
        ).complete([{"role": "user", "content": "Hello."}])


def test_stream_read_failure_is_controlled(tmp_path: Path):
    def failed(_: httpx.Request) -> httpx.Response:
        raise httpx.ReadError("PRIVATE_STREAM_ERROR")

    with pytest.raises(ConversationProviderError) as caught:
        OpenRouterConversationClient(
            settings(tmp_path, openrouter_max_attempts=1),
            client=httpx.Client(transport=httpx.MockTransport(failed)),
        ).complete([{"role": "user", "content": "Hello."}])
    assert "PRIVATE_STREAM_ERROR" not in str(caught.value)


@pytest.mark.parametrize("content,expected_field,expected_type", [
    (json.dumps(_payload(feelings=["PRIVATE_MODEL_VALUE"])), "feelings", "value_error"),
    (json.dumps(_payload(**{"PRIVATE_UNKNOWN_KEY": "PRIVATE_MODEL_VALUE"})),
     "unrecognized_field", "extra_forbidden"),
    ("PRIVATE_RAW_MODEL_REPLY", "root", "json_invalid"),
])
def test_rejection_diagnostics_log_only_safe_field_and_error_types(
    tmp_path: Path, caplog: pytest.LogCaptureFixture,
    content: str, expected_field: str, expected_type: str,
):
    with caplog.at_level(logging.WARNING, logger="journalpulse.intelligence"):
        with pytest.raises(ConversationProviderError):
            OpenRouterConversationClient(
                settings(tmp_path),
                client=httpx.Client(transport=httpx.MockTransport(lambda _: _response(content))),
            ).complete([{"role": "user", "content": "PRIVATE_JOURNAL_INPUT"}])
    records = [record for record in caplog.records if record.name == "journalpulse.intelligence"]
    assert len(records) == 1
    diagnostic = json.loads(records[0].getMessage().split(" ", 1)[1])
    assert diagnostic["stage"] == "output_schema"
    assert {"field": expected_field, "type": expected_type} in diagnostic["errors"]
    assert "PRIVATE_" not in records[0].getMessage()
    assert "test-only-key" not in records[0].getMessage()
    assert records[0].exc_info is None


@pytest.mark.parametrize("payload,expected_stage", [
    ({"choices": []}, "envelope"),
    ({"choices": [{"message": {"content": json.dumps(_payload())}}], "usage": None}, "usage_metadata"),
])
def test_rejection_diagnostics_distinguish_envelope_from_metadata(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, payload: dict, expected_stage: str,
):
    with caplog.at_level(logging.WARNING, logger="journalpulse.intelligence"):
        with pytest.raises(ConversationProviderError):
            OpenRouterConversationClient(
                settings(tmp_path),
                client=httpx.Client(transport=httpx.MockTransport(
                    lambda _: httpx.Response(200, json=payload),
                )),
            ).complete([{"role": "user", "content": "PRIVATE_JOURNAL_INPUT"}])
    diagnostic = json.loads(caplog.records[-1].getMessage().split(" ", 1)[1])
    assert diagnostic["stage"] == expected_stage
    assert "PRIVATE_" not in caplog.records[-1].getMessage()


def test_cross_field_semantic_rejection_is_identified_without_output(
    tmp_path: Path, caplog: pytest.LogCaptureFixture,
):
    with caplog.at_level(logging.WARNING, logger="journalpulse.intelligence"):
        with pytest.raises(ConversationProviderError):
            OpenRouterConversationClient(
                settings(tmp_path),
                client=httpx.Client(transport=httpx.MockTransport(
                    lambda _: _response(json.dumps(_payload(offer_action=True, card_reason=""))),
                )),
            ).complete([{"role": "user", "content": "PRIVATE_JOURNAL_INPUT"}])
    diagnostic = json.loads(caplog.records[-1].getMessage().split(" ", 1)[1])
    assert diagnostic["stage"] == "output_semantics"
    assert diagnostic["errors"] == [{"field": "card_reason", "type": "nonblank_reason_required"}]


def test_validation_diagnostics_are_bounded(tmp_path: Path, caplog: pytest.LogCaptureFixture):
    output = _payload(**{f"PRIVATE_FIELD_{index}": "PRIVATE_VALUE" for index in range(30)})
    with caplog.at_level(logging.WARNING, logger="journalpulse.intelligence"):
        with pytest.raises(ConversationProviderError):
            OpenRouterConversationClient(
                settings(tmp_path),
                client=httpx.Client(transport=httpx.MockTransport(
                    lambda _: _response(json.dumps(output)),
                )),
            ).complete([{"role": "user", "content": "PRIVATE_JOURNAL_INPUT"}])
    diagnostic = json.loads(caplog.records[-1].getMessage().split(" ", 1)[1])
    assert len(diagnostic["errors"]) == 5
    assert "PRIVATE_" not in caplog.records[-1].getMessage()


def test_schema_failure_exposes_only_safe_diagnostic_headers(tmp_path: Path):
    private = "private journal wording and model output"
    body = _payload(reply=17, **{private: private})
    with pytest.raises(ConversationProviderError) as caught:
        OpenRouterConversationClient(
            settings(tmp_path),
            client=httpx.Client(transport=httpx.MockTransport(lambda _: _response(json.dumps(body)))),
        ).complete([{"role": "user", "content": private}])
    headers = caught.value.diagnostic_headers
    assert headers["X-JournalPulse-Error-Stage"] == "output_schema"
    assert "reply.string_type" in headers["X-JournalPulse-Error-Fields"]
    assert "unrecognized_field.extra_forbidden" in headers["X-JournalPulse-Error-Fields"]
    assert private not in json.dumps(headers)
    assert "17" not in json.dumps(headers)


def test_upstream_failure_is_distinguishable_from_invalid_model_output(tmp_path: Path):
    with pytest.raises(ConversationProviderError) as caught:
        OpenRouterConversationClient(
            settings(tmp_path, openrouter_max_attempts=1),
            client=httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(
                200, json={"error": {"code": 502, "message": "private provider diagnostic"}},
            ))),
        ).complete([{"role": "user", "content": "Fictional writing."}])
    assert caught.value.diagnostic_headers == {"X-JournalPulse-Error-Stage": "upstream_error"}


def test_diagnostic_headers_remain_bounded_and_strip_unknown_values():
    private = "PRIVATE_VALUE\r\nX-Evil: secret"
    headers = CompletionDiagnostic(private, tuple((private, private) for _ in range(30))).headers
    assert headers["X-JournalPulse-Error-Stage"] == "provider_response"
    assert headers["X-JournalPulse-Error-Fields"].split(",") == [
        "unrecognized_field.validation_error",
    ] * 5
    assert "PRIVATE" not in json.dumps(headers)
    assert "secret" not in json.dumps(headers)


@pytest.mark.parametrize("text,kind", [
    ("", "empty"),
    ("```json\nPRIVATE_MODEL_TEXT\n```", "markdown_fenced"),
    ("PRIVATE_MODEL_TEXT", "plain_text"),
    ('{"reply":"PRIVATE_MODEL_TEXT" trailing', "json_like"),
])
def test_invalid_json_shape_is_visible_without_response_text(tmp_path: Path, text: str, kind: str):
    with pytest.raises(ConversationProviderError) as caught:
        OpenRouterConversationClient(
            settings(tmp_path),
            client=httpx.Client(transport=httpx.MockTransport(lambda _: _response(text))),
        ).complete([{"role": "user", "content": "PRIVATE_USER_TEXT"}])
    headers = caught.value.diagnostic_headers
    assert headers["X-JournalPulse-Error-Content"] == kind
    assert headers["X-JournalPulse-Error-Finish"] == "stop"
    assert "PRIVATE" not in json.dumps(headers)


def test_provider_refusal_is_not_misreported_as_a_json_syntax_error(tmp_path: Path):
    def refused(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={
            "provider": "Azure", "choices": [{"finish_reason": "stop", "message": {
                "content": "", "refusal": "PRIVATE_REFUSAL_TEXT",
            }}],
        })
    with pytest.raises(ConversationProviderError) as caught:
        OpenRouterConversationClient(
            settings(tmp_path), client=httpx.Client(transport=httpx.MockTransport(refused)),
        ).complete([{"role": "user", "content": "PRIVATE_USER_TEXT"}])
    assert caught.value.diagnostic_headers["X-JournalPulse-Error-Stage"] == "provider_refusal"
    assert caught.value.status_code == 422
    assert caught.value.diagnostic_headers["X-JournalPulse-Error-Provider"] == "Azure"
    assert "PRIVATE" not in json.dumps(caught.value.diagnostic_headers)


class SteppedClock:
    """A controlled monotonic clock; each provider call advances it by a set amount."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def test_retries_share_one_budget_under_the_function_limit(tmp_path: Path):
    # Vercel stops the function at 120s. A slow first attempt must shrink the second
    # attempt's deadline instead of granting it another full chat timeout.
    clock = SteppedClock()
    timeouts: list[float] = []

    def slow(request: httpx.Request) -> httpx.Response:
        timeouts.append(request.extensions["timeout"]["connect"] or 0.0)
        clock.now += 60.0
        raise httpx.ReadTimeout("slow upstream")

    with pytest.raises(ConversationProviderError, match="respond in time"):
        OpenRouterConversationClient(
            settings(tmp_path, chat_timeout_seconds=45.0, openrouter_max_attempts=2),
            client=httpx.Client(transport=httpx.MockTransport(slow)),
            sleeper=lambda delay: setattr(clock, "now", clock.now + delay),
            clock=clock,
        ).complete([{"role": "user", "content": "Hello."}])
    assert len(timeouts) == 2
    worst_case = 60.0 + 0.15 + timeouts[1] + STREAM_READ_GRACE_SECONDS
    assert worst_case <= CHAT_PROVIDER_BUDGET_SECONDS


def test_no_retry_starts_when_the_budget_is_spent(tmp_path: Path):
    clock = SteppedClock()
    calls = 0

    def slow(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        clock.now += 94.0
        raise httpx.ReadTimeout("slow upstream")

    with pytest.raises(ConversationProviderError, match="respond in time"):
        OpenRouterConversationClient(
            settings(tmp_path, chat_timeout_seconds=95.0, openrouter_max_attempts=3),
            client=httpx.Client(transport=httpx.MockTransport(slow)),
            sleeper=lambda _delay: None,
            clock=clock,
        ).complete([{"role": "user", "content": "Hello."}])
    assert calls == 1


def test_rate_limit_retry_is_skipped_when_it_cannot_fit(tmp_path: Path):
    clock = SteppedClock()
    calls = 0

    def limited(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        clock.now += 92.0
        return httpx.Response(429, json={"error": {"message": "slow down", "code": 429}})

    with pytest.raises(ConversationProviderError, match="temporarily unavailable"):
        OpenRouterConversationClient(
            settings(tmp_path, openrouter_max_attempts=2),
            client=httpx.Client(transport=httpx.MockTransport(limited)),
            sleeper=lambda _delay: None,
            clock=clock,
        ).complete([{"role": "user", "content": "Hello."}])
    assert calls == 1
