"""Exercise transport ownership without calling Supabase or a paid provider."""

import json
from pathlib import Path
from uuid import UUID

import httpx
import pytest
from fastapi import HTTPException

from journalpulse.auth import resolve_auth
from journalpulse.config import Settings
from journalpulse.intelligence import (
    ConversationProviderError,
    OpenRouterConversationClient,
    OpenRouterReflectionClient,
    safe_analyze,
)
from journalpulse.persistence import StorageUnavailable, SupabaseRepository, supabase_readiness

USER_ID = "00000000-0000-4000-8000-000000000009"


def configured(tmp_path: Path) -> Settings:
    return Settings(
        environment="test",
        database_path=tmp_path / "unused.db",
        resource_catalog_path=Path("unused.json"),
        openrouter_api_key="test-only-key",
        openrouter_model="openai/gpt-6-luna",
        openrouter_base_url="https://openrouter.ai/api/v1",
        openrouter_zdr=True,
        openrouter_timeout_seconds=2,
        openrouter_max_attempts=1,
        supabase_url="https://project.supabase.co",
        supabase_anon_key="public-anon-key",
        raw_text_retention_default=False,
    )


def completion(kind: str) -> dict:
    content = {
        "reply": "You enjoyed the quiet.", "offer_action": False,
        "resource_intent": "reflect", "card_reason": "", "summary": "A quiet walk.",
        "feelings": [],
    } if kind == "conversation" else {
        "valence": 0.2, "arousal": 0.3, "agency": 0.5, "emotion_tags": ["calm"],
        "confidence": 0.8, "uncertainty": None, "summary": "A quiet walk.",
        "interpretation": "The walk felt peaceful.", "reflection_question": "What felt quiet?",
        "resource_intent": "reflect",
    }
    return {"choices": [{"message": {"content": json.dumps(content)}}]}


@pytest.mark.parametrize("kind", ["auth", "readiness", "repository", "reflection", "conversation"])
@pytest.mark.parametrize("failure", [False, True])
@pytest.mark.parametrize("borrowed", [False, True])
def test_http_clients_close_owned_transports_and_preserve_borrowed_ones(
    tmp_path: Path, monkeypatch, kind: str, failure: bool, borrowed: bool,
):
    created: list[httpx.Client] = []
    original_client = httpx.Client

    def handler(_: httpx.Request) -> httpx.Response:
        if failure:
            raise httpx.ConnectError("test transport unavailable")
        body = {
            "auth": {"id": USER_ID},
            "readiness": {"schema": "phase-a-2", "signing": "valid"},
            "repository": [],
        }.get(kind, completion(kind))
        return httpx.Response(200, json=body)

    def factory(*args, **kwargs) -> httpx.Client:
        kwargs["transport"] = httpx.MockTransport(handler)
        transport = original_client(*args, **kwargs)
        created.append(transport)
        return transport

    supplied = factory() if borrowed else None
    monkeypatch.setattr(httpx, "Client", factory)
    settings = configured(tmp_path)

    def operation():
        if kind == "auth":
            return resolve_auth(
                settings, authorization="Bearer verified-session", development_user=None,
                client=supplied,
            )
        if kind == "readiness":
            return supabase_readiness(settings, client=supplied)
        if kind == "repository":
            return SupabaseRepository(settings, "verified-session", client=supplied).list_journal_entries(
                UUID(USER_ID)
            )
        if kind == "reflection":
            return OpenRouterReflectionClient(settings, client=supplied).analyze("A quiet walk.", {})
        return OpenRouterConversationClient(settings, client=supplied).complete(
            [{"role": "user", "content": "A quiet walk."}]
        )

    try:
        if failure and kind != "readiness":
            expected_errors = (HTTPException, StorageUnavailable, httpx.HTTPError, ConversationProviderError)
            with pytest.raises(expected_errors):
                operation()
        else:
            result = operation()
            if failure:
                assert result == {"database": "unreachable"}
        assert len(created) == 1
        assert created[0].is_closed is not borrowed
    finally:
        # Close borrowed test transports after checking that production code did not.
        for transport in created:
            transport.close()


@pytest.mark.parametrize("status", [302, 429, 500, 503])
def test_auth_outage_does_not_report_a_bad_session(tmp_path: Path, status: int):
    with httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(status))) as client:
        with pytest.raises(HTTPException) as failure:
            resolve_auth(
                configured(tmp_path), authorization="Bearer verified-session",
                development_user=None, client=client,
            )
    assert failure.value.status_code == 503


@pytest.mark.parametrize("body", [b"not json", b"[]", b"{}", b'{"id":null}', b'{"id":"bad-uuid"}'])
def test_malformed_auth_identity_is_an_auth_service_failure(tmp_path: Path, body: bytes):
    with httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(200, content=body))) as client:
        with pytest.raises(HTTPException) as failure:
            resolve_auth(
                configured(tmp_path), authorization="Bearer verified-session",
                development_user=None, client=client,
            )
    assert failure.value.status_code == 503


def test_empty_bearer_token_is_rejected_without_a_remote_request(tmp_path: Path):
    calls = []
    with httpx.Client(transport=httpx.MockTransport(lambda request: calls.append(request))) as client:
        with pytest.raises(HTTPException) as failure:
            resolve_auth(
                configured(tmp_path), authorization="Bearer   ", development_user=None, client=client,
            )
    assert failure.value.status_code == 401
    assert calls == []


@pytest.mark.parametrize("body", [b"not json", b"[]", b"null"])
def test_malformed_readiness_response_fails_closed(tmp_path: Path, body: bytes):
    with httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(200, content=body))) as client:
        assert supabase_readiness(configured(tmp_path), client=client) == {"database": "invalid_response"}


@pytest.mark.parametrize("body", [
    [], {"choices": []}, {"choices": None}, {"choices": [None]},
    {"choices": [{"message": None}]}, {**completion("reflection"), "usage": None},
])
def test_malformed_legacy_reflection_envelope_uses_its_documented_fallback(tmp_path: Path, body: object):
    with httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(200, json=body))) as client:
        result = safe_analyze(
            configured(tmp_path), text="A quiet walk.", context={}, consent=True, self_report=None,
            client=OpenRouterReflectionClient(configured(tmp_path), client=client),
        )
    assert result.model_run.used_fallback
    assert result.model_run.fallback_reason == "openrouter_ValueError"


@pytest.mark.parametrize("kind", ["auth", "readiness", "repository", "reflection", "conversation"])
def test_deeply_nested_provider_json_remains_a_controlled_failure(tmp_path: Path, kind: str):
    # This valid JSON is well below the model byte cap but exceeds decoder stack depth.
    body = b"[" * 10_000 + b"0" + b"]" * 10_000
    with httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(200, content=body))) as client:
        settings = configured(tmp_path)
        if kind == "auth":
            with pytest.raises(HTTPException) as failure:
                resolve_auth(
                    settings, authorization="Bearer verified-session", development_user=None, client=client,
                )
            assert failure.value.status_code == 503
        elif kind == "readiness":
            assert supabase_readiness(settings, client=client) == {"database": "invalid_response"}
        elif kind == "repository":
            with pytest.raises(StorageUnavailable, match="invalid response"):
                repository = SupabaseRepository(settings, "verified-session", client=client)
                repository.list_journal_entries(UUID(USER_ID))
        elif kind == "reflection":
            result = safe_analyze(
                settings, text="A quiet walk.", context={}, consent=True, self_report=None,
                client=OpenRouterReflectionClient(settings, client=client),
            )
            assert result.model_run.used_fallback
        else:
            with pytest.raises(ConversationProviderError) as failure:
                OpenRouterConversationClient(settings, client=client).complete(
                    [{"role": "user", "content": "A quiet walk."}]
                )
            assert failure.value.status_code == 502


def test_repository_invalid_json_becomes_a_storage_failure(tmp_path: Path):
    transport = httpx.MockTransport(lambda _: httpx.Response(200, text="not json"))
    with httpx.Client(transport=transport) as client:
        with pytest.raises(StorageUnavailable, match="invalid response"):
            SupabaseRepository(configured(tmp_path), "verified-session", client=client).list_journal_entries(
                UUID(USER_ID)
            )


@pytest.mark.parametrize("kind", ["reflection", "conversation"])
def test_owned_provider_transport_is_reused_across_retries_then_closed(
    tmp_path: Path, monkeypatch, kind: str,
):
    calls = []
    created = []
    original_client = httpx.Client

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(503) if len(calls) == 1 else httpx.Response(200, json=completion(kind))

    def factory(*args, **kwargs):
        client = original_client(*args, **kwargs, transport=httpx.MockTransport(handler))
        created.append(client)
        return client

    monkeypatch.setattr(httpx, "Client", factory)
    settings = configured(tmp_path)
    settings = Settings(**{**settings.__dict__, "openrouter_max_attempts": 2})
    if kind == "reflection":
        OpenRouterReflectionClient(settings, sleeper=lambda _: None).analyze("A quiet walk.", {})
    else:
        OpenRouterConversationClient(settings, sleeper=lambda _: None).complete(
            [{"role": "user", "content": "A quiet walk."}]
        )
    assert len(calls) == 2
    assert len(created) == 1
    assert created[0].is_closed


@pytest.mark.parametrize("finish_reason,refusal", [
    ("stop", "This request was declined."), ("content_filter", None), ("length", None),
])
def test_legacy_native_decline_or_truncation_cannot_be_marked_as_verified_ai(
    tmp_path: Path, finish_reason: str, refusal: str | None,
):
    calls = []
    body = completion("reflection")
    body["choices"][0]["finish_reason"] = finish_reason
    body["choices"][0]["message"]["refusal"] = refusal

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, json=body)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        settings = configured(tmp_path)
        result = safe_analyze(
            settings, text="A quiet walk.", context={}, consent=True, self_report=None,
            client=OpenRouterReflectionClient(settings, client=client),
        )
    assert result.model_run.used_fallback
    assert result.model_run.model == "deterministic-fallback"
    assert result.model_run.fallback_reason == "openrouter_ValueError"
    assert len(calls) == 1


def test_legacy_native_refusal_takes_priority_over_truncated_or_malformed_content(tmp_path: Path):
    body = {"choices": [{
        "finish_reason": "length",
        "message": {"refusal": "This request was declined.", "content": {"not": "text"}},
    }]}
    with httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(200, json=body))) as client:
        with pytest.raises(ValueError, match="declined"):
            OpenRouterReflectionClient(configured(tmp_path), client=client).analyze("A quiet walk.", {})
