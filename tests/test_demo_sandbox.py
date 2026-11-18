"""The demo sandbox: off by default, isolated, seeded with fiction, and never a provider request."""

from __future__ import annotations

from pathlib import Path
from uuid import UUID, uuid4

import httpx
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from journalpulse.api import create_app
from journalpulse.auth import LOCAL_DEMO_PREFIX, resolve_auth
from journalpulse.config import Settings
from journalpulse.demo import DEMO_ENTRIES, DEMO_LABEL, create_anonymous_supabase_user

CATALOG = Path(__file__).resolve().parents[1] / "assets/resources/catalog.json"


class ProviderMustNotBeCalled:
    """Stands in for every model and search client. Any use fails the test."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def __getattr__(self, name: str):
        def fail(*args, **kwargs):
            self.calls.append(name)
            raise AssertionError(f"a demo identity reached the provider: {name}")

        return fail


def settings(tmp_path: Path, **overrides) -> Settings:
    # AI and search are configured, so only the demo boundary keeps requests local.
    values = dict(
        environment="test",
        database_path=tmp_path / "demo.db",
        resource_catalog_path=CATALOG,
        openrouter_api_key="test-only-not-a-provider-key",
        openrouter_model="test-double",
        openrouter_base_url="https://provider.example",
        openrouter_zdr=True,
        openrouter_timeout_seconds=1,
        supabase_url=None,
        supabase_anon_key=None,
        raw_text_retention_default=False,
        analysis_rate_limit_per_minute=200,
        write_signing_key="test-only-demo-signing-key-0123456789abcdef",
        search_feature_enabled=True,
        search_api_key="test-only-search-key",
        demo_enabled=True,
    )
    values.update(overrides)
    return Settings(**values)


def app_and_provider(configured: Settings):
    provider = ProviderMustNotBeCalled()
    app = create_app(
        settings=configured,
        conversation_client=provider,  # type: ignore[arg-type]
        discovery_client=provider,  # type: ignore[arg-type]
        intelligence_client=provider,  # type: ignore[arg-type]
        activity_follow_up_generator=provider,  # type: ignore[arg-type]
    )
    return app, provider


def demo_headers(client: TestClient) -> dict[str, str]:
    response = client.post("/v1/demo/sessions")
    assert response.status_code == 201, response.text
    body = response.json()
    assert body["label"] == DEMO_LABEL
    assert body["user_id"].startswith(LOCAL_DEMO_PREFIX)
    assert body["access_token"] is None
    return {"X-JournalPulse-User": body["user_id"], "X-JournalPulse-Data-Revision": "0"}


def test_demo_route_is_absent_unless_enabled(tmp_path: Path) -> None:
    with TestClient(create_app(settings=settings(tmp_path, demo_enabled=False))) as client:
        assert client.post("/v1/demo/sessions").status_code == 404


def test_demo_session_is_seeded_with_three_entries_and_one_completed_simple_loop(tmp_path: Path) -> None:
    app, provider = app_and_provider(settings(tmp_path))
    with TestClient(app) as client:
        headers = demo_headers(client)
        entries = client.get("/v1/journal/entries", headers=headers).json()["items"]
        assert sorted(item["text"] for item in entries) == sorted(DEMO_ENTRIES)
        history = client.get("/v1/activity-history", headers=headers)
        assert history.status_code == 200, history.text
        items = history.json()["items"]
        assert len(items) == 1
        assert items[0]["participation"] == "completed"
    assert provider.calls == []


def test_two_demo_identities_are_isolated(tmp_path: Path) -> None:
    app, _ = app_and_provider(settings(tmp_path))
    with TestClient(app) as client:
        first, second = demo_headers(client), demo_headers(client)
        assert first != second
        entry_id = client.get("/v1/journal/entries", headers=first).json()["items"][0]["id"]
        assert client.get(f"/v1/journal/entries/{entry_id}", headers=second).status_code == 404
        assert client.delete(f"/v1/journal/entries/{entry_id}", headers=second).status_code == 404
        assert len(client.get("/v1/journal/entries", headers=first).json()["items"]) == 3


def test_demo_identity_asking_for_ai_gets_simple_mode_and_no_provider_call(tmp_path: Path) -> None:
    app, provider = app_and_provider(settings(tmp_path))
    with TestClient(app) as client:
        headers = demo_headers(client)
        chat = client.post(
            "/v1/conversations", headers=headers,
            json={"client_request_id": str(uuid4()), "llm_consent": True},
        )
        assert chat.status_code == 201, chat.text
        assert chat.json()["mode"] == "guided"
        assert chat.json()["llm_consent"] is False
        turn = client.post(
            f"/v1/conversations/{chat.json()['id']}/messages", headers=headers,
            json={"client_message_id": str(uuid4()), "text": "A fictional test message."},
        )
        assert turn.status_code == 200, turn.text
    assert provider.calls == []


def test_every_ai_and_search_route_refuses_a_demo_identity(tmp_path: Path) -> None:
    app, provider = app_and_provider(settings(tmp_path))
    with TestClient(app) as client:
        headers = demo_headers(client)
        entry_id = client.get("/v1/journal/entries", headers=headers).json()["items"][0]["id"]
        reflect = client.post(
            f"/v1/journal/entries/{entry_id}/reflect", headers=headers, json={"llm_consent": True},
        )
        assert reflect.status_code == 503
        search = client.post(
            "/v1/discovery/search", headers=headers,
            json={"original_query": "short calming walk", "llm_consent": True},
        )
        assert search.status_code == 403
        chat = client.post("/v1/conversations", headers=headers, json={"llm_consent": True}).json()
        inline = client.post(
            f"/v1/conversations/{chat['id']}/discover", headers=headers,
            json={"expected_revision": chat["revision"], "llm_consent": True},
        )
        assert inline.status_code == 409  # the demo chat holds no AI consent
        analysis = client.post(
            "/v1/reflections/analyze", headers=headers,
            json={"text": "A fictional calm afternoon.", "llm_consent": True},
        )
        assert analysis.status_code == 200, analysis.text
        assert analysis.json()["model_run"]["used_fallback"] is True
    assert provider.calls == []


def test_demo_creation_is_rate_limited(tmp_path: Path) -> None:
    app, _ = app_and_provider(settings(tmp_path, demo_creation_limit_per_hour=2))
    with TestClient(app) as client:
        assert client.post("/v1/demo/sessions").status_code == 201
        assert client.post("/v1/demo/sessions").status_code == 201
        limited = client.post("/v1/demo/sessions")
        assert limited.status_code == 429
        assert int(limited.headers["Retry-After"]) > 0


def test_invalid_demo_rate_limit_is_a_configuration_issue(tmp_path: Path) -> None:
    invalid = settings(tmp_path, demo_creation_limit_per_hour=0)
    assert "demo_rate_limit_invalid" in invalid.configuration_issues


def supabase_settings(tmp_path: Path) -> Settings:
    return settings(
        tmp_path, supabase_url="https://auth.example", supabase_anon_key="test-only-anon-key"
    )


CLAIMS = [(True, True), (False, False), (None, False), ("true", False)]


@pytest.mark.parametrize(("claim", "expected"), CLAIMS)
def test_supabase_anonymous_claim_marks_a_demo_identity(tmp_path: Path, claim, expected: bool) -> None:
    user_id = uuid4()

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/auth/v1/user"
        identity: dict = {"id": str(user_id)}
        if claim is not None:
            identity["is_anonymous"] = claim
        return httpx.Response(200, json=identity)

    with httpx.Client(transport=httpx.MockTransport(handler)) as transport:
        auth = resolve_auth(
            supabase_settings(tmp_path), authorization="Bearer token",
            development_user=None, client=transport,
        )
    assert auth.user_id == user_id and auth.demo is expected


def test_local_demo_prefix_marks_a_demo_identity(tmp_path: Path) -> None:
    configured = settings(tmp_path)
    demo_user = LOCAL_DEMO_PREFIX + "0000-4000-8000-000000000001"
    demo = resolve_auth(configured, authorization=None, development_user=demo_user)
    ordinary = resolve_auth(configured, authorization=None, development_user=str(uuid4()))
    assert demo.demo and not ordinary.demo


def test_supabase_demo_creation_requires_an_anonymous_session(tmp_path: Path) -> None:
    user_id = uuid4()
    replies = {
        "anonymous": {"access_token": "a", "refresh_token": "r", "expires_in": 3600,
                      "user": {"id": str(user_id), "is_anonymous": True}},
        "email": {"access_token": "a", "user": {"id": str(user_id), "is_anonymous": False}},
    }
    for kind, body in replies.items():
        def handler(request: httpx.Request, body=body) -> httpx.Response:
            assert request.url.path == "/auth/v1/signup"
            assert request.headers["apikey"] == "test-only-anon-key"
            return httpx.Response(200, json=body)

        with httpx.Client(transport=httpx.MockTransport(handler)) as transport:
            if kind == "anonymous":
                created = create_anonymous_supabase_user(supabase_settings(tmp_path), transport)
                assert created["user_id"] == UUID(str(user_id))
            else:
                with pytest.raises(HTTPException) as refused:
                    create_anonymous_supabase_user(supabase_settings(tmp_path), transport)
                assert refused.value.status_code == 503

    def disabled(request: httpx.Request) -> httpx.Response:
        return httpx.Response(422, json={"msg": "Anonymous sign-ins are disabled"})

    with httpx.Client(transport=httpx.MockTransport(disabled)) as transport:
        with pytest.raises(HTTPException):
            create_anonymous_supabase_user(supabase_settings(tmp_path), transport)
