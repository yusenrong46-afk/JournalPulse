import asyncio
from dataclasses import replace

import httpx
from fastapi.testclient import TestClient

from journalpulse.api import create_app
from journalpulse.middleware import SlidingWindowRateLimiter
from test_research_beta_api import settings


def test_chunked_request_uses_the_same_size_error_as_content_length(tmp_path):
    configured = replace(settings(tmp_path), max_request_bytes=5_000)
    app = create_app(settings=configured)

    async def oversized_body():
        yield b'{"text":"' + b"a" * 3_000
        yield b"b" * 3_000 + b'","llm_consent":false}'

    async def request():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            return await client.post(
                "/v1/reflections/analyze", content=oversized_body(),
                headers={"Content-Type": "application/json", "X-Request-ID": "chunked-test-123"},
            )

    response = asyncio.run(request())
    assert response.status_code == 413
    assert response.json()["code"] == "request_too_large"
    assert response.headers["x-request-id"] == "chunked-test-123"


def test_negative_content_length_is_rejected(tmp_path):
    with TestClient(create_app(settings=settings(tmp_path))) as client:
        response = client.post("/v1/reflections/analyze", headers={"Content-Length": "-1"})
    assert response.status_code == 400
    assert response.json()["code"] == "invalid_content_length"


def test_small_chunked_json_is_replayed_intact(tmp_path):
    async def body():
        yield b'{"text":"A quiet walk was useful today.",'
        yield b'"llm_consent":false,"locale":"CA"}'

    async def request():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=create_app(settings=settings(tmp_path))),
            base_url="http://test",
        ) as client:
            return await client.post(
                "/v1/reflections/analyze", content=body(),
                headers={"Content-Type": "application/json"},
            )

    response = asyncio.run(request())
    assert response.status_code == 200
    assert response.json()["model_run"]["used_fallback"] is True


def test_rate_limiter_reclaims_expired_users_without_resetting_active_limits():
    limiter = SlidingWindowRateLimiter(limit=2)
    for number in range(1_000):
        assert limiter.check(f"inactive-{number}", now=0)[0]
    assert limiter.check("active", now=59)[0]
    assert limiter.check("active", now=60)[0]
    assert limiter.check("active", now=61) == (False, 58)
    # Expired identities must not accumulate for the lifetime of a server process.
    assert set(limiter._events) == {"active"}
