"""Portfolio demo sandbox: an email-less identity with fictional data and AI off.

``POST /v1/demo/sessions`` exists only when ``JOURNALPULSE_DEMO_ENABLED`` is true.
It creates an isolated identity, seeds it through the app's own public routes, and
returns what the browser needs to act as that identity.

* Hosted (Supabase): the identity is a Supabase anonymous user. Row-level security
  applies exactly as for an email account; the API marks it ``demo`` from the
  ``is_anonymous`` claim, which the browser cannot set.
* Local (SQLite): the identity is a UUID with :data:`LOCAL_DEMO_PREFIX`. Local
  development already trusts the user header, so this is only a marker.

Demo identities never reach a model or search provider: see ``AuthContext.demo``.
Seed text is invented for this file and contains no real person's writing.
Hosted demo accounts are erased after 24 hours by the scheduled job in
``supabase/migrations/202611180001_demo_sandbox.sql``, which calls the existing
``delete_my_journalpulse_data`` erasure function for each expired anonymous user.
"""

from __future__ import annotations

import secrets
from collections.abc import Callable
from typing import Any
from uuid import UUID, uuid4

import httpx
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

from .auth import LOCAL_DEMO_PREFIX
from .config import Settings
from .http_clients import managed_http_client
from .middleware import SlidingWindowRateLimiter

DEMO_LABEL = "Demo: fictional data, AI off"
DEMO_RESOURCE_ID = "guided_meditation_2m"
# Fictional entries written for the demo. Keep them obviously invented and harmless.
DEMO_ENTRIES = (
    "Demo entry (fictional): Watered the balcony tomatoes before work. Two of them are finally "
    "turning red. Small thing, but it set a calm tone for the morning.",
    "Demo entry (fictional): The bus was late and I missed the start of the team meeting. I felt "
    "rushed all afternoon. Tomorrow I will leave ten minutes earlier.",
    "Demo entry (fictional): Called my cousin about the weekend picnic plan. We picked the park by "
    "the river. Looking forward to a slow Saturday.",
)
DEMO_CHAT_MESSAGE = "Demo message (fictional): I feel a bit scattered after a busy day."


class DemoSessionResponse(BaseModel):
    user_id: UUID
    label: str = DEMO_LABEL
    # Present only for hosted (Supabase) demo identities.
    access_token: str | None = None
    refresh_token: str | None = None
    expires_in: int | None = None


def new_local_demo_user() -> UUID:
    return UUID(LOCAL_DEMO_PREFIX + str(uuid4())[len(LOCAL_DEMO_PREFIX) :])


def create_anonymous_supabase_user(settings: Settings, client: httpx.Client | None = None) -> dict[str, Any]:
    """Create one Supabase anonymous user (no email). Anonymous sign-ins must be enabled."""
    assert settings.supabase_url is not None
    try:
        with managed_http_client(client, timeout=10) as transport:
            response = transport.post(
                f"{settings.supabase_url.rstrip('/')}/auth/v1/signup",
                headers={"apikey": settings.supabase_anon_key or "", "Content-Type": "application/json"},
                json={"data": {}},
                timeout=10,
                follow_redirects=False,
            )
    except httpx.HTTPError as exc:
        raise HTTPException(503, "The demo is unavailable right now. Please try again later.") from exc
    if response.status_code != 200:
        # Anonymous sign-ins disabled, or Supabase's own anonymous rate limit reached.
        raise HTTPException(503, "The demo is unavailable right now. Please try again later.")
    try:
        body = response.json()
        user = body["user"]
        if user.get("is_anonymous") is not True or not body.get("access_token"):
            raise ValueError("not an anonymous session")
        return {
            "user_id": UUID(user["id"]),
            "access_token": body["access_token"],
            "refresh_token": body.get("refresh_token"),
            "expires_in": body.get("expires_in"),
        }
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        raise HTTPException(503, "The demo is unavailable right now. Please try again later.") from exc


def seed_demo_account(call: Callable[..., httpx.Response]) -> None:
    """Seed three fictional entries and one completed Simple Luna activity loop.

    ``call(method, path, json=...)`` sends a request to this same API as the demo
    identity, so seeding uses the public routes and their ownership checks rather
    than a second implementation.
    """

    def ok(response: httpx.Response) -> dict[str, Any]:
        if response.status_code >= 400:
            raise HTTPException(503, "The demo could not be prepared. Please try again.")
        return response.json()

    for text in DEMO_ENTRIES:
        ok(call("POST", "/v1/journal/entries", json={"client_request_id": str(uuid4()), "text": text}))
    chat = ok(call(
        "POST", "/v1/conversations", json={"client_request_id": str(uuid4()), "llm_consent": False},
    ))
    turn = ok(call(
        "POST", f"/v1/conversations/{chat['id']}/messages",
        json={"client_message_id": str(uuid4()), "text": DEMO_CHAT_MESSAGE, "goal": "settle"},
    ))
    session = ok(call(
        "POST", f"/v1/conversations/{chat['id']}/activity-sessions",
        json={
            "client_request_id": str(uuid4()),
            "expected_conversation_revision": turn["conversation"]["revision"],
            "resource_id": DEMO_RESOURCE_ID,
            "user_selected": True,
        },
    ))
    for command in ("start", "finish_early"):
        session = ok(call(
            "POST", f"/v1/activity-sessions/{session['id']}/commands",
            json={
                "client_request_id": str(uuid4()),
                "expected_revision": session["revision"],
                "expected_conversation_revision": session["conversation_revision"],
                "command": command,
            },
        ))
    ok(call(
        "POST", f"/v1/activity-sessions/{session['id']}/report",
        json={
            "client_request_id": str(uuid4()),
            "expected_revision": session["revision"],
            "expected_conversation_revision": session["conversation_revision"],
            "participation": "completed",
            "state_change": "toward_target",
            "note": "Demo note (fictional): the pause helped me slow down.",
        },
    ))


def register_demo_routes(app: FastAPI, *, settings: Settings) -> None:
    # Per-instance guard on creation by client address. Hosted deployments also rely on
    # Supabase's own anonymous sign-in rate limit, which is shared across instances.
    limiter = SlidingWindowRateLimiter(limit=settings.demo_creation_limit_per_hour, window_seconds=3600)

    @app.post("/v1/demo/sessions", response_model=DemoSessionResponse, status_code=201)
    def create_demo_session(request: Request) -> DemoSessionResponse:
        if not settings.demo_enabled:
            raise HTTPException(404, "Not Found")
        address = request.client.host if request.client else "unknown"
        allowed, retry_after = limiter.check(address)
        if not allowed:
            raise HTTPException(
                429, "Too many demo sessions from this network. Please try again later.",
                headers={"Retry-After": str(retry_after)},
            )
        if settings.supabase_enabled:
            created = create_anonymous_supabase_user(settings)
            headers = {"Authorization": f"Bearer {created['access_token']}"}
            result = DemoSessionResponse(
                user_id=created["user_id"], access_token=created["access_token"],
                refresh_token=created["refresh_token"], expires_in=created["expires_in"],
            )
        else:
            user_id = new_local_demo_user()
            headers = {"X-JournalPulse-User": str(user_id)}
            result = DemoSessionResponse(user_id=user_id)
        # A brand-new identity has data revision 0; mutations must state it.
        headers["X-JournalPulse-Data-Revision"] = "0"
        # Imported here: the test client lives in Starlette, needed only for in-process seeding.
        from fastapi.testclient import TestClient

        # No context manager: the app's lifespan is already running in this process.
        client = TestClient(app, base_url="http://127.0.0.1", raise_server_exceptions=False)

        def call(method: str, path: str, **kwargs: Any) -> httpx.Response:
            return client.request(
                method, path, headers={**headers, "X-Request-ID": secrets.token_hex(8)}, **kwargs
            )

        seed_demo_account(call)
        return result
