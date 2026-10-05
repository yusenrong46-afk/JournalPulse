from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

import httpx
from fastapi import HTTPException

from .config import Settings
from .http_clients import managed_http_client


@dataclass(frozen=True)
class AuthContext:
    user_id: UUID
    access_token: str | None = None


def resolve_auth(
    settings: Settings,
    *,
    authorization: str | None,
    development_user: str | None,
    client: httpx.Client | None = None,
) -> AuthContext:
    if settings.supabase_enabled:
        assert settings.supabase_url is not None
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Missing bearer token")
        token = authorization.removeprefix("Bearer ").strip()
        if not token:
            raise HTTPException(status_code=401, detail="Missing bearer token")
        try:
            with managed_http_client(client, timeout=10) as transport:
                response = transport.get(
                    f"{settings.supabase_url.rstrip('/')}/auth/v1/user",
                    headers={
                        "apikey": settings.supabase_anon_key or "",
                        "Authorization": f"Bearer {token}",
                    },
                    timeout=10,
                    follow_redirects=False,
                )
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=503, detail="Authentication service unavailable") from exc
        if response.status_code in {401, 403}:
            raise HTTPException(status_code=401, detail="Invalid or expired session")
        if response.status_code != 200:
            # An upstream outage or throttle says nothing about this person's JWT.
            raise HTTPException(status_code=503, detail="Authentication service unavailable")
        try:
            identity = response.json()
            if not isinstance(identity, dict) or not isinstance(identity.get("id"), str):
                raise ValueError("Invalid authentication identity")
            user_id = UUID(identity["id"])
        except (ValueError, RecursionError) as exc:
            raise HTTPException(status_code=503, detail="Authentication service unavailable") from exc
        return AuthContext(user_id=user_id, access_token=token)

    if settings.environment == "production":
        raise HTTPException(status_code=503, detail="Supabase authentication is required in production")
    try:
        return AuthContext(user_id=UUID(development_user or "00000000-0000-4000-8000-000000000001"))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid development user ID") from exc
