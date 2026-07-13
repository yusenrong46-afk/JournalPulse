from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

import httpx
from fastapi import HTTPException

from .config import Settings


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
        try:
            response = (client or httpx.Client(timeout=10)).get(
                f"{settings.supabase_url.rstrip('/')}/auth/v1/user",
                headers={
                    "apikey": settings.supabase_anon_key or "",
                    "Authorization": f"Bearer {token}",
                },
            )
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=503, detail="Authentication service unavailable") from exc
        if response.status_code != 200:
            raise HTTPException(status_code=401, detail="Invalid or expired session")
        return AuthContext(user_id=UUID(response.json()["id"]), access_token=token)

    if settings.environment == "production":
        raise HTTPException(status_code=503, detail="Supabase authentication is required in production")
    try:
        return AuthContext(user_id=UUID(development_user or "00000000-0000-4000-8000-000000000001"))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid development user ID") from exc
