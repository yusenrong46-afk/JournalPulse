"""Fresh-auth, self-only account closure with a narrowly scoped retry receipt.

The initial request verifies the session through Supabase before trusting AMR
claims. A pending/completed operation can be retried with its unguessable receipt
for 15 minutes, even after Auth deletion has invalidated the ordinary session.
Neither the wire request nor a retry selects an arbitrary user ID.
"""

from __future__ import annotations

import base64
import hashlib
import json
import re
from datetime import datetime
from typing import Literal
from uuid import UUID

import httpx
from fastapi import HTTPException
from pydantic import UUID4, BaseModel, ConfigDict, Field

from .auth import AuthContext, resolve_auth
from .config import Settings
from .http_clients import managed_http_client
from .signing import signed_payload

RECENT_AUTH_SECONDS = 300
AUTH_METHODS = {"password", "otp", "magiclink", "oauth", "sso/saml", "totp"}
CONTACT = "yusenrong46@gmail.com"


class AccountClosureRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    confirmation: Literal["delete my account"]
    request_id: UUID4 = Field(strict=False)
    receipt_secret: str = Field(pattern=r"^[a-f0-9]{64}$", repr=False)


def require_recent_auth(auth: AuthContext, now: datetime, issuer: str | None = None) -> None:
    """Only inspect claims after resolve_auth validated this exact token remotely.

    Refresh timestamps are not sign-in timestamps. Missing AMR never falls back to
    iat, user_metadata or the client's clock.
    """
    try:
        token = auth.access_token or ""
        if len(token) > 16_384:
            raise ValueError
        encoded = token.split(".")
        if len(encoded) != 3 or not re.fullmatch(r"[A-Za-z0-9_-]+", encoded[1]):
            raise ValueError
        claims = json.loads(base64.urlsafe_b64decode(encoded[1] + "=" * (-len(encoded[1]) % 4)))
        if (
            claims.get("sub") != str(auth.user_id)
            or claims.get("role") != "authenticated"
            or claims.get("is_anonymous") is not False
        ):
            raise ValueError
        if issuer is not None and claims.get("iss") != issuer:
            raise ValueError
        methods = claims.get("amr")
        if not isinstance(methods, list) or len(methods) > 20:
            raise ValueError
        stamps = [
            item["timestamp"]
            for item in methods
            if isinstance(item, dict)
            and item.get("method") in AUTH_METHODS
            and type(item.get("timestamp")) is int
        ]
        moment = int(now.timestamp())
        if not stamps or max(stamps) > moment + 30 or moment - max(stamps) > RECENT_AUTH_SECONDS:
            raise ValueError
    except (ValueError, TypeError, KeyError, AttributeError, RecursionError):
        raise HTTPException(
            status_code=403,
            detail=(
                "Please sign in again using a new email link, "
                "then confirm account closure within five minutes."
            ),
        ) from None


class AccountClosureService:
    def __init__(self, settings: Settings, client: httpx.Client | None = None):
        self.settings, self.client = settings, client

    def close(self, request: AccountClosureRequest, authorization: str | None, now: datetime) -> None:
        settings = self.settings
        if not settings.account_closure_configured:
            raise HTTPException(
                status_code=503, detail=f"Account closure is unavailable. Contact {CONTACT}."
            )
        assert settings.supabase_url and settings.supabase_service_role_key and settings.write_signing_key
        base = settings.supabase_url.rstrip("/")
        admin = {
            "apikey": settings.supabase_service_role_key,
            "Authorization": "Bearer " + settings.supabase_service_role_key,
        }
        receipt_hash = hashlib.sha256(request.receipt_secret.encode()).hexdigest()
        with managed_http_client(self.client, timeout=15) as client:

            def rpc(name: str, body: dict, *, headers: dict | None = None):
                try:
                    response = client.post(
                        base + "/rest/v1/rpc/" + name,
                        json=body,
                        headers=headers or admin,
                        timeout=10,
                        follow_redirects=False,
                    )
                except httpx.HTTPError:
                    raise unavailable() from None
                if response.status_code >= 400:
                    raise unavailable()
                if name == "jp_resolve_account_closure" and not response.content:
                    return None
                try:
                    return response.json()
                except (ValueError, RecursionError):
                    raise unavailable() from None

            # This lookup exposes no data without the random 256-bit operation receipt.
            operation = rpc(
                "jp_resolve_account_closure",
                {"operation_id": str(request.request_id), "proof_hash": receipt_hash},
            )
            resumed = operation is not None
            if operation is None:
                auth = resolve_auth(
                    settings, authorization=authorization, development_user=None, client=client
                )
                require_recent_auth(auth, now, base + "/auth/v1")
                body = signed_payload(
                    "begin_account_closure",
                    auth.user_id,
                    {"request_id": str(request.request_id), "receipt_hash": receipt_hash},
                    settings.write_signing_key,
                )
                operation = rpc(
                    "jp_begin_account_closure",
                    body,
                    headers={
                        "apikey": settings.supabase_anon_key or "",
                        "Authorization": "Bearer " + (auth.access_token or ""),
                    },
                )
                if not isinstance(operation, dict) or operation.get("user_id") != str(auth.user_id):
                    raise unavailable()
            try:
                owner = UUID(operation["user_id"])
                state = operation["state"]
                if operation["request_id"] != str(request.request_id) or state not in {"pending", "complete"}:
                    raise ValueError
            except (ValueError, TypeError, KeyError):
                raise unavailable() from None
            if authorization and resumed:
                try:
                    current = resolve_auth(
                        settings, authorization=authorization, development_user=None, client=client
                    )
                except HTTPException as error:
                    if error.status_code != 401:
                        raise
                else:
                    if current.user_id != owner:
                        raise HTTPException(
                            status_code=403, detail="This closure request belongs to another session."
                        )
            if state == "complete":
                return
            # The durable pending row blocks writes before any administrative deletion.
            try:
                response = client.request(
                    "DELETE",
                    base + f"/auth/v1/admin/users/{owner}",
                    headers=admin,
                    json={"should_soft_delete": False},
                    timeout=15,
                    follow_redirects=False,
                )
                if response.status_code not in {200, 204, 404}:
                    raise unavailable()
                if response.status_code == 200:
                    data = response.json()
                    returned = data.get("user", data) if isinstance(data, dict) else None
                    if not isinstance(returned, dict) or returned.get("id") != str(owner):
                        raise unavailable()
                verification = client.get(
                    base + f"/auth/v1/admin/users/{owner}", headers=admin, timeout=10, follow_redirects=False
                )
            except (httpx.HTTPError, ValueError, RecursionError):
                # A timeout might follow a successful deletion. Retain the operation
                # receipt; a deliberate retry can finish even if the user no longer exists.
                raise unavailable() from None
            if verification.status_code != 404:
                raise unavailable()
            finished = rpc(
                "jp_finish_account_closure",
                {"operation_id": str(request.request_id), "proof_hash": receipt_hash},
            )
            if not isinstance(finished, dict) or finished.get("state") != "complete":
                raise unavailable()


def unavailable() -> HTTPException:
    return HTTPException(
        status_code=503,
        detail=(
            "Account closure could not be confirmed. If closure started, new writes are paused. "
            "Keep this page open and retry the same request, or contact " + CONTACT + "."
        ),
    )
