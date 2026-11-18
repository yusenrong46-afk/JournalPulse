"""Supabase-shaped auth stand-in for `make onboard` (local environments only).

It does the three things the app and the exported web need from Supabase Auth, and no more:

* ``POST /auth/v1/signup`` with an empty body: an anonymous user (the portfolio demo), inserted
  into ``auth.users`` with ``is_anonymous = true`` and returned with an HS256 token that
  PostgREST accepts.
* ``GET /auth/v1/user``: the identity behind a token (the API calls this on every request).
* ``/rest/v1/*``: a pass-through to PostgREST.

Email sign-in is not offered: an onboarded environment is for the demo journey and for
development, and hosted environments use real Supabase Auth. Standard library plus httpx and
the ``psql`` client; configured only through the environment:

    JP_AUTH_JWT_SECRET    HS256 secret shared with PostgREST (required)
    JP_AUTH_DATABASE_URL  libpq URI of the environment database (required)
    JP_POSTGREST_URL      PostgREST base URL (required)
    JP_AUTH_PORT          listen port (default 54321)
    JP_AUTH_BIND          listen address (default 127.0.0.1; the container uses 0.0.0.0)
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import subprocess
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from uuid import UUID, uuid4

import httpx

TOKEN_SECONDS = 86_400


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def mint(secret: str, claims: dict) -> str:
    header = _b64(json.dumps({"alg": "HS256", "typ": "JWT"}).encode())
    body = _b64(json.dumps(claims).encode())
    signature = hmac.new(secret.encode(), f"{header}.{body}".encode(), hashlib.sha256).digest()
    return f"{header}.{body}.{_b64(signature)}"


def verify(secret: str, token: str) -> dict | None:
    try:
        header, body, signature = token.split(".")
        expected = hmac.new(secret.encode(), f"{header}.{body}".encode(), hashlib.sha256).digest()
        if not hmac.compare_digest(_b64(expected), signature):
            return None
        claims = json.loads(base64.urlsafe_b64decode(body + "=" * (-len(body) % 4)))
    except (ValueError, json.JSONDecodeError):
        return None
    return claims if isinstance(claims, dict) and claims.get("exp", 0) > time.time() else None


def anon_key(secret: str, *, days: int = 30) -> str:
    """The public client key: a token for PostgREST's anon role."""
    return mint(secret, {"role": "anon", "exp": int(time.time()) + days * 86_400})


def _psql(database_url: str, sql: str) -> None:
    completed = subprocess.run(
        ["psql", database_url, "-v", "ON_ERROR_STOP=1", "-X", "-q"],
        input=sql, text=True, capture_output=True, check=False,
    )
    if completed.returncode != 0:
        # The URI holds a password: report only that the insert failed.
        raise RuntimeError("auth.users insert failed")


def make_handler(secret: str, database_url: str, postgrest_url: str) -> type[BaseHTTPRequestHandler]:
    expected_anon_role = "anon"

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, format: str, *args: object) -> None:  # noqa: A002 - stdlib signature
            return  # never log paths or bodies

        def _send(self, status: int, body: bytes, content_type: str = "application/json", extra=()) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            for key, value in extra:
                self.send_header(key, value)
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass  # the caller gave up first (its timeout is shorter); nothing to deliver

        def _json(self, status: int, payload: object) -> None:
            self._send(status, json.dumps(payload).encode())

        def _body(self) -> bytes | None:
            length = int(self.headers.get("Content-Length") or 0)
            return self.rfile.read(length) if length else None

        def do_OPTIONS(self) -> None:  # noqa: N802 - stdlib naming
            self._send(204, b"", extra=[
                ("Access-Control-Allow-Headers", "*"), ("Access-Control-Allow-Methods", "*"),
            ])

        def _proxy(self) -> None:
            headers = {
                key: value for key, value in self.headers.items()
                if key.lower() in {"authorization", "content-type", "prefer", "accept", "range"}
            }
            try:
                response = httpx.request(
                    self.command, f"{postgrest_url}{self.path.removeprefix('/rest/v1')}",
                    headers=headers, content=self._body(), timeout=15,
                )
            except httpx.HTTPError:
                # Like Supabase's gateway: an unreachable database is a 5xx, not a dropped connection.
                return self._json(504, {"message": "upstream timeout"})
            passthrough = [(k, v) for k, v in response.headers.items() if k.lower() == "content-range"]
            self._send(
                response.status_code, response.content,
                response.headers.get("content-type", "application/json"), passthrough,
            )

        def _route(self) -> None:
            if self.path.startswith("/rest/v1"):
                return self._proxy()
            if self.path == "/health":
                return self._json(200, {"status": "ok"})
            if self.path == "/auth/v1/user" and self.command == "GET":
                token = (self.headers.get("Authorization") or "").removeprefix("Bearer ").strip()
                claims = verify(secret, token)
                if not claims or claims.get("role") != "authenticated":
                    return self._json(401, {"message": "invalid token"})
                return self._json(200, {
                    "id": claims["sub"], "aud": "authenticated", "role": "authenticated",
                    "is_anonymous": claims.get("is_anonymous") is True,
                })
            if self.path == "/auth/v1/signup" and self.command == "POST":
                key = verify(secret, self.headers.get("apikey") or "")
                if not key or key.get("role") != expected_anon_role:
                    return self._json(401, {"message": "invalid apikey"})
                body = self._body()
                request = json.loads(body) if body else {}
                if request.get("email") or request.get("phone"):
                    return self._json(422, {"message": "only anonymous sign-ins are available here"})
                user_id = UUID(str(uuid4()))  # a UUID literal: nothing from the request reaches SQL
                _psql(database_url, f"insert into auth.users (id, is_anonymous) values ('{user_id}', true);")
                token = mint(secret, {
                    "sub": str(user_id), "role": "authenticated", "aud": "authenticated",
                    "is_anonymous": True, "exp": int(time.time()) + TOKEN_SECONDS,
                })
                return self._json(200, {
                    "access_token": token, "token_type": "bearer", "expires_in": TOKEN_SECONDS,
                    "refresh_token": f"onboarding-refresh-{user_id}",
                    "user": {
                        "id": str(user_id), "aud": "authenticated", "role": "authenticated",
                        "is_anonymous": True,
                    },
                })
            return self._json(404, {"message": "not found"})

        do_GET = do_POST = do_PATCH = do_DELETE = _route  # noqa: N815

    return Handler


def main() -> None:
    secret = os.environ["JP_AUTH_JWT_SECRET"]
    if len(secret) < 32:
        raise SystemExit("JP_AUTH_JWT_SECRET must be at least 32 characters.")
    handler = make_handler(
        secret, os.environ["JP_AUTH_DATABASE_URL"], os.environ["JP_POSTGREST_URL"].rstrip("/")
    )
    server = ThreadingHTTPServer(
        (os.getenv("JP_AUTH_BIND", "127.0.0.1"), int(os.getenv("JP_AUTH_PORT", "54321"))), handler
    )
    print("onboarding auth stand-in ready", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
