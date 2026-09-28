"""Run a local stand-in for production for the browser integration tests.

Real: PostgreSQL with every migration applied, PostgREST 12 (the REST layer Supabase uses),
the FastAPI app with its Supabase adapter, and the exported Next.js site served by FastAPI.
Stand-ins: the Supabase auth endpoint (it verifies JWTs minted here instead of GoTrue) and
the model provider (a deterministic fake, so CI never makes a paid call).

    uv run python scripts/integration_stack.py      # Playwright starts this itself

Uses JOURNALPULSE_PG_DSN like scripts/verify_postgres_schema.py, or local peer auth.
"""

from __future__ import annotations

import atexit
import base64
import hashlib
import hmac
import json
import os
import shutil
import signal
import subprocess
import sys
import tarfile
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit
from uuid import UUID

import httpx
import uvicorn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from journalpulse.api import create_app  # noqa: E402
from journalpulse.config import Settings  # noqa: E402
from journalpulse.domain import ModelRun  # noqa: E402
from journalpulse.guided import guess_feelings  # noqa: E402
from journalpulse.intelligence import ConversationCompletion  # noqa: E402

DATABASE = "jp_integration"
JWT_SECRET = "integration-only-jwt-secret-0123456789abcdef"
SIGNING_KEY = "integration-only-signing-key-0123456789abcdef"
AUTHENTICATOR_PASSWORD = "integration-authenticator"
POSTGREST_VERSION = "v12.2.12"
POSTGREST_PORT = int(os.getenv("JP_POSTGREST_PORT", "3301"))
GATEWAY_PORT = int(os.getenv("JP_GATEWAY_PORT", "54321"))
API_PORT = int(os.getenv("JP_API_PORT", "8100"))
GATEWAY_URL = f"http://127.0.0.1:{GATEWAY_PORT}"
WEB_DIST = ROOT / "web" / "out"
USERS = {
    "alex": UUID("a1a1a1a1-0000-4000-8000-000000000001"),
    "blair": UUID("b2b2b2b2-0000-4000-8000-000000000002"),
    "casey": UUID("c3c3c3c3-0000-4000-8000-000000000003"),
    "dana": UUID("d4d4d4d4-0000-4000-8000-000000000004"),
}


def b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def mint(claims: dict) -> str:
    header = b64(json.dumps({"alg": "HS256", "typ": "JWT"}).encode())
    body = b64(json.dumps(claims).encode())
    signature = hmac.new(JWT_SECRET.encode(), f"{header}.{body}".encode(), hashlib.sha256).digest()
    return f"{header}.{body}.{b64(signature)}"


def verify(token: str) -> dict | None:
    try:
        header, body, signature = token.split(".")
    except ValueError:
        return None
    expected = hmac.new(JWT_SECRET.encode(), f"{header}.{body}".encode(), hashlib.sha256).digest()
    if not hmac.compare_digest(b64(expected), signature):
        return None
    claims = json.loads(base64.urlsafe_b64decode(body + "=" * (-len(body) % 4)))
    return claims if claims.get("exp", 0) > time.time() else None


ANON_KEY = mint({"role": "anon", "exp": int(time.time()) + 7 * 86400})


def session_token(user: UUID) -> str:
    return mint(
        {"sub": str(user), "role": "authenticated", "aud": "authenticated", "exp": int(time.time()) + 86400}
    )


# PostgreSQL -------------------------------------------------------------------------------


def dsn(database: str, *, user: str | None = None, password: str | None = None) -> str | None:
    raw = os.environ.get("JOURNALPULSE_PG_DSN")
    if not raw:
        if user is None:
            return None
        return f"postgresql://{user}:{password}@127.0.0.1:5432/{database}"
    parts = urlsplit(raw)
    netloc = parts.netloc
    if user is not None:
        host = netloc.rsplit("@", 1)[-1]
        netloc = f"{user}:{password}@{host}"
    return urlunsplit(parts._replace(path=f"/{database}", netloc=netloc))


def psql(database: str, sql: str) -> str:
    target = dsn(database)
    command = (
        ["psql", target, "-v", "ON_ERROR_STOP=1", "-X", "-q", "-t", "-A"]
        if target
        else [
            "sudo",
            "-u",
            "postgres",
            "psql",
            "-d",
            database,
            "-v",
            "ON_ERROR_STOP=1",
            "-X",
            "-q",
            "-t",
            "-A",
        ]
    )
    completed = subprocess.run(command, input=sql, text=True, capture_output=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr or completed.stdout)
    return completed.stdout.strip()


def prepare_database() -> None:
    psql(
        "postgres",
        f"""
        select pg_terminate_backend(pid) from pg_stat_activity
        where datname = '{DATABASE}' and pid <> pg_backend_pid();
        drop database if exists {DATABASE};
        create database {DATABASE};
        """,
    )
    psql(DATABASE, (ROOT / "scripts" / "pg_harness.sql").read_text())
    for path in sorted((ROOT / "supabase" / "migrations").glob("*.sql")):
        psql(DATABASE, path.read_text())
    users = ", ".join(f"('{user_id}', '{name}@example.test')" for name, user_id in USERS.items())
    psql(
        DATABASE,
        f"""
        do $$
        begin
          if not exists (select 1 from pg_roles where rolname = 'authenticator') then
            create role authenticator login noinherit;
          end if;
        end $$;
        alter role authenticator with login password '{AUTHENTICATOR_PASSWORD}';
        grant anon, authenticated to authenticator;
        insert into auth.users (id, email) values {users};
        insert into private.server_secrets (name, value) values ('write_signing_key', '{SIGNING_KEY}');
        """,
    )


# PostgREST --------------------------------------------------------------------------------


def postgrest_binary() -> Path:
    found = shutil.which("postgrest")
    if found:
        return Path(found)
    cache = Path.home() / ".cache" / "journalpulse" / POSTGREST_VERSION
    binary = cache / "postgrest"
    if binary.exists():
        return binary
    cache.mkdir(parents=True, exist_ok=True)
    archive = cache / "postgrest.tar.xz"
    url = (
        f"https://github.com/PostgREST/postgrest/releases/download/{POSTGREST_VERSION}/"
        f"postgrest-{POSTGREST_VERSION}-linux-static-x86-64.tar.xz"
    )
    urllib.request.urlretrieve(url, archive)
    with tarfile.open(archive, "r:xz") as bundle:
        bundle.extract("postgrest", cache, filter="data")
    binary.chmod(0o755)
    return binary


def start_postgrest() -> subprocess.Popen:
    environment = {
        **os.environ,
        "PGRST_DB_URI": dsn(DATABASE, user="authenticator", password=AUTHENTICATOR_PASSWORD) or "",
        "PGRST_DB_SCHEMAS": "public",
        "PGRST_DB_ANON_ROLE": "anon",
        "PGRST_JWT_SECRET": JWT_SECRET,
        "PGRST_SERVER_PORT": str(POSTGREST_PORT),
        "PGRST_SERVER_HOST": "127.0.0.1",
    }
    process = subprocess.Popen([str(postgrest_binary())], env=environment)
    atexit.register(process.terminate)
    for _ in range(100):
        try:
            if httpx.get(f"http://127.0.0.1:{POSTGREST_PORT}/", timeout=1).status_code < 500:
                return process
        except httpx.HTTPError:
            pass
        time.sleep(0.2)
    raise RuntimeError("PostgREST did not start")


# Supabase-shaped gateway --------------------------------------------------------------------


class Gateway(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002 - stdlib signature
        return

    def _send(self, status: int, body: bytes, content_type: str = "application/json", extra=()) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        for key, value in extra:
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: int, payload: object) -> None:
        self._send(status, json.dumps(payload).encode())

    def do_OPTIONS(self) -> None:  # noqa: N802 - stdlib naming
        self._send(
            204, b"", extra=[("Access-Control-Allow-Headers", "*"), ("Access-Control-Allow-Methods", "*")]
        )

    def _proxy(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else None
        path = self.path.removeprefix("/rest/v1")
        headers = {
            key: value
            for key, value in self.headers.items()
            if key.lower() in {"authorization", "content-type", "prefer", "accept", "range"}
        }
        response = httpx.request(
            self.command,
            f"http://127.0.0.1:{POSTGREST_PORT}{path}",
            headers=headers,
            content=body,
            timeout=15,
        )
        passthrough = [
            (key, value) for key, value in response.headers.items() if key.lower() == "content-range"
        ]
        self._send(
            response.status_code,
            response.content,
            response.headers.get("content-type", "application/json"),
            passthrough,
        )

    def _route(self) -> None:
        if self.path.startswith("/rest/v1"):
            return self._proxy()
        if self.path == "/auth/v1/user" and self.command == "GET":
            token = (self.headers.get("Authorization") or "").removeprefix("Bearer ").strip()
            claims = verify(token)
            if not claims or claims.get("role") != "authenticated":
                return self._json(401, {"message": "invalid token"})
            return self._json(200, {"id": claims["sub"], "aud": "authenticated"})
        if self.path.startswith("/test/session/"):
            name = self.path.rsplit("/", 1)[-1]
            if name not in USERS:
                return self._json(404, {"message": "unknown user"})
            return self._json(200, {"user_id": str(USERS[name]), "access_token": session_token(USERS[name])})
        if self.path.startswith("/test/age/") and self.command == "POST":
            conversation_id = str(UUID(self.path.rsplit("/", 1)[-1]))
            psql(
                DATABASE,
                f"update public.conversations set updated_at = now() - interval '25 hours' "
                f"where id = '{conversation_id}';",
            )
            return self._json(200, {"aged": conversation_id})
        if self.path == "/test/purge" and self.command == "POST":
            return self._json(
                200, json.loads(psql(DATABASE, "select public.jp_purge_expired_conversations();"))
            )
        return self._json(404, {"message": "not found"})

    do_GET = do_POST = do_PATCH = do_DELETE = _route  # noqa: N815


def start_gateway() -> ThreadingHTTPServer:
    server = ThreadingHTTPServer(("127.0.0.1", GATEWAY_PORT), Gateway)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


# Web and API ---------------------------------------------------------------------------------


def build_web() -> None:
    if os.getenv("JP_SKIP_WEB_BUILD") == "1" and (WEB_DIST / "index.html").exists():
        return
    environment = {
        **os.environ,
        "JOURNALPULSE_STATIC_EXPORT": "true",
        "NEXT_PUBLIC_SUPABASE_URL": GATEWAY_URL,
        "NEXT_PUBLIC_SUPABASE_ANON_KEY": ANON_KEY,
        "NEXT_PUBLIC_API_BASE_URL": "",
    }
    subprocess.run(["npm", "run", "build"], cwd=ROOT / "web", env=environment, check=True)


class DeterministicLuna:
    """Stands in for OpenRouter: fixed replies, keyword feelings, and an offer on turn two."""

    def complete(self, messages: list[dict[str, str]]) -> ConversationCompletion:
        user_texts = [message["content"] for message in messages if message["role"] == "user"]
        turn = len(user_texts)
        return ConversationCompletion(
            reply=(
                "Thank you for telling me. What feels heaviest right now?"
                if turn < 2
                else "That sounds like a lot. Would you like to find one small thing together?"
            ),
            offer_action=turn >= 2,
            resource_intent="ground",
            card_reason="A reviewed option." if turn >= 2 else "",
            summary="A check-in during integration testing.",
            feelings=tuple(guess_feelings(user_texts)),
            model_run=ModelRun(
                model="integration-fake-luna", provider="fake", latency_ms=1, schema_valid=True
            ),
        )


def api_settings() -> Settings:
    return Settings(
        environment="integration",
        database_path=ROOT / "artifacts" / "integration-unused.db",
        resource_catalog_path=ROOT / "assets" / "resources" / "catalog.json",
        openrouter_api_key="integration-fake-key",
        openrouter_model="integration-fake-luna",
        openrouter_base_url="http://127.0.0.1:9/unused",
        openrouter_zdr=True,
        openrouter_timeout_seconds=1,
        supabase_url=GATEWAY_URL,
        supabase_anon_key=ANON_KEY,
        raw_text_retention_default=False,
        cors_origins=(f"http://127.0.0.1:{API_PORT}",),
        analysis_rate_limit_per_minute=int(os.getenv("JP_RATE_LIMIT", "12")),
        chat_model="integration-fake-luna",
        write_signing_key=SIGNING_KEY,
    )


def main() -> None:
    prepare_database()
    start_postgrest()
    start_gateway()
    build_web()
    os.environ["JOURNALPULSE_WEB_DIST"] = str(WEB_DIST)
    app = create_app(settings=api_settings(), conversation_client=DeterministicLuna())
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    print(f"integration stack ready: api http://127.0.0.1:{API_PORT}, gateway {GATEWAY_URL}", flush=True)
    uvicorn.run(app, host="127.0.0.1", port=API_PORT, log_level="warning")


if __name__ == "__main__":
    main()
