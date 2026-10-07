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
import platform
import re
import secrets
import shutil
import signal
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit
from uuid import UUID, uuid4

import httpx
import uvicorn

from scratch_postgres import postgres_uri, require_local_postgres_dsn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from journalpulse.api import create_app  # noqa: E402
from journalpulse.config import Settings  # noqa: E402
from journalpulse.discovery_models import (  # noqa: E402
    DiscoveryCandidate,
    DiscoveryProvenance,
    DiscoveryRequest,
    DiscoveryResponse,
)
from journalpulse.discovery_prompts import DISCOVERY_PROMPT_VERSION  # noqa: E402
from journalpulse.domain import (  # noqa: E402
    ActivityConstraintInputs,
    ModelRun,
)
from journalpulse.guided import guess_feelings  # noqa: E402
from journalpulse.guided_action import ActivityDirective, GuidedActionContext  # noqa: E402
from journalpulse.intelligence import ConversationCompletion  # noqa: E402

DATABASE = "jp_integration"
JWT_SECRET = "integration-only-jwt-secret-0123456789abcdef"
SIGNING_KEY = "integration-only-signing-key-0123456789abcdef"
# A run owns only this fresh role; never rotate a shared local Supabase login.
AUTHENTICATOR_ROLE = "jp_test_auth_" + uuid4().hex
AUTHENTICATOR_PASSWORD = secrets.token_urlsafe(32)
_authenticator_created = False
POSTGREST_VERSION = "v12.2.12"
# SHA-256 of the official GitHub release archives, recorded by the maintainers on 2026-10-05.
# PostgREST publishes no checksums for these assets, so this pins the reviewed bytes (trust on
# first use); it does not prove upstream authenticity. Add a platform only with its own hash.
POSTGREST_ASSETS: dict[tuple[str, str], tuple[str, str]] = {
    ("Linux", "x86_64"): (
        "linux-static-x86-64",
        "5de4092f1719da3353c40bf96c8dec6913f2254a7cd0b61cc05f233153b557d5",
    ),
    ("Darwin", "arm64"): (
        "macos-aarch64",
        "66eb150109409caea1b90c819418d326b2a5fd59789209e0dd45c87078c71909",
    ),
}
POSTGREST_DOWNLOAD_TIMEOUT_SECONDS = 60
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
    "erin": UUID("e5e5e5e5-0000-4000-8000-000000000005"),
    "frank": UUID("f6f6f6f6-0000-4000-8000-000000000006"),
    "grace": UUID("a7a7a7a7-0000-4000-8000-000000000007"),
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
    require_local_postgres_dsn(raw)
    if not raw:
        if user is None:
            return None
        return f"postgresql://{user}:{password}@127.0.0.1:5432/{database}"
    parts = urlsplit(raw)
    netloc = parts.netloc
    if user is not None:
        host = netloc.rsplit("@", 1)[-1]
        netloc = f"{user}:{password}@{host}"
        # Admin URI options must not override the isolated PostgREST role.
        parts = parts._replace(query=urlencode([
            (key, value) for key, value in parse_qsl(parts.query, keep_blank_values=True)
            if key not in {"user", "password", "passfile"}
        ]))
    return postgres_uri(parts._replace(path=f"/{database}", netloc=netloc))


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
    global _authenticator_created
    require_local_postgres_dsn(os.getenv("JOURNALPULSE_PG_DSN"))
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
        begin;
        create role "{AUTHENTICATOR_ROLE}" login noinherit password '{AUTHENTICATOR_PASSWORD}';
        grant anon, authenticated to "{AUTHENTICATOR_ROLE}";
        insert into auth.users (id, email) values {users};
        insert into private.server_secrets (name, value) values ('write_signing_key', '{SIGNING_KEY}');
        commit;
        """,
    )
    _authenticator_created = True


def cleanup_authenticator() -> None:
    """Best-effort removal of the role this run successfully created."""
    if not _authenticator_created:
        return
    try:
        psql("postgres", f'drop role if exists "{AUTHENTICATOR_ROLE}";')
    except RuntimeError:
        # Do not echo a connection string or SQL containing the random password.
        print(
            f"Remove the leftover test role {AUTHENTICATOR_ROLE} from the disposable cluster.",
            file=sys.stderr,
        )


def stop_postgrest(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


# PostgREST --------------------------------------------------------------------------------


def postgrest_version_matches(binary: Path) -> bool:
    """A binary on PATH or in the cache is used only if it reports the pinned release."""
    try:
        result = subprocess.run(
            [str(binary), "--version"], capture_output=True, text=True, timeout=10, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return False
    expected = re.escape(POSTGREST_VERSION.removeprefix("v"))
    return re.match(rf"PostgREST {expected}\b", result.stdout.strip()) is not None


def postgrest_binary() -> Path:
    found = shutil.which("postgrest")
    if found and postgrest_version_matches(Path(found)):
        return Path(found)
    if found:
        print(f"Ignoring {found}: it is not PostgREST {POSTGREST_VERSION}.", file=sys.stderr)
    asset = POSTGREST_ASSETS.get((platform.system(), platform.machine()))
    if asset is None:
        raise RuntimeError(
            f"No pinned PostgREST download for {platform.system()} {platform.machine()}; "
            f"put PostgREST {POSTGREST_VERSION.removeprefix('v')} on PATH instead."
        )
    name, expected_sha256 = asset
    cache = Path.home() / ".cache" / "journalpulse" / POSTGREST_VERSION
    binary = cache / "postgrest"
    if binary.exists() and postgrest_version_matches(binary):
        return binary
    cache.mkdir(parents=True, exist_ok=True)
    url = (
        f"https://github.com/PostgREST/postgrest/releases/download/{POSTGREST_VERSION}/"
        f"postgrest-{POSTGREST_VERSION}-{name}.tar.xz"
    )
    # Verify the archive before extracting anything, and only rename a finished binary into
    # place so an interrupted or tampered download never becomes the cached executable.
    with tempfile.TemporaryDirectory(dir=cache) as staging:
        archive = Path(staging) / "postgrest.tar.xz"
        digest = hashlib.sha256()
        with urllib.request.urlopen(url, timeout=POSTGREST_DOWNLOAD_TIMEOUT_SECONDS) as response:
            with archive.open("wb") as output:
                while chunk := response.read(1 << 16):
                    digest.update(chunk)
                    output.write(chunk)
        if not hmac.compare_digest(digest.hexdigest(), expected_sha256):
            raise RuntimeError(f"PostgREST download {name} failed its pinned SHA-256 check.")
        with tarfile.open(archive, "r:xz") as bundle:
            bundle.extract("postgrest", staging, filter="data")
        extracted = Path(staging) / "postgrest"
        extracted.chmod(0o755)
        if not postgrest_version_matches(extracted):
            raise RuntimeError(f"The verified archive did not contain PostgREST {POSTGREST_VERSION}.")
        os.replace(extracted, binary)
    return binary


def start_postgrest() -> subprocess.Popen:
    environment = {
        **os.environ,
        "PGRST_DB_URI": dsn(DATABASE, user=AUTHENTICATOR_ROLE, password=AUTHENTICATOR_PASSWORD) or "",
        "PGRST_DB_SCHEMAS": "public",
        "PGRST_DB_ANON_ROLE": "anon",
        "PGRST_JWT_SECRET": JWT_SECRET,
        "PGRST_SERVER_PORT": str(POSTGREST_PORT),
        "PGRST_SERVER_HOST": "127.0.0.1",
    }
    process = subprocess.Popen([str(postgrest_binary())], env=environment)
    atexit.register(stop_postgrest, process)
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
    """A named fake for legacy provider flags and the new quiet-activity contract."""

    def complete(self, messages: list[dict[str, str]]) -> ConversationCompletion:
        user_texts = [message["content"] for message in messages if message["role"] == "user"]
        # A selected entry is a separate untrusted background message, not a turn
        # the person typed. Counting it would offer action one message too early.
        turn = sum(not text.startswith("Selected journal entry ") for text in user_texts)
        listening = any(
            message["role"] == "system" and "Just talk" in message["content"] for message in messages
        )
        return ConversationCompletion(
            reply=(
                "I’m listening. What else is on your mind?"
                if listening
                else "Thank you for telling me. What feels heaviest right now?"
                if turn < 2
                else "That sounds like a lot. Would you like to find one small thing together?"
            ),
            # Keep an adversarial offer flag while listening: server state must win.
            offer_action=turn >= 2,
            resource_intent="ground",
            card_reason="A reviewed option." if turn >= 2 else "",
            summary="A check-in during integration testing.",
            feelings=tuple(guess_feelings(user_texts)),
            model_run=ModelRun(
                model="integration-fake-luna", provider="fake", latency_ms=1, schema_valid=True
            ),
        )

    def complete_guided(
        self,
        messages: list[dict[str, str]],
        context: GuidedActionContext,
    ) -> ConversationCompletion:
        latest = next((item["content"] for item in reversed(messages) if item["role"] == "user"), "")
        # This stand-in proves the real session path, never model quality. Other
        # provider scenarios retain the old six-field schema to exercise backwards
        # compatibility and adversarial readiness flags. The manual UI loop itself
        # is tested explicitly in no-AI guided mode, rather than faking an AI flow.
        if "participant_report" in latest:
            return ConversationCompletion(
                reply="You reported what happened. We can leave it here or keep talking.",
                offer_action=False,
                resource_intent="ground",
                card_reason="",
                summary="A saved fictional activity check-in.",
                model_run=ModelRun(
                    model="integration-fake-luna", provider="fake", latency_ms=1, schema_valid=True
                ),
                activity=ActivityDirective(
                    move="outcome",
                    goal=None,
                    selected_resource_id=None,
                    constraints=context.constraints,
                    search_topic=None,
                ),
            )
        if "two minutes" in latest.lower() and "quiet" in latest.lower() and context.action_allowed:
            return ConversationCompletion(
                reply=("A short quiet pause could fit the two minutes you have. "
                       "You can try it or keep talking."),
                offer_action=True,
                resource_intent="ground",
                card_reason="A silent seated option for two minutes.",
                summary="A fictional request for a quiet pause.",
                model_run=ModelRun(
                    model="integration-fake-luna", provider="fake", latency_ms=1, schema_valid=True
                ),
                activity=ActivityDirective(
                    move="propose",
                    goal="settle",
                    selected_resource_id="guided_meditation_2m",
                    constraints=ActivityConstraintInputs(time_minutes=2, no_audio=True, seated=True),
                    search_topic=None,
                ),
            )
        return self.complete(messages)


class DeterministicDiscovery:
    """Local test double; no Brave queries or paid model calls are made."""

    def search(self, payload: DiscoveryRequest) -> DiscoveryResponse:
        candidates = [
            DiscoveryCandidate(
                title=f"Fictional reflection resource {index}",
                url=f"https://example.org/reflection-{index}",
                description="A synthetic search snippet for the browser integration fixture.",
                why_selected="This fixture snippet addresses the approved reflection topic.",
            )
            for index in range(1, 7)
            if f"https://example.org/reflection-{index}" not in payload.excluded_urls
        ][:2]
        return DiscoveryResponse(
            original_query=payload.original_query,
            updated_query=f"{payload.original_query} {payload.feedback or ''}".strip(),
            candidates=candidates,
            provenance=DiscoveryProvenance(
                prompt_version=DISCOVERY_PROMPT_VERSION,
                retrieved_at="2026-10-04T12:00:00+00:00",
                candidate_count=len(candidates),
                model_runs=[
                    ModelRun(
                        model="integration-fake-discovery",
                        provider="test-double",
                        latency_ms=0,
                        schema_valid=True,
                        prompt_version=DISCOVERY_PROMPT_VERSION,
                    )
                ],
            ),
            limitations=["Synthetic integration fixtures; no live pages or search providers were consulted."],
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
        search_feature_enabled=True,
        search_api_key="integration-fake-search-key",
    )


def main() -> None:
    # Registered first, so PostgREST's later cleanup closes its sessions first.
    atexit.register(cleanup_authenticator)
    prepare_database()
    start_postgrest()
    start_gateway()
    build_web()
    os.environ["JOURNALPULSE_WEB_DIST"] = str(WEB_DIST)
    app = create_app(
        settings=api_settings(),
        conversation_client=DeterministicLuna(),
        discovery_client=DeterministicDiscovery(),
    )
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    print(f"integration stack ready: api http://127.0.0.1:{API_PORT}, gateway {GATEWAY_URL}", flush=True)
    uvicorn.run(app, host="127.0.0.1", port=API_PORT, log_level="warning")


if __name__ == "__main__":
    main()
