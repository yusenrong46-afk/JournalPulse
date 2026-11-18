# ruff: noqa: E501
"""Stand up a complete local JournalPulse environment, check it, and report how long it took.

    make onboard NAME=<env>                       # uv run python scripts/onboard.py <env>
    uv run python scripts/onboard.py <env> --runtime compose|host|auto [--keep]

One environment is PostgreSQL with every migration in ``supabase/migrations`` applied in
order, PostgREST 12, a Supabase-shaped auth stand-in (anonymous demo sign-ins only,
``scripts/onboarding_auth.py``), and the API serving the exported web app. AI and search are
off and every port binds 127.0.0.1. Steps:

1. database: a fresh database with the real migrations, its PostgREST login and signing key;
2. the PostgreSQL verifiers (``make test-postgres``'s list) against the same cluster;
3. images or web export, then the services;
4. ``/ready`` must return 200, and ``/health`` must name this commit;
5. the demo journey (``web/tests/demo/demo-journey.spec.ts``) against the environment;
6. ``outputs/onboarding/<env>/REPORT.md`` with the time of every step; then teardown.

Runtimes. ``compose`` runs everything in Docker Compose (``deploy/onboarding/compose.yaml``).
``host`` runs the same processes on this machine against the local cluster in
``JOURNALPULSE_PG_DSN`` (restricted to localhost). ``auto`` (the default) tries Compose and
falls back to ``host`` when Docker or its images are unavailable; the report says which ran
and why. No secret is printed or written to the report.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import secrets
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from onboarding_auth import anon_key  # noqa: E402
from scratch_postgres import postgres_uri, require_local_postgres_dsn  # noqa: E402

COMPOSE_FILE = ROOT / "deploy" / "onboarding" / "compose.yaml"
VERIFIERS = (  # the Makefile's PG_VERIFIERS, in the same order
    "verify_postgres_schema", "verify_journal_schema", "verify_integrity_upgrade",
    "verify_activity_schema", "verify_sentinel_repairs", "verify_erasure_schema",
    "verify_demo_sandbox",
)
NAME = re.compile(r"^[a-z][a-z0-9-]{0,30}$")
READY_TIMEOUT_SECONDS = 180


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def git_commit() -> str:
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else ""


def dirty_tree() -> bool:
    result = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT,
                            capture_output=True, text=True, check=False)
    return bool(result.stdout.strip())


def psql(dsn: str, sql: str) -> str:
    completed = subprocess.run(
        ["psql", dsn, "-v", "ON_ERROR_STOP=1", "-X", "-q", "-t", "-A"],
        input=sql, text=True, capture_output=True, check=False,
    )
    if completed.returncode != 0:
        # The URI may hold a password; psql's own message names the failing statement.
        raise RuntimeError((completed.stderr or completed.stdout).strip().splitlines()[-1])
    return completed.stdout.strip()


def with_database(dsn: str, database: str, *, user: str | None = None, password: str | None = None) -> str:
    parts = urlsplit(dsn)
    netloc = parts.netloc
    if user is not None:
        netloc = f"{user}:{password}@{netloc.rsplit('@', 1)[-1]}"
    return postgres_uri(parts._replace(path=f"/{database}", netloc=netloc))


def http_get(url: str, timeout: float = 10) -> tuple[int, dict[str, str], str]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:  # noqa: S310 - loopback only
            return response.status, dict(response.headers), response.read().decode()
    except urllib.error.HTTPError as error:
        return error.code, dict(error.headers), error.read().decode()


@dataclass
class Step:
    name: str
    seconds: float
    result: str


@dataclass
class Environment:
    name: str
    runtime: str
    runtime_note: str = ""
    commit: str = field(default_factory=git_commit)
    api_port: int = field(default_factory=free_port)
    auth_port: int = field(default_factory=free_port)
    postgrest_port: int = field(default_factory=free_port)
    db_port: int = field(default_factory=free_port)
    jwt_secret: str = field(default_factory=lambda: secrets.token_urlsafe(48))
    signing_key: str = field(default_factory=lambda: secrets.token_urlsafe(48))
    authenticator_password: str = field(default_factory=lambda: secrets.token_urlsafe(24))
    db_password: str = field(default_factory=lambda: secrets.token_urlsafe(24))
    dirty: bool = field(default_factory=lambda: dirty_tree())  # before any build touches files
    steps: list[Step] = field(default_factory=list)
    facts: dict[str, object] = field(default_factory=dict)
    processes: list[subprocess.Popen] = field(default_factory=list)

    # Names ---------------------------------------------------------------------------------
    @property
    def slug(self) -> str:
        return self.name.replace("-", "_")

    @property
    def database(self) -> str:
        return f"jp_onboard_{self.slug}"

    @property
    def authenticator(self) -> str:
        return f"jp_onboard_auth_{self.slug}"

    @property
    def project(self) -> str:
        return f"jp-onboard-{self.name}"

    @property
    def output(self) -> Path:
        return ROOT / "outputs" / "onboarding" / self.name

    @property
    def api_url(self) -> str:
        return f"http://127.0.0.1:{self.api_port}"

    @property
    def anon(self) -> str:
        return anon_key(self.jwt_secret)

    def admin_dsn(self) -> str:
        if self.runtime == "compose":
            return f"postgresql://postgres:{self.db_password}@127.0.0.1:{self.db_port}/postgres"
        dsn = os.environ.get("JOURNALPULSE_PG_DSN", "")
        if not dsn:
            raise SystemExit("The host runtime needs JOURNALPULSE_PG_DSN (a local PostgreSQL 16 cluster).")
        require_local_postgres_dsn(dsn)
        return dsn

    def compose_environment(self) -> dict[str, str]:
        return {
            **os.environ,
            "COMPOSE_PROJECT_NAME": self.project,
            "JP_DB_PASSWORD": self.db_password, "JP_DB_PORT": str(self.db_port),
            "JP_DATABASE": self.database, "JP_AUTHENTICATOR_ROLE": self.authenticator,
            "JP_AUTHENTICATOR_PASSWORD": self.authenticator_password, "JP_JWT_SECRET": self.jwt_secret,
            "JP_AUTH_PORT": str(self.auth_port), "JP_API_PORT": str(self.api_port),
            "JP_ANON_KEY": self.anon, "JP_SIGNING_KEY": self.signing_key, "JP_GIT_COMMIT": self.commit,
        }

    def compose(self, *arguments: str, check: bool = True, quiet: bool = False) -> subprocess.CompletedProcess:
        command = ["docker", "compose", "-f", str(COMPOSE_FILE), *arguments]
        return subprocess.run(
            command, env=self.compose_environment(), check=check, text=True,
            capture_output=quiet,
        )

    # Steps ---------------------------------------------------------------------------------
    def step(self, name: str, action) -> object:
        print(f"== {name}", flush=True)
        started = time.perf_counter()
        try:
            outcome = action()
        except BaseException as error:
            self.steps.append(Step(name, time.perf_counter() - started, f"FAILED: {error}"))
            raise
        self.steps.append(Step(name, time.perf_counter() - started, str(outcome or "ok")))
        return outcome

    def start_database(self) -> str:
        if self.runtime != "compose":
            return "local cluster from JOURNALPULSE_PG_DSN"
        self.compose("up", "-d", "--wait", "db")
        return "pgvector/pgvector:pg16 container healthy"

    def prepare_database(self) -> str:
        admin = self.admin_dsn()
        psql(admin, f"""
            select pg_terminate_backend(pid) from pg_stat_activity
            where datname = '{self.database}' and pid <> pg_backend_pid();
            drop database if exists {self.database};
            create database {self.database};
        """)
        target = with_database(admin, self.database)
        psql(target, (ROOT / "scripts" / "pg_harness.sql").read_text())
        migrations = sorted((ROOT / "supabase" / "migrations").glob("*.sql"))
        for path in migrations:
            psql(target, path.read_text())
        psql(admin, f"""
            drop role if exists "{self.authenticator}";
            create role "{self.authenticator}" login noinherit password '{self.authenticator_password}';
            grant anon, authenticated to "{self.authenticator}";
        """)
        psql(target, f"insert into private.server_secrets (name, value) values "
                     f"('write_signing_key', '{self.signing_key}');")
        self.facts["migrations"] = len(migrations)
        return f"{len(migrations)} migrations applied to {self.database}"

    def run_verifiers(self) -> str:
        results = {}
        for script in VERIFIERS:
            started = time.perf_counter()
            completed = subprocess.run(
                [sys.executable, str(ROOT / "scripts" / f"{script}.py")],
                env={**os.environ, "JOURNALPULSE_PG_DSN": self.admin_dsn()},
                cwd=ROOT, capture_output=True, text=True, check=False,
            )
            last = (completed.stdout.strip().splitlines() or [""])[-1]
            seconds = round(time.perf_counter() - started, 1)
            results[script] = {"passed": completed.returncode == 0, "seconds": seconds, "last_line": last}
            print(f"   {script}: {'passed' if completed.returncode == 0 else 'FAILED'} ({seconds} s)", flush=True)
            if completed.returncode != 0:
                self.facts["verifiers"] = results
                tail = (completed.stderr.strip().splitlines() or ["no error output"])[-1]
                raise RuntimeError(f"{script} failed: {tail}")
        self.facts["verifiers"] = results
        return f"{len(VERIFIERS)} of {len(VERIFIERS)} verifiers passed"

    def build(self) -> str:
        if self.runtime == "compose":
            self.compose("--profile", "app", "build")
            return "images built (Dockerfile.api with the exported web, auth stand-in)"
        environment = {
            **os.environ,
            "JOURNALPULSE_STATIC_EXPORT": "true",
            "NEXT_PUBLIC_SUPABASE_URL": f"http://127.0.0.1:{self.auth_port}",
            "NEXT_PUBLIC_SUPABASE_ANON_KEY": self.anon,
            "NEXT_PUBLIC_API_BASE_URL": "",
            "NEXT_PUBLIC_JOURNALPULSE_DEMO_ENABLED": "true",
        }
        command = ["npm", "--prefix", str(ROOT / "web"), "run", "build"]
        if subprocess.run(command, env=environment, check=False).returncode != 0:
            # Next.js 16.4 stops the first build in an agent session to show an upgrade
            # notice; a real build error fails twice.
            print("web build stopped; retrying once", flush=True)
            subprocess.run(command, env=environment, check=True)
        web = ROOT / ".onboarding" / self.name / "web"
        if web.exists():
            shutil.rmtree(web)
        shutil.copytree(ROOT / "web" / "out", web)
        self.facts["web_dist"] = web
        return "web exported"

    def start_services(self) -> str:
        if self.runtime == "compose":
            self.compose("--profile", "app", "up", "-d")
            return "postgrest, auth and api containers started"
        from integration_stack import postgrest_binary  # pinned PostgREST 12 (SHA-256 checked)

        admin = self.admin_dsn()
        common = {key: value for key, value in os.environ.items() if not key.startswith("PGRST_")}
        self.processes.append(subprocess.Popen([str(postgrest_binary())], stdout=subprocess.DEVNULL, env={
            **common,
            "PGRST_DB_URI": with_database(
                admin, self.database, user=self.authenticator, password=self.authenticator_password
            ),
            "PGRST_DB_SCHEMAS": "public", "PGRST_DB_ANON_ROLE": "anon",
            "PGRST_JWT_SECRET": self.jwt_secret, "PGRST_SERVER_HOST": "127.0.0.1",
            "PGRST_SERVER_PORT": str(self.postgrest_port), "PGRST_LOG_LEVEL": "error",
        }))
        self.processes.append(subprocess.Popen([sys.executable, str(ROOT / "scripts" / "onboarding_auth.py")], env={
            **os.environ,
            "JP_AUTH_JWT_SECRET": self.jwt_secret,
            "JP_AUTH_DATABASE_URL": with_database(admin, self.database),
            "JP_POSTGREST_URL": f"http://127.0.0.1:{self.postgrest_port}",
            "JP_AUTH_PORT": str(self.auth_port),
        }))
        sqlite = ROOT / ".onboarding" / self.name / "unused.db"
        self.processes.append(subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "journalpulse.api:app", "--host", "127.0.0.1",
             "--port", str(self.api_port), "--log-level", "warning"],
            cwd=ROOT,
            env={
                **os.environ,
                "PYTHONPATH": str(ROOT / "src"),
                "JOURNALPULSE_ENV": "onboarding",
                "JOURNALPULSE_DB_PATH": str(sqlite),
                "JOURNALPULSE_WEB_DIST": str(self.facts["web_dist"]),
                "SUPABASE_URL": f"http://127.0.0.1:{self.auth_port}",
                "SUPABASE_ANON_KEY": self.anon,
                "JOURNALPULSE_WRITE_SIGNING_KEY": self.signing_key,
                "JOURNALPULSE_DEMO_ENABLED": "true",
                "JOURNALPULSE_CORS_ORIGINS": self.api_url,
                "JOURNALPULSE_GIT_COMMIT": self.commit,
                # Blank values win over a local .env, so no provider can be configured.
                "JOURNALPULSE_LLM_ENABLED": "false", "JOURNALPULSE_SEARCH_ENABLED": "false",
                "JOURNALPULSE_LLM_API_KEY": "", "OPENROUTER_API_KEY": "", "JOURNALPULSE_SEARCH_API_KEY": "",
            },
        ))
        return "postgrest, auth stand-in and api processes started"

    def wait_ready(self) -> str:
        deadline = time.monotonic() + READY_TIMEOUT_SECONDS
        status, body = 0, ""
        while time.monotonic() < deadline:
            for process in self.processes:
                if process.poll() is not None:
                    raise RuntimeError(f"a service exited with code {process.returncode}")
            try:
                status, _, body = http_get(f"{self.api_url}/ready", timeout=5)
            except (urllib.error.URLError, ConnectionError, TimeoutError):
                status = 0
            if status == 200:
                self.facts["ready"] = json.loads(body)
                return "200 " + json.dumps(json.loads(body)["checks"], sort_keys=True)
            time.sleep(1)
        raise RuntimeError(f"/ready did not return 200 within {READY_TIMEOUT_SECONDS} s (last {status}: {body[:200]})")

    def check_identity(self) -> str:
        status, headers, _ = http_get(f"{self.api_url}/health")
        served = headers.get("X-JournalPulse-Commit") or headers.get("x-journalpulse-commit")
        if status != 200 or not self.commit or served != self.commit:
            raise RuntimeError(f"/health {status} named commit {served!r}, expected {self.commit!r}")
        self.facts["served_commit"] = served
        return f"/health 200, X-JournalPulse-Commit {served[:12]}"

    def demo_journey(self) -> str:
        pictures = self.output / "journey"
        pictures.mkdir(parents=True, exist_ok=True)
        environment = {
            **os.environ,
            "JP_DEMO_BASE_URL": self.api_url,
            "JP_DEMO_EXTRA_HOSTS": f"127.0.0.1:{self.auth_port}",
            "JP_DEMO_OUTPUT_DIR": str(pictures),
        }
        result = subprocess.run(
            ["npm", "exec", "--", "playwright", "test", "-c", "playwright.demo.config.ts"],
            cwd=ROOT / "web", env=environment, check=False,
        )
        if result.returncode != 0:
            self.journey_diagnostics()
            raise subprocess.CalledProcessError(result.returncode, result.args)
        return "passed: Try the demo, journal save, Simple Luna activity, check-in, Journey; no outside request"

    def journey_diagnostics(self) -> None:
        """After a failed journey, print what the page showed and the service logs (fictional data only)."""
        for context in sorted((ROOT / "web" / "test-results-demo").glob("*/error-context.md")):
            print(f"== {context.relative_to(ROOT)}", flush=True)
            print(context.read_text(errors="replace")[:6000], flush=True)
        if self.runtime == "compose":
            print("== service logs (api, auth)", flush=True)
            self.compose("logs", "--no-color", "--tail", "120", "api", "auth", check=False)

    def stop(self) -> str:
        for process in self.processes:
            if process.poll() is None:
                process.terminate()
        for process in self.processes:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
        self.processes.clear()
        if self.runtime == "compose":
            self.compose("--profile", "app", "down", "-v", "--remove-orphans", check=False)
            return "containers and volume removed"
        admin = self.admin_dsn()
        psql(admin, f"""
            select pg_terminate_backend(pid) from pg_stat_activity
            where datname = '{self.database}' and pid <> pg_backend_pid();
            drop database if exists {self.database};
            drop role if exists "{self.authenticator}";
        """)
        return f"processes stopped; {self.database} and its login dropped"

    # Report --------------------------------------------------------------------------------
    def write_report(self, outcome: str) -> Path:
        self.output.mkdir(parents=True, exist_ok=True)
        total = sum(step.seconds for step in self.steps)
        lines = [
            f"# Onboarding report: `{self.name}`",
            "",
            "Generated by `scripts/onboard.py` (`make onboard NAME=" + self.name + "`). Do not edit by hand.",
            "",
            "| | |",
            "|---|---|",
            f"| Result | **{outcome}** |",
            f"| Date (UTC) | {datetime.now(UTC).strftime('%Y-%m-%d %H:%M')} |",
            f"| Commit | `{self.commit or 'unknown'}`{' (working tree had uncommitted changes)' if self.dirty else ''} |",
            f"| Runtime | {self.runtime}{(' (' + self.runtime_note + ')') if self.runtime_note else ''} |",
            f"| Total time | {total:.1f} s |",
            "| AI and search | off (`JOURNALPULSE_LLM_ENABLED=false`, no provider key in the API's environment) |",
            "",
            "## Steps",
            "",
            "| Step | Seconds | Result |",
            "|---|---:|---|",
        ]
        lines += [f"| {s.name} | {s.seconds:.1f} | {s.result.replace('|', '/')} |" for s in self.steps]
        verifiers = self.facts.get("verifiers")
        if isinstance(verifiers, dict):
            lines += ["", "## PostgreSQL verifiers", "", "| Script | Seconds | Result |", "|---|---:|---|"]
            lines += [
                f"| `{name}` | {item['seconds']} | {'passed' if item['passed'] else 'FAILED'} |"
                for name, item in verifiers.items()
            ]
        ready = self.facts.get("ready")
        if isinstance(ready, dict):
            lines += ["", "## `/ready` checks", "", "| Check | Value |", "|---|---|"]
            lines += [f"| {key} | `{value}` |" for key, value in sorted(ready["checks"].items())]
        lines += [
            "",
            "Ports were chosen at random on 127.0.0.1 and the environment was torn down after the run "
            "(unless `--keep`). No secret appears in this report.",
            "",
        ]
        path = self.output / "REPORT.md"
        path.write_text("\n".join(lines))
        return path


def base_images() -> list[str]:
    """External images named in FROM lines of the Dockerfiles that Compose builds."""
    images: list[str] = []
    for dockerfile in (ROOT / "Dockerfile.api", ROOT / "deploy" / "onboarding" / "Dockerfile.auth"):
        stages: set[str] = set()
        for line in dockerfile.read_text().splitlines():
            parts = line.split()
            if len(parts) >= 2 and parts[0].upper() == "FROM":
                if parts[1] not in stages and parts[1] not in images:
                    images.append(parts[1])
                if len(parts) >= 4 and parts[2].upper() == "AS":
                    stages.add(parts[3])
    return images


def compose_available(environment: Environment) -> str | None:
    """None when Compose can run this environment; otherwise the reason, in one line."""
    if shutil.which("docker") is None:
        return "docker is not installed"
    try:
        environment.compose("version", quiet=True)
        environment.compose("--profile", "app", "pull", "--ignore-buildable", "db", "postgrest", quiet=True)
        # The images are built here too, so their base images must be reachable as well.
        for image in base_images():
            if subprocess.run(["docker", "image", "inspect", image], capture_output=True, check=False).returncode:
                subprocess.run(["docker", "pull", "-q", image], capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as error:
        detail = (error.stderr or error.stdout or "").strip().splitlines()
        reason = next((line for line in reversed(detail) if line.strip()), f"exit {error.returncode}")
        # Signed CDN URLs carry tokens: keep only the host.
        reason = re.sub(r"https?://([^/\s\"]+)\S*", r"https://\1/...", reason)
        if len(reason) > 200:  # keep the start (what failed) and the end (why: 429, Forbidden, ...)
            reason = f"{reason[:110]} ... {reason[-80:]}"
        return f"Docker Compose unavailable: {reason}"
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("name", help="environment name: lower-case letters, digits and hyphens")
    parser.add_argument("--runtime", choices=("auto", "compose", "host"), default="auto")
    parser.add_argument("--keep", action="store_true", help="leave it running until Ctrl+C")
    arguments = parser.parse_args()
    if not NAME.fullmatch(arguments.name):
        raise SystemExit("NAME must be lower-case letters, digits and hyphens, starting with a letter.")

    environment = Environment(arguments.name, "compose" if arguments.runtime != "host" else "host")
    if arguments.runtime == "auto":
        started = time.perf_counter()
        reason = compose_available(environment)
        if reason:
            environment.runtime, environment.runtime_note = "host", f"fell back: {reason}"
            print(f"Compose is not usable here ({reason}); using the host runtime.", flush=True)
        environment.steps.append(Step("choose runtime", time.perf_counter() - started, environment.runtime))

    outcome = "FAILED"
    try:
        environment.step("start database", environment.start_database)
        environment.step("apply migrations", environment.prepare_database)
        environment.step("PostgreSQL verifiers", environment.run_verifiers)
        environment.step("build", environment.build)
        environment.step("start services", environment.start_services)
        environment.step("/ready", environment.wait_ready)
        environment.step("build identity", environment.check_identity)
        environment.step("demo journey", environment.demo_journey)
        outcome = "PASS"
        if arguments.keep:
            print(f"{environment.name} is running at {environment.api_url} (Ctrl+C stops it)", flush=True)
            try:
                while all(process.poll() is None for process in environment.processes):
                    time.sleep(1)
            except KeyboardInterrupt:
                pass
    finally:
        try:
            environment.step("teardown", environment.stop)
        finally:
            report = environment.write_report(outcome)
            print(f"onboard {environment.name}: {outcome}; report {report.relative_to(ROOT)}", flush=True)
    if outcome != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
