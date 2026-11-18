# ruff: noqa: E501
"""Local incident drill: the database becomes unreachable, then recovers.

    uv run python scripts/incident_drill.py --db-container <name>     # host runtime
    uv run python scripts/incident_drill.py --runtime compose         # Compose runtime

It stands up an onboarding environment (scripts/onboard.py), signs in an anonymous demo
identity, then makes PostgreSQL unreachable by pausing its container (``docker pause``; with
Compose, ``docker compose pause db``). It records, with times:

1. when ``/ready`` turns 503 and what its checks say;
2. that a journal save during the outage is refused with 503 and nothing is stored;
3. recovery (``docker unpause``), when ``/ready`` is 200 again, and that retrying the same
   save (same ``client_request_id``) stores exactly one entry.

The timeline is printed as JSON and written to outputs/incident-drill/<date>.json, which
docs/INCIDENT_DRILL.md cites. Only a container you name (or the Compose one) is paused; the
fictional entry text is the only content written. Never run it against a shared database.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from onboard import Environment, http_get  # noqa: E402

ENTRY = "Fictional drill entry: a slow walk helped."


def call(url: str, method: str, body: dict | None, headers: dict[str, str], timeout: float = 60) -> tuple[int, dict]:
    request = urllib.request.Request(
        url, method=method, data=json.dumps(body).encode() if body is not None else None,
        headers={"Content-Type": "application/json", **headers},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 - loopback only
            return response.status, json.loads(response.read() or b"{}")
    except urllib.error.HTTPError as error:
        try:
            return error.code, json.loads(error.read() or b"{}")
        except json.JSONDecodeError:
            return error.code, {}
    except (urllib.error.URLError, TimeoutError) as error:
        return 0, {"error": error.__class__.__name__}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runtime", choices=("host", "compose"), default="host")
    parser.add_argument("--db-container", help="host runtime: the disposable local PostgreSQL container to pause")
    arguments = parser.parse_args()
    if arguments.runtime == "host" and not arguments.db_container:
        raise SystemExit("--db-container is required with the host runtime (the container behind JOURNALPULSE_PG_DSN).")

    environment = Environment("incident-drill", arguments.runtime)
    timeline: list[dict] = []
    started = time.monotonic()

    def note(event: str, **details: object) -> None:
        item = {"t_seconds": round(time.monotonic() - started, 1), "event": event, **details}
        timeline.append(item)
        print(json.dumps(item), flush=True)

    def pause(paused: bool) -> None:
        verb = "pause" if paused else "unpause"
        if arguments.runtime == "compose":
            environment.compose(verb, "db")
        else:
            subprocess.run(["docker", verb, arguments.db_container], check=True, capture_output=True)

    def ready() -> tuple[int, dict]:
        status, _, body = http_get(f"{environment.api_url}/ready", timeout=60)
        try:
            return status, json.loads(body)
        except json.JSONDecodeError:
            return status, {}

    def wait_ready(expected: int, limit: float = 180) -> dict:
        deadline = time.monotonic() + limit
        while time.monotonic() < deadline:
            status, body = ready()
            if status == expected:
                return body
            time.sleep(2)
        raise RuntimeError(f"/ready did not return {expected} within {limit} s")

    paused = False
    try:
        for name, action in (
            ("start database", environment.start_database), ("apply migrations", environment.prepare_database),
            ("build", environment.build), ("start services", environment.start_services),
            ("/ready", environment.wait_ready),
        ):
            environment.step(name, action)
        note("environment ready", runtime=arguments.runtime)

        status, session = call(
            f"http://127.0.0.1:{environment.auth_port}/auth/v1/signup", "POST", {}, {"apikey": environment.anon}
        )
        assert status == 200, status
        headers = {"Authorization": f"Bearer {session['access_token']}"}
        request_id = str(uuid4())
        body = {"text": ENTRY, "client_request_id": request_id}
        note("signed in an anonymous drill identity")

        pause(True)
        paused = True
        note("database paused (unreachable)")
        outage = wait_ready(503)
        note("/ready returned 503", checks={k: v for k, v in outage.get("checks", {}).items() if k in {"database", "schema", "signing"}})
        status, detail = call(f"{environment.api_url}/v1/journal/entries", "POST", body, headers)
        note("journal save during outage", status=status, detail=detail.get("detail"))
        assert status == 503, (status, detail)

        pause(False)
        paused = False
        note("database resumed")
        wait_ready(200)
        note("/ready returned 200")
        status, saved = call(f"{environment.api_url}/v1/journal/entries", "POST", body, headers)
        note("same save retried after recovery", status=status)
        assert status in {200, 201}, (status, saved)
        status, page = call(f"{environment.api_url}/v1/journal/entries", "GET", None, headers)
        count = len(page.get("items", []))
        note("entries stored for the drill identity", count=count)
        assert count == 1, count  # one entry: the refused attempt stored nothing
    except AssertionError as error:
        note("drill check failed", detail=str(error))
        raise
    finally:
        if paused:
            pause(False)
        environment.stop()
        output = ROOT / "outputs" / "incident-drill"
        output.mkdir(parents=True, exist_ok=True)
        path = output / f"{datetime.now(UTC).strftime('%Y-%m-%d')}.json"
        path.write_text(json.dumps({"commit": environment.commit, "runtime": arguments.runtime, "timeline": timeline}, indent=2) + "\n")
        print(f"timeline written to {path.relative_to(ROOT)}", flush=True)


if __name__ == "__main__":
    main()
