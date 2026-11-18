# ruff: noqa: E501
"""Hourly synthetic check of a deployed JournalPulse (run by .github/workflows/synthetic.yml).

    python scripts/synthetic_check.py --base-url https://journalpulse.vercel.app \
        --history status/history.jsonl --badge status/availability.json

Probes, with no credentials and no journal text:

* ``GET /health`` must return 200; its ``X-JournalPulse-Commit`` header names the deployed commit;
* ``GET /ready`` must return 200 (every required layer ready, see docs/OPERATIONS.md);
* ``POST /v1/demo/sessions`` with no token says whether the demo is live: 404 means the demo is
  off (not a failure), 401 means the route exists. The workflow runs the browser demo journey
  only once the demo is live and passes its outcome as ``--journey``; a failed journey counts
  as down.

One JSON line is appended to the history. The badge file is a shields.io endpoint with the
availability over the last 30 days (a check counts as up when /health and /ready pass and any demo journey run passed).
Standard library only; exits 0 so the workflow can record the result before it decides.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime, timedelta
from pathlib import Path

TIMEOUT_SECONDS = 20
WINDOW = timedelta(days=30)


def probe(url: str, method: str = "GET") -> tuple[int, dict[str, str], float]:
    request = urllib.request.Request(url, method=method, data=b"{}" if method == "POST" else None, headers={
        "User-Agent": "journalpulse-synthetic-check", "Content-Type": "application/json",
    })
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:  # noqa: S310 - https URL from config
            status, headers = response.status, dict(response.headers)
    except urllib.error.HTTPError as error:
        status, headers = error.code, dict(error.headers)
    except (urllib.error.URLError, TimeoutError, ConnectionError):
        status, headers = 0, {}
    return status, {k.lower(): v for k, v in headers.items()}, round(time.perf_counter() - started, 3)


def check(base_url: str, now: datetime, journey: str = "skipped") -> dict[str, object]:
    base = base_url.rstrip("/")
    health, health_headers, health_seconds = probe(f"{base}/health")
    ready, _, ready_seconds = probe(f"{base}/ready")
    demo_status, _, _ = probe(f"{base}/v1/demo/sessions", method="POST")
    demo = {404: "off", 401: "live", 403: "live"}.get(demo_status, f"unexpected:{demo_status}")
    return {
        "time": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "health": health,
        "ready": ready,
        "commit": health_headers.get("x-journalpulse-commit", "unknown"),
        "health_seconds": health_seconds,
        "ready_seconds": ready_seconds,
        "demo": demo,
        "journey": journey,
        "up": health == 200 and ready == 200 and journey != "failure",
    }


def read_history(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def consecutive_failures(history: list[dict]) -> int:
    count = 0
    for entry in reversed(history):
        if entry.get("up"):
            break
        count += 1
    return count


def availability(history: list[dict], now: datetime) -> tuple[float | None, int]:
    since = now - WINDOW
    recent = [
        entry for entry in history
        if datetime.strptime(entry["time"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC) >= since
    ]
    if not recent:
        return None, 0
    return 100 * sum(1 for entry in recent if entry.get("up")) / len(recent), len(recent)


def badge(percent: float | None, checks: int) -> dict[str, object]:
    if percent is None:
        return {"schemaVersion": 1, "label": "30-day availability", "message": "no data", "color": "lightgrey"}
    color = "brightgreen" if percent >= 99.5 else "yellow" if percent >= 99.0 else "red"
    return {
        "schemaVersion": 1, "label": "30-day availability",
        "message": f"{percent:.2f}% of {checks} hourly checks", "color": color,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--badge", type=Path, required=True)
    parser.add_argument("--journey", choices=("success", "failure", "skipped"), default="skipped",
                        help="outcome of the browser demo journey, when the workflow ran it")
    arguments = parser.parse_args()
    if not arguments.base_url.startswith("https://"):
        raise SystemExit("--base-url must be an https URL")

    now = datetime.now(UTC)
    entry = check(arguments.base_url, now, arguments.journey)
    history = [*read_history(arguments.history), entry]
    arguments.history.parent.mkdir(parents=True, exist_ok=True)
    with arguments.history.open("a") as handle:
        handle.write(json.dumps(entry, sort_keys=True) + "\n")
    percent, checks = availability(history, now)
    arguments.badge.write_text(json.dumps(badge(percent, checks), sort_keys=True) + "\n")

    failures = consecutive_failures(history)
    print(json.dumps({**entry, "consecutive_failures": failures, "availability_30d": percent}, sort_keys=True))
    output = os.getenv("GITHUB_OUTPUT")
    if output:
        with open(output, "a") as handle:
            handle.write(f"up={'true' if entry['up'] else 'false'}\n")
            handle.write(f"demo={entry['demo']}\n")
            handle.write(f"consecutive_failures={failures}\n")


if __name__ == "__main__":
    main()
