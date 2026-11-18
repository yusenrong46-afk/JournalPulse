"""The hourly synthetic check's arithmetic: availability window, failure streak, badge."""

from __future__ import annotations

import importlib.util
import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("synthetic_check", ROOT / "scripts" / "synthetic_check.py")
assert _spec and _spec.loader
synthetic = importlib.util.module_from_spec(_spec)
sys.modules["synthetic_check"] = synthetic
_spec.loader.exec_module(synthetic)

NOW = datetime(2026, 11, 18, 12, 0, tzinfo=UTC)


def entry(hours_ago: float, up: bool) -> dict:
    return {"time": (NOW - timedelta(hours=hours_ago)).strftime("%Y-%m-%dT%H:%M:%SZ"), "up": up}


def test_two_consecutive_failures_are_counted_from_the_newest_entry() -> None:
    history = [entry(5, False), entry(4, True), entry(3, False), entry(2, False)]
    assert synthetic.consecutive_failures(history) == 2
    assert synthetic.consecutive_failures([*history, entry(1, True)]) == 0
    assert synthetic.consecutive_failures([]) == 0


def test_availability_counts_only_the_last_thirty_days() -> None:
    history = [entry(24 * 31, False), entry(3, True), entry(2, True), entry(1, False), entry(0, True)]
    percent, checks = synthetic.availability(history, NOW)
    assert checks == 4
    assert percent == 75.0
    assert synthetic.availability([], NOW) == (None, 0)


def test_badge_is_a_shields_endpoint_and_says_when_there_is_no_data() -> None:
    assert synthetic.badge(None, 0)["message"] == "no data"
    good = synthetic.badge(99.9, 700)
    assert good == {
        "schemaVersion": 1, "label": "30-day availability",
        "message": "99.90% of 700 hourly checks", "color": "brightgreen",
    }
    assert synthetic.badge(98.0, 10)["color"] == "red"


def test_a_failed_demo_journey_counts_as_down(monkeypatch) -> None:
    answers = {"/health": (200, {"x-journalpulse-commit": "abc1234"}), "/ready": (200, {}),
               "/v1/demo/sessions": (401, {})}

    def fake_probe(url: str, method: str = "GET"):
        status, headers = answers[url.removeprefix("https://example.test")]
        return status, headers, 0.01

    monkeypatch.setattr(synthetic, "probe", fake_probe)
    passed = synthetic.check("https://example.test", NOW, "success")
    assert passed["up"] is True and passed["demo"] == "live" and passed["commit"] == "abc1234"
    assert synthetic.check("https://example.test", NOW, "failure")["up"] is False
    answers["/ready"] = (503, {})
    assert synthetic.check("https://example.test", NOW)["up"] is False
    json.dumps(passed)  # one JSON line per check
