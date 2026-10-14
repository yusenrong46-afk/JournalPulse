"""Bounded regression cases, not a multilingual or clinical qualification."""
import json
from pathlib import Path

import pytest

from journalpulse.safety import assess_safety

DATA = json.loads((Path(__file__).parents[1] / "assets/evaluation/safety/development.json").read_text())
CASES = [case for case in DATA["cases"] if case["language"] == "en" and case["expected"] != "review"]


@pytest.mark.parametrize("case", CASES, ids=[case["id"] for case in CASES])
def test_clear_english_development_regressions(case):
    assert assess_safety(case["text"]).mode.value == case["expected"]
