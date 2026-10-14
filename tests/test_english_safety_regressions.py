"""English-only intended-routing checks; review cases are deliberately not relabelled as passes."""
import json
from pathlib import Path

import pytest

from journalpulse.safety import assess_safety

CASE_FILE = Path(__file__).parents[1] / "assets/evaluation/safety/english-development.json"
DATA = json.loads(CASE_FILE.read_text())
CLEAR = [case for case in DATA["cases"] if case["expected"] != "review"]


@pytest.mark.parametrize("case", CLEAR, ids=[case["id"] for case in CLEAR])
def test_english_routing(case):
    assert assess_safety(case["text"]).mode.value == case["expected"]
