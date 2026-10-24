"""Bounded delayed-intent pairs; these synthetic cases are not safety accuracy."""
import json
from pathlib import Path

import pytest

from journalpulse.safety import assess_safety

FIXTURES = json.loads((Path(__file__).parents[1] /
                      "assets/evaluation/safety/temporal-negation-2026-10-24.json").read_text())


@pytest.mark.parametrize("case", FIXTURES, ids=[c["id"] for c in FIXTURES])
def test_temporal_pairs(case):
    assert assess_safety(case["text"]).mode.value == case["expected"]
