"""Executable fictional pilot examples from the PRD; no real observations or telemetry.

These assertions demonstrate denominators and costs. They are preparation for a
manual research report, not a product analytics engine or a participant result.
"""
from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from decimal import Decimal


def cost_per_completion(charges: list[str | None], completions: int) -> Decimal | None:
    if completions == 0 or any(c is None for c in charges):
        return None
    return sum((Decimal(c) for c in charges if c is not None), Decimal()) / completions


def examples() -> dict:
    # Replayed receipt W1 does not establish another workflow or another charge.
    recorded = [
        {"id": "W1", "requested": "AI", "actual": "AI", "complete": True, "charges": ["0.03", "0.04"]},
        {"id": "W2", "requested": "AI", "actual": "AI", "complete": False, "charges": ["0.05"]},
        {"id": "W3", "requested": "Simple", "actual": "Simple", "complete": True, "charges": []},
    ]
    recorded.append(dict(recorded[0]))
    workflows = {w["id"]: w for w in recorded}
    assert len(workflows) == 3
    ai = [w for w in workflows.values() if w["requested"] == "AI"]
    ai_complete = sum(w["complete"] and w["actual"] == "AI" for w in ai)
    total = sum((Decimal(c) for w in workflows.values() for c in w["charges"]), Decimal())
    assert total == Decimal("0.12") and ai_complete == 1
    assert sum(w["complete"] for w in workflows.values()) == 2
    cost_per_ai = cost_per_completion([c for w in workflows.values() for c in w["charges"]], ai_complete)
    assert cost_per_ai == Decimal("0.12")
    assert cost_per_completion(["0.12"], 0) is None
    # Fallback is retained in requested-mode accounting but is never AI completion.
    fallback = {"requested": "AI", "actual": "Simple", "complete": True}
    assert not (fallback["actual"] == "AI" and fallback["complete"])
    # Modern reports include negative participation. Timer expiry adds no report.
    reports = {"S1": "completed", "S2": "partial", "S3": "not_tried", "S4": "stopped"}
    confirmed = {"S1", "S2", "S3", "S4", "S5"}
    tried = sum(v in {"completed", "partial"} for v in reports.values())
    assert tried == 2 and len(reports) == 4 and len(confirmed - reports.keys()) == 1
    start = datetime(2026, 11, 1, tzinfo=UTC)
    end = start + timedelta(hours=168)
    # Fully observed zero use stays eligible. Unknown/partial windows stay separate.
    cohort = [
        {"id": "P1", "mature": True, "complete_observation": True, "activation": start},
        {"id": "P2", "mature": True, "complete_observation": True, "activation": None},
        {"id": "P3", "mature": True, "complete_observation": False, "activation": None},
        {"id": "P4", "mature": False, "complete_observation": True, "activation": start},
        {"id": "P5", "mature": True, "complete_observation": True,
         "activation": end - timedelta(hours=2)},
    ]
    eligible = [p for p in cohort if p["mature"] and p["complete_observation"]]
    activated = [p for p in eligible if p["activation"] is not None]
    return_eligible = [p for p in activated if end - p["activation"] >= timedelta(hours=24)]
    assert len(eligible) == 3 and len(activated) == 2 and len(return_eligible) == 1
    # A later day uses enrollment-relative blocks, not local midnight.
    assert int(((start + timedelta(hours=25)) - start).total_seconds() // 86400) == 1
    # Failed and incomplete cohorts still carry attributable charges; an unknown
    # charge prevents an exact cost claim, rather than being silently zero.
    assert cost_per_completion(["0.05", None], 1) is None
    return {
        "scope": "FICTIONAL_CALCULATION_ONLY_NOT_PILOT_EVIDENCE",
        "build": "fictional-build-A; do not pool another build silently",
        "workflow_overall": {"completed": 2, "fully_followed": 3},
        "workflow_AI": {"completed": 1, "fully_followed_requested_AI": 2},
        "workflow_Simple": {"completed": 1, "fully_followed_requested_Simple": 1},
        "provider_cost_usd": str(total), "cost_per_completed_AI_usd": str(cost_per_ai),
        "cost_per_completed_all_modes_usd": str(total / 2),
        "zero_AI_completion_cost": None, "unknown_charge_exact_cost": None,
        "reported_tried": {"tried_sessions": tried, "reported_sessions": len(reports),
                           "confirmed_sessions": len(confirmed), "no_report_sessions": 1},
        "participant_accounting": {"enrolled": 5, "fully_observed_mature": 3,
                                   "unknown_observation": 1, "partial_window": 1,
                                   "activated": 2, "known_zero_use": 1,
                                   "return_eligible": 1, "late_activations": 1},
        "real_participants": 0, "real_pilot_status": "NOT_RUN",
    }


if __name__ == "__main__":
    print(json.dumps(examples(), indent=2))
