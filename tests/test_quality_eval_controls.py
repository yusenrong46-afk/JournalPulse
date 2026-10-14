"""Offline controls: these tests never call a model or a real account."""

import importlib.util
import json
import sys
from decimal import Decimal
from pathlib import Path

import httpx
import pytest

spec = importlib.util.spec_from_file_location(
    "quality_eval", Path(__file__).parents[1] / "scripts/quality_eval.py"
)
assert spec and spec.loader
q = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = q
spec.loader.exec_module(q)


def request():
    return {
        "model": q.CANDIDATE,
        "max_tokens": 100,
        "messages": [{"role": "user", "content": "Fictional evaluation message."}],
    }


def endpoint():
    return {
        "tag": "azure",
        "provider_name": "Azure",
        "pricing": {"prompt": "0.0000001", "completion": "0.0000005"},
    }


def test_unknown_attempt_blocks_all_further_calls(tmp_path):
    ledger = q.Ledger(tmp_path / "budget.json")
    ledger.reserve(request(), Decimal("0.10"), "case-1", "candidate")
    with pytest.raises(q.EvaluationStopped, match="Unresolved"):
        ledger.reserve(request(), Decimal("0.10"), "case-2", "candidate")
    assert len(json.loads(ledger.path.read_text())["attempts"]) == 1


def test_budget_is_global_and_duplicate_attempts_are_not_retried(tmp_path):
    ledger = q.Ledger(tmp_path / "budget.json")
    attempt = ledger.reserve(request(), Decimal("9.90"), "case-1", "candidate")
    ledger.settle(attempt, "gen-fictional-1", "9.90")
    reopened = q.Ledger(ledger.path)
    with pytest.raises(q.EvaluationStopped, match="already attempted"):
        reopened.reserve(request(), Decimal("0.01"), "case-1", "candidate")
    with pytest.raises(q.EvaluationStopped, match=r"US\$10"):
        reopened.reserve(request(), Decimal("0.11"), "case-2", "candidate")


@pytest.mark.parametrize("cost", [None, True, "NaN", "Infinity", "-1", "invalid"])
def test_invalid_cost_is_never_zero(cost):
    with pytest.raises(q.EvaluationStopped):
        q.number(cost)


def test_pricing_breach_is_recorded_and_stops_execution(tmp_path):
    ledger = q.Ledger(tmp_path / "budget.json")
    attempt = ledger.reserve(request(), Decimal("0.10"), "case-1", "candidate")
    with pytest.raises(q.EvaluationStopped, match="exceeded"):
        ledger.settle(attempt, "gen-fictional-1", "0.20")
    row = json.loads(ledger.path.read_text())["attempts"][0]
    assert row["state"] == "pricing_breach" and row["actual_cost_usd"] == "0.20"


def test_transport_failure_leaves_reservation_and_does_not_retry(tmp_path):
    calls = []

    def handler(req):
        calls.append(req)
        raise httpx.ReadTimeout("fictional transport failure", request=req)

    ledger = q.Ledger(tmp_path / "budget.json")
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(q.EvaluationStopped, match="unknown charge"):
            q.paid_call(
                client, "fictional-key", request(), endpoint(), ledger, tmp_path, "case-1", "candidate"
            )
    assert len(calls) == 1
    assert json.loads(ledger.path.read_text())["attempts"][0]["state"] == "reserved"
    observation = json.loads(next((tmp_path / "observations").glob("*.json")).read_text())
    assert observation["transport_failed"] is True and observation["http_status"] is None
    assert observation["stage"] == "candidate" and observation["latency_ms"] >= 0


def test_recorded_cost_precedes_model_identity_validation(tmp_path):
    def handler(req):
        return httpx.Response(
            200,
            json={
                "id": "gen-fictional",
                "model": "wrong-model",
                "provider": "Azure",
                "usage": {"cost": 0.0001},
            },
        )

    ledger = q.Ledger(tmp_path / "budget.json")
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(q.EvaluationStopped, match="Returned model"):
            q.paid_call(
                client, "fictional-key", request(), endpoint(), ledger, tmp_path, "case-1", "candidate"
            )
    row = json.loads(ledger.path.read_text())["attempts"][0]
    assert row["state"] == "settled" and row["actual_cost_usd"] == "0.0001"
    assert list((tmp_path / "responses").glob("*.json"))
    observation = json.loads(next((tmp_path / "observations").glob("*.json")).read_text())
    assert observation["transport_failed"] is False and observation["http_status"] == 200


def test_routes_preserve_privacy_and_reserve_a_conservative_cost():
    routed, cost = q.routed(request(), endpoint())
    assert routed["provider"]["zdr"] is True
    assert routed["provider"]["require_parameters"] is True
    assert routed["provider"]["allow_fallbacks"] is False
    assert routed["provider"]["only"] == ["azure"]
    assert cost > Decimal("0.0004")


def test_frozen_case_counts_and_family_split():
    data = json.loads(
        (Path(__file__).parents[1] / "assets/evaluation/luna-quality-2026-10-14.json").read_text()
    )
    dataset = q.Dataset.model_validate(data)
    assert len(dataset.cases) == 100
    data["cases"][0]["split"] = "holdout"
    with pytest.raises(ValueError):
        q.Dataset.model_validate(data)


def test_judge_schema_is_provider_compatible_without_relaxing_local_validation():
    schema = q.judge_schema()
    for name in q.DIMENSIONS:
        assert schema["properties"][name]["enum"] == [1, 2, 3, 4, 5]
    for field in schema["properties"].values():
        assert not ({"minimum", "maximum", "minLength", "maxLength", "maxItems"} & field.keys())
    with pytest.raises(ValueError):
        q.Grade.model_validate(
            {
                "relevance": 6,
                "grounding": 5,
                "instruction_following": 5,
                "usefulness": 5,
                "critical_safety": "pass",
                "rationale": "Test",
                "evidence": ["Fictional"],
            }
        )


def test_all_packets_prepare_and_frozen_inputs_detect_changes(tmp_path):
    bundle = tmp_path / "frozen-test"
    q.prepare(bundle)
    packets = q.checked_packets(bundle)
    assert len(packets) == 100
    assert all(p["request"]["model"] == q.CANDIDATE for p in packets)
    assert all(p["request"]["provider"]["zdr"] for p in packets)
    assert all(p["case"]["origin"]["kind"] == "original_fictional" for p in packets)
    (bundle / "frozen/src/journalpulse/safety.py").write_text("changed")
    with pytest.raises(q.EvaluationStopped, match="Frozen source changed"):
        q.checked_packets(bundle)
