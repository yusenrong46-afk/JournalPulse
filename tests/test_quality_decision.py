"""Synthetic policy mechanics only; no model or human qualification is claimed."""
import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
spec = importlib.util.spec_from_file_location("quality_decision", Path(__file__).parents[1] /
                                               "scripts/quality_decision.py")
assert spec and spec.loader
qd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(qd)


def grade(value=4, critical="pass"):
    return dict.fromkeys(qd.DIMENSIONS, value) | {
        "critical_safety": critical, "rationale": "Fictional test evidence", "evidence": ["Example"]}


@pytest.fixture
def cohort(tmp_path, monkeypatch):
    packets, rows, attempts = [], [], []
    selection = {"development": [], "holdout": []}
    for split in selection:
        for i in range(10):
            case_id = f"{split}-{i}"
            selection[split].append(case_id)
            packets.append({"case": {"id": case_id, "split": split, "expected_route": "normal",
                                     "group": "safety" if i == 9 else "general", "family": f"f-{i}"},
                            "safety": {"mode": "normal"}})
            rows.append({"id": case_id, "route": "normal", "routing_gate": "pass", "status": "judged",
                         "grade": grade(), "candidate_attempt": case_id + "c",
                         "judge_attempt": case_id + "j"})
            for suffix, stage in [("c", "candidate"), ("j", "judge")]:
                attempts.append({"id": case_id + suffix, "case_id": case_id, "stage": stage,
                                 "state": "settled", "actual_cost_usd": "0.01"})
    for i in range(2):
        case_id = f"support-{i}"
        packets.append({"case": {"id": case_id, "split": "holdout", "expected_route": "support",
                                 "group": "safety", "family": "support"}, "safety": {"mode": "support"}})
        rows.append({"id": case_id, "route": "support", "routing_gate": "pass",
                     "status": "deterministic_support"})
    policy = Path(__file__).parents[1] / "docs/evaluations/QUALITY_POLICY_2026-10-24.json"
    policy_target = tmp_path / "frozen/docs/evaluations/QUALITY_POLICY_2026-10-24.json"
    policy_target.parent.mkdir(parents=True)
    policy_target.write_bytes(policy.read_bytes())
    ledger_path = tmp_path / "ledger.json"
    ledger_path.write_text(json.dumps({"cap_usd": "10", "attempts": attempts}))
    for name, value in {"manifest.json": {"packets_sha256": "fictional", "ledger_path": str(ledger_path)},
                        "review-selection.json": selection, "results.json": rows,
                        "summary.json": {"unresolved_attempts": 0}}.items():
        (tmp_path / name).write_text(json.dumps(value))
    (tmp_path / "REPORT.md").write_text("Fictional policy test; not a model report")
    monkeypatch.setattr(qd, "checked_packets", lambda _: packets)
    review = {"packets_sha256": "fictional", "reviewer": {"id": "fixture-only",
               "independent": True, "blind_scores_recorded_before_judge": True},
              "sampled": [{"id": i, "grade": grade()} for ids in selection.values() for i in ids],
              "adjudication": [{"id": f"support-{i}", "verdict": "pass", "rationale": "Fixture only",
                                "evidence": ["Example"]} for i in range(2)]}
    bind_review(tmp_path, review)
    return tmp_path, rows, review


def bind_review(bundle, review):
    review["source_manifest_sha256"] = hashlib.sha256((bundle / "manifest.json").read_bytes()).hexdigest()
    review["model_results_sha256"] = hashlib.sha256((bundle / "results.json").read_bytes()).hexdigest()


def update(bundle, rows, review):
    (bundle / "results.json").write_text(json.dumps(rows))
    bind_review(bundle, review)


def test_missing_human_review_blocks_and_originals_are_unchanged(cohort):
    bundle, _, _ = cohort
    before = {p.name: p.read_bytes() for p in bundle.glob('*') if p.is_file()}
    assert qd.decision(bundle)["decision"] == "BLOCKED"
    assert before == {p.name: p.read_bytes() for p in bundle.glob('*') if p.is_file()}


def test_complete_fictional_policy_control(cohort):
    bundle, _, review = cohort
    assert qd.decision(bundle, review)["decision"] == "PASS"


def test_critical_failure_fails_even_with_high_averages_and_missing_reviews(cohort):
    bundle, _, review = cohort
    review["sampled"] = review["sampled"][:1]
    review["sampled"][0]["grade"] = grade(5, "fail")
    assert qd.decision(bundle, review)["decision"] == "FAIL"


def test_judge_flag_needs_separate_human_adjudication(cohort):
    bundle, rows, review = cohort
    rows[0]["grade"] = grade(5, "uncertain")
    update(bundle, rows, review)
    assert qd.decision(bundle, review)["decision"] == "BLOCKED"
    review["adjudication"].append({"id": rows[0]["id"], "verdict": "pass",
                                   "rationale": "Fictional unsupported-flag control",
                                   "evidence": ["Example"]})
    assert qd.decision(bundle, review)["decision"] == "PASS"
    assert json.loads((bundle / "results.json").read_text())[0]["grade"]["critical_safety"] == "uncertain"


def test_missing_results_stay_in_denominator(cohort):
    bundle, rows, review = cohort
    rows[-3]["status"] = "unobserved"
    rows[-3].pop("grade")
    update(bundle, rows, review)
    result = qd.decision(bundle, review)
    assert result["decision"] == "BLOCKED"
    assert result["metrics"]["normal_holdout"]["denominator"] == 10


def test_safety_group_normal_controls_are_in_quality_tail(cohort):
    bundle, rows, review = cohort
    rows[-3]["grade"] = grade(1)
    update(bundle, rows, review)
    result = qd.decision(bundle, review)
    assert result["decision"] == "FAIL"
    assert result["metrics"]["normal_holdout"]["denominator"] == 10
    assert result["metrics"]["relevance"]["cases"] == 9


def test_duplicate_reviews_do_not_replace_missing_people(cohort):
    bundle, _, review = cohort
    review["sampled"][-1] = copy.deepcopy(review["sampled"][0])
    assert qd.decision(bundle, review)["decision"] == "BLOCKED"


def test_missing_accounting_blocks(cohort):
    bundle, _, review = cohort
    (bundle / "ledger.json").unlink()
    assert qd.decision(bundle, review)["decision"] == "BLOCKED"


def test_missing_summary_does_not_hide_a_confirmed_critical_failure(cohort):
    bundle, _, review = cohort
    (bundle / "REPORT.md").unlink()
    review["sampled"][0]["grade"] = grade(5, "fail")
    assert qd.decision(bundle, review)["decision"] == "FAIL"


def test_invalid_human_grades_block_instead_of_becoming_observed_low_scores(cohort):
    bundle, _, review = cohort
    for row in review["sampled"][:3]:
        row["grade"] = None
    result = qd.decision(bundle, review)
    assert result["decision"] == "BLOCKED"
    assert result["metrics"]["human"]["denominator"] == 20


@pytest.mark.parametrize('field', ['source_manifest_sha256', 'model_results_sha256'])
def test_human_review_from_another_source_or_output_cannot_qualify(cohort, field):
    bundle, _, review = cohort
    review[field] = '0' * 64
    assert qd.decision(bundle, review)['decision'] == 'BLOCKED'
