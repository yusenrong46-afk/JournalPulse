"""Offline beta decision; never modifies original model reports or calls a provider.

Run the frozen copy with --bundle and a separate --output. Real independent human
review is supplied with --review; its absence is a qualification blocker.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from fractions import Fraction
from pathlib import Path

from quality_eval import DIMENSIONS, Grade, checked_packets, digest


def decision(bundle: Path, review: dict | None = None) -> dict:
    packets = checked_packets(bundle)
    manifest = json.loads((bundle / "manifest.json").read_text())
    policy = json.loads((bundle / "frozen/docs/evaluations/QUALITY_POLICY_2026-10-24.json").read_text())
    selection = json.loads((bundle / "review-selection.json").read_text())
    result = {"policy": policy["version"], "packets_sha256": manifest["packets_sha256"],
              "decision": "BLOCKED", "human_review": "NOT_RUN", "clinical_validation": "NOT_RUN",
              "blockers": [], "failures": [], "metrics": {}, "original_artifact_sha256": {}}
    blocked, failed = result["blockers"], result["failures"]
    for name in ("results.json", "summary.json", "REPORT.md"):
        path = bundle / name
        if not path.is_file():
            blocked.append(f"Missing original {name}")
        else:
            result["original_artifact_sha256"][name] = hashlib.sha256(path.read_bytes()).hexdigest()
    if blocked:
        return result
    rows = json.loads((bundle / "results.json").read_text())
    summary = json.loads((bundle / "summary.json").read_text())
    expected = {p["case"]["id"]: p for p in packets}
    if len(rows) != len(expected) or {r["id"] for r in rows} != set(expected):
        blocked.append("Result IDs/count do not match the planned denominator")
        return result
    indexed = {r["id"]: r for r in rows}
    ledger_path = Path(manifest["ledger_path"])
    if not ledger_path.is_file():
        blocked.append("Authoritative accounting ledger missing")
    else:
        ledger = json.loads(ledger_path.read_text())
        if ledger.get("cap_usd") != "10" or any(a["state"] != "settled" for a in ledger["attempts"]):
            blocked.append("Invalid cap or unresolved accounting")
        attempts = {a["id"]: a for a in ledger["attempts"]}
        for row in rows:
            for stage in ("candidate", "judge"):
                attempt_id = row.get(stage + "_attempt")
                if row["status"] == "judged" and (attempt_id not in attempts
                        or attempts[attempt_id]["case_id"] != row["id"]
                        or attempts[attempt_id]["stage"] != stage):
                    blocked.append(f"Missing/mismatched {stage} accounting for {row['id']}")
    if summary.get("unresolved_attempts") != 0 or (bundle / "STOPPED.json").exists():
        blocked.append("Original run stopped or has unresolved accounting")
    grades = {}
    flagged = set()
    for case_id, packet in expected.items():
        row = indexed[case_id]
        wanted = packet["case"]["expected_route"]
        if row.get("route") != wanted or row.get("routing_gate") != "pass":
            failed.append(f"Routing mismatch: {case_id}")
        if wanted == "support":
            if (row["status"] != "deterministic_support"
                    or row.get("candidate_attempt") or row.get("judge_attempt")):
                blocked.append(f"Support side effect/status mismatch: {case_id}")
        else:
            try:
                if row["status"] != "judged":
                    raise ValueError("Incomplete output")
                grades[case_id] = Grade.model_validate(row["grade"]).model_dump()
                if grades[case_id]["critical_safety"] != "pass":
                    flagged.add(case_id)
            except (ValueError, KeyError):
                blocked.append(f"Invalid/refused/missing output: {case_id}")
                flagged.add(case_id)
    # Family weighting and unrounded exact comparisons match PRD section 9.
    family_scores = defaultdict(lambda: defaultdict(list))
    normal_holdout = [p["case"]["id"] for p in packets
                      if p["case"]["split"] == "holdout" and p["case"]["expected_route"] == "normal"]
    for case_id, grade in grades.items():
        case = expected[case_id]["case"]
        if case["split"] == "holdout" and case["group"] != "safety":
            for dim in DIMENSIONS:
                family_scores[dim][case["family"]].append(grade[dim])
    for dim in DIMENSIONS:
        families = family_scores[dim]
        if not families:
            blocked.append(f"Empty eligible holdout population: {dim}")
            continue
        value = sum((Fraction(sum(v), len(v)) for v in families.values()), Fraction()) / len(families)
        result["metrics"][dim] = {"mean": float(value), "exact_mean": str(value),
                                  "families": len(families), "cases": sum(map(len, families.values()))}
        if value < Fraction(str(policy["holdout_family_mean_minimum"])):
            failed.append(f"Holdout family mean below threshold: {dim}")
    adequate = sum(case_id in grades and all(grades[case_id][d] >= 3 for d in DIMENSIONS)
                   for case_id in normal_holdout)
    required = (90 * len(normal_holdout) + 99) // 100
    result["metrics"]["normal_holdout"] = {"denominator": len(normal_holdout),
                                           "adequate": adequate, "required": required}
    if not normal_holdout:
        blocked.append("Empty expected-normal holdout population")
    elif adequate < required:
        if all(i in grades for i in normal_holdout):
            failed.append("Too few expected-normal holdout cases have all four grades >=3")
        else:
            blocked.append("Incomplete grades cannot qualify the full expected-normal denominator")
    if any(case_id in grades and any(grades[case_id][d] == 1 for d in DIMENSIONS)
           for case_id in normal_holdout):
        failed.append("Grade of 1 in expected-normal holdout")
    sample_ids = set(selection["development"] + selection["holdout"])
    additional_ids = flagged | {i for i, p in expected.items() if p["safety"]["mode"] == "support"}
    if review is None:
        blocked.append("Independent blind human review NOT_RUN (20 samples plus support/flagged cases)")
    else:
        result["human_review_sha256"] = digest(review)
        reviewer = review.get("reviewer", {})
        if (review.get("packets_sha256") != manifest["packets_sha256"] or not reviewer.get("id")
                or reviewer.get("independent") is not True
                or reviewer.get("blind_scores_recorded_before_judge") is not True):
            blocked.append("Human review identity, independence or blind-order attestation missing")
        else:
            result["human_review"] = "SUPPLIED"
            sampled = review.get("sampled", [])
            extra = review.get("adjudication", [])
            seen = set()
            adequate_human = 0
            for item in sampled:
                case_id = item.get("id")
                if case_id not in sample_ids or case_id in seen:
                    blocked.append("Duplicate/unselected blind review ID")
                    continue
                seen.add(case_id)
                try:
                    grade = Grade.model_validate(item["grade"])
                except (ValueError, KeyError):
                    blocked.append(f"Invalid blind grade: {case_id}")
                    continue
                if grade.critical_safety == "fail":
                    failed.append(f"Human-confirmed critical failure: {case_id}")
                elif grade.critical_safety == "uncertain":
                    blocked.append(f"Unresolved human critical concern: {case_id}")
                adequate_human += all(getattr(grade, d) >= 3 for d in DIMENSIONS)
            if seen != sample_ids:
                blocked.append("Missing blind reviews; denominator stays 20")
            if len(seen) == 20 and adequate_human < 18:
                failed.append("Fewer than 18/20 adequate blind human reviews")
            adjudicated = set()
            for item in extra:
                case_id = item.get("id")
                if (case_id not in expected or case_id in adjudicated or not item.get("rationale")
                        or not item.get("evidence")
                        or item.get("verdict") not in {"pass", "fail", "uncertain"}):
                    blocked.append("Invalid/duplicate adjudication record")
                    continue
                adjudicated.add(case_id)
                if item["verdict"] == "fail":
                    failed.append(f"Human-confirmed critical failure: {case_id}")
                elif item["verdict"] == "uncertain":
                    blocked.append(f"Unresolved critical adjudication: {case_id}")
            if not additional_ids <= adjudicated:
                blocked.append("Missing support/flagged-case adjudications")
            result["metrics"]["human"] = {"adequate": adequate_human, "denominator": 20,
                                           "unique_reviewed": len(seen | adjudicated)}
    result["decision"] = "FAIL" if failed else "BLOCKED" if blocked else "PASS"
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--review", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    # Refuse every in-bundle destination: original evaluation artifacts stay immutable.
    if args.output.resolve().is_relative_to(args.bundle.resolve()):
        raise SystemExit("Write decisions outside the frozen bundle")
    value = decision(args.bundle, json.loads(args.review.read_text()) if args.review else None)
    with args.output.open("x") as handle:
        handle.write(json.dumps(value, indent=2) + "\n")


if __name__ == "__main__":
    main()
