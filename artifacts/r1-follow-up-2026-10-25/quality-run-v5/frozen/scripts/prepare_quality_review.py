"""Export a source/result-bound blind review kit; no provider calls or original edits."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from quality_eval import QualityCase, checked_packets, judge_request


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export_review(bundle: Path, output: Path) -> dict:
    if output.resolve().is_relative_to(bundle.resolve()):
        raise ValueError("Review records must stay outside the frozen bundle")
    packets = checked_packets(bundle)
    originals = [bundle / name for name in ("results.json", "summary.json", "REPORT.md")]
    if not all(p.is_file() for p in originals):
        raise ValueError("Original run reports are incomplete; no review kit was created")
    manifest = json.loads((bundle / "manifest.json").read_text())
    selected = json.loads((bundle / "review-selection.json").read_text())
    sample_ids = selected["development"] + selected["holdout"]
    if len(sample_ids) != 20 or len(set(sample_ids)) != 20:
        raise ValueError("The locked review denominator must remain20")
    rows = json.loads((bundle / "results.json").read_text())
    expected = {p["case"]["id"]: p for p in packets}
    if (len(rows) != len(expected) or len({r["id"] for r in rows}) != len(expected)
            or {r["id"] for r in rows} != set(expected) or not set(sample_ids) <= expected.keys()):
        raise ValueError("Result identities do not match the frozen case set")
    indexed = {r["id"]: r for r in rows}
    binding = {"packets_sha256": manifest["packets_sha256"],
               "source_manifest_sha256": sha(bundle / "manifest.json"),
               "model_results_sha256": sha(bundle / "results.json")}

    def context(case_id: str) -> dict:
        packet, row = expected[case_id], indexed[case_id]
        evidence = json.loads(judge_request(
            QualityCase.model_validate(packet["case"]), packet, row.get("response")
        )["messages"][1]["content"])
        return {"id": case_id, "status": row["status"], **evidence}

    # Blind context contains the exact response/case/activity descriptors, never
    # judge scores or model/provider metadata. Do not rewrite self-identifying
    # response text: the reviewer must record any resulting blinding limitation.
    blind = {**binding, "status": "AWAITING_REAL_INDEPENDENT_REVIEW",
             "sample_denominator": 20, "cases": [context(i) for i in sample_ids],
             "blinding_note": "Identity metadata and judge grades are omitted; preserve exact response text."
                              " Record any self-disclosure that compromises blinding."}
    additional_ids = [i for i, p in expected.items() if p["safety"]["mode"] == "support"
                      or indexed[i]["status"] != "judged"
                      or indexed[i].get("grade", {}).get("critical_safety") != "pass"]
    adjudication = {**binding, "open_only_after_blind_grades_saved": True,
                   "cases": [{**context(i), "original_judge_grade": indexed[i].get("grade")}
                             for i in additional_ids],
                   "instruction": "Review every support/flagged/invalid case and failures the judge missed."
                                  " Do not overwrite blind grades or original model records."}
    template = {**binding, "status": "BLANK_TEMPLATE_NOT_HUMAN_REVIEW",
                "reviewer": {"id": None, "independent": None,
                             "blind_scores_recorded_before_judge": None},
                "sampled": [{"id": i, "grade": None} for i in sample_ids],
                "adjudication": [{"id": i, "verdict": None, "rationale": None, "evidence": []}
                                 for i in additional_ids]}
    receipt = {**binding, "policy_version": manifest["policy_version"],
               "original_artifact_sha256": {p.name: sha(p) for p in originals},
               "sample_denominator": 20, "additional_review_cases": len(additional_ids),
               "human_review": "NOT_RUN", "originals_modified": False}
    # Exclusive creation refuses overwrites and prevents accidental review mixing.
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    for name, value in {"BLIND_CASES.json": blind, "ADJUDICATION_AFTER_BLIND.json": adjudication,
                        "HUMAN_REVIEW_TEMPLATE.json": template, "EXPORT_RECEIPT.json": receipt}.items():
        with (output / name).open("x") as handle:
            (output / name).chmod(0o600)
            handle.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if Path(__file__).resolve() != (args.bundle / "frozen/scripts/prepare_quality_review.py").resolve():
        raise SystemExit("Run the frozen review-export script for this bundle")
    try:
        receipt = export_review(args.bundle, args.output)
    except (ValueError, FileExistsError):
        raise SystemExit("Review export stopped: check reports, identities and a new output path") from None
    print(f"Prepared{receipt['sample_denominator']} blind cases; human review remains NOT_RUN.")


if __name__ == "__main__":
    main()
