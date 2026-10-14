"""Offline development diagnostics. No providers, credentials, or clinical claims."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from journalpulse.config import PROJECT_ROOT
from journalpulse.safety import assess_safety


def evaluate(path: Path) -> dict:
    dataset = json.loads(path.read_text())
    rows = []
    counts: Counter = Counter()
    for case in dataset["cases"]:
        expected = case["expected"]
        if expected not in {"normal", "support", "review"}:
            raise ValueError("Unknown expected label")
        observed = assess_safety(case["text"]).mode.value
        outcome = (
            "review_required" if expected == "review" else
            "pass" if observed == expected else
            "missed_support" if expected == "support" else "false_alarm"
        )
        rows.append({**case, "observed": observed, "outcome": outcome})
        counts[outcome] += 1
    return {
        "dataset_version": dataset["version"], "provenance": dataset["provenance"],
        "qualification": "NOT_QUALIFIED", "total_cases": len(rows),
        "scored_cases": len(rows) - counts["review_required"],
        "counts": dict(counts), "cases": rows,
        "limits": "Development diagnostics only. Unresolved and review cases remain visible. "
        "Do not use this set for training and then claim held-out performance.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases", type=Path, default=PROJECT_ROOT / "assets/evaluation/safety/development.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate(args.cases)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    summary_keys = ("qualification", "total_cases", "scored_cases", "counts")
    print(json.dumps({key: report[key] for key in summary_keys}))
    raise SystemExit(1 if any(report["counts"].get(key) for key in
                             ("missed_support", "false_alarm", "review_required")) else 0)


if __name__ == "__main__":
    main()
