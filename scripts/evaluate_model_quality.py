#!/usr/bin/env python3
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from emotion_journal.evaluation import evaluate_prediction_cases, load_eval_cases
from emotion_journal.model import get_default_predictor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate JournalPulse on realistic synthetic journal cases.")
    parser.add_argument(
        "--eval-file",
        type=Path,
        default=PROJECT_ROOT / "data" / "evals" / "journalpulse_eval.jsonl",
        help="JSONL file of eval cases.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "reports" / "model_quality_eval.json",
        help="Where to write the JSON report.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Evaluate only the first N cases for a quick smoke run.",
    )
    parser.add_argument(
        "--fail-under-accepted",
        type=float,
        default=None,
        help="Exit non-zero when accepted accuracy is below this threshold.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cases = load_eval_cases(args.eval_file)
    if args.limit is not None:
        cases = cases[: args.limit]

    report = evaluate_prediction_cases(get_default_predictor(), cases)
    report["generated_at"] = datetime.now(timezone.utc).isoformat()
    report["eval_file"] = str(args.eval_file)
    report["model_quality_note"] = (
        "This eval set is synthetic but product-shaped; use it for regression detection and error analysis, "
        "not as a clinical or population-level performance claim."
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(f"cases: {report['total_cases']}")
    print(f"primary_accuracy: {report['primary_accuracy']}")
    print(f"accepted_accuracy: {report['accepted_accuracy']}")
    print(f"non_crisis_primary_accuracy: {report['non_crisis_primary_accuracy']}")
    print(f"non_crisis_accepted_accuracy: {report['non_crisis_accepted_accuracy']}")
    print(f"top3_primary_recall: {report['top3_primary_recall']}")
    print(f"mixed_signal_rate: {report['mixed_signal_rate']}")
    print(f"report: {args.output}")

    if (
        args.fail_under_accepted is not None
        and report["accepted_accuracy"] is not None
        and report["accepted_accuracy"] < args.fail_under_accepted
    ):
        print(
            f"accepted_accuracy below threshold: {report['accepted_accuracy']} < {args.fail_under_accepted}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
