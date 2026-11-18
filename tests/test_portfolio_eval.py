"""The offline portfolio evaluation: recomputation, guard and README generation."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from journalpulse.guided_evaluation import _paired_summary

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("portfolio_eval", ROOT / "scripts" / "portfolio_eval.py")
assert spec and spec.loader
portfolio_eval = importlib.util.module_from_spec(spec)
spec.loader.exec_module(portfolio_eval)


def test_interval_matches_the_benchmark_code() -> None:
    values = [1.0, 0.0, -1.0, 2.0, 0.0, 1.0]
    ours = portfolio_eval.paired(values)
    theirs = _paired_summary(values)
    assert ours["mean"] == round(theirs["mean_difference"], 4)
    assert [ours["low"], ours["high"]] == [round(v, 4) for v in theirs["approximate_95_percent_interval"]]


def test_benchmark_recompute_matches_documented_figures_without_a_model() -> None:
    part = portfolio_eval.benchmark_part()
    assert part["recomputed"] == portfolio_eval.DOCUMENTED_BENCHMARK
    assert part["discrepancies"] == [] and part["model_calls"] == 0


def test_safety_part_keeps_the_not_qualified_label() -> None:
    part = portfolio_eval.safety_part()
    assert part["qualification"] == "NOT_QUALIFIED" and part["total_cases"] == 34


def results(**tests: float) -> dict:
    committed = json.loads(portfolio_eval.RESULTS.read_text())
    committed["tests"] = {**committed["tests"], **tests}
    return committed


def test_guard_fails_when_a_guarded_number_regresses() -> None:
    baseline = json.loads(portfolio_eval.BASELINE.read_text())
    assert portfolio_eval.guard(results(), baseline) == []
    problems = portfolio_eval.guard(results(backend_branch_coverage_percent=80.0), baseline)
    assert problems and "backend_branch_coverage_percent" in problems[0]
    worse_safety = results()
    worse_safety["safety"] = {**worse_safety["safety"], "pass": 28, "missed_support": 4}
    assert len(portfolio_eval.guard(worse_safety, baseline)) == 2


def test_readme_block_replacement_is_exact() -> None:
    text = f"before\n{portfolio_eval.BEGIN}\nold\n{portfolio_eval.END}\nafter\n"
    assert portfolio_eval.replace_block(text, "NEW") == "before\nNEW\nafter\n"
