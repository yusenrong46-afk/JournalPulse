"""Fictional artifact-integrity examples; no model calls or human review."""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
spec = importlib.util.spec_from_file_location(
    "prepare_quality_review", Path(__file__).parents[1] / "scripts/prepare_quality_review.py")
assert spec and spec.loader
review_export = importlib.util.module_from_spec(spec)
spec.loader.exec_module(review_export)
import quality_eval as q  # noqa: E402


@pytest.fixture
def completed_fictional_bundle(tmp_path):
    bundle = tmp_path / "bundle"
    q.prepare(bundle)
    rows = []
    for packet in q.checked_packets(bundle):
        support = packet["safety"]["mode"] == "support"
        rows.append({"id": packet["case"]["id"],
                     "status": "deterministic_support" if support else "judged",
                     "response": packet["safety"]["support_message"] if support else
                     {"reply": "Exact fictional reply.", "reflection": {"summary": "Fictional context"}},
                     "grade": None if support else {"critical_safety": "pass", "relevance": 4}})
    (bundle / "results.json").write_text(json.dumps(rows))
    (bundle / "summary.json").write_text('{"human_review":"NOT_RUN"}')
    (bundle / "REPORT.md").write_text("Fictional test fixture, not a model report")
    return bundle


def test_locked_blind_context_omits_grades_and_preserves_exact_response(completed_fictional_bundle, tmp_path):
    bundle = completed_fictional_bundle
    output = tmp_path / "review"
    before = {p.name: p.read_bytes() for p in bundle.glob('*') if p.is_file()}
    receipt = review_export.export_review(bundle, output)
    blind = json.loads((output / "BLIND_CASES.json").read_text())
    selection = json.loads((bundle / "review-selection.json").read_text())
    assert [c["id"] for c in blind["cases"]] == selection["development"] + selection["holdout"]
    assert len(blind["cases"]) == 20
    assert all(c["response"]["reply"] == "Exact fictional reply." for c in blind["cases"])
    assert all("activity_context" in c and "case" in c for c in blind["cases"])
    assert all("grade" not in c and "model" not in c and "provider" not in c for c in blind["cases"])
    text = (output / "BLIND_CASES.json").read_text()
    assert q.CANDIDATE not in text and q.JUDGE not in text
    assert receipt["additional_review_cases"] == 14 and receipt["human_review"] == "NOT_RUN"
    template = json.loads((output / "HUMAN_REVIEW_TEMPLATE.json").read_text())
    assert template["source_manifest_sha256"] == review_export.sha(bundle / "manifest.json")
    assert template["model_results_sha256"] == review_export.sha(bundle / "results.json")
    assert all(c["grade"] is None for c in template["sampled"])
    assert before == {p.name: p.read_bytes() for p in bundle.glob('*') if p.is_file()}


def test_unavailable_sample_stays_in_denominator_and_additional_review(completed_fictional_bundle, tmp_path):
    bundle = completed_fictional_bundle
    rows = json.loads((bundle / "results.json").read_text())
    selected = json.loads((bundle / "review-selection.json").read_text())["holdout"][0]
    row = next(r for r in rows if r["id"] == selected)
    row.update(status="candidate_invalid_or_refused", response=None, grade=None)
    (bundle / "results.json").write_text(json.dumps(rows))
    output = tmp_path / "review"
    review_export.export_review(bundle, output)
    blind = json.loads((output / "BLIND_CASES.json").read_text())
    assert len(blind["cases"]) == 20
    assert next(c for c in blind["cases"] if c["id"] == selected)["response"] is None
    queue = json.loads((output / "ADJUDICATION_AFTER_BLIND.json").read_text())
    assert selected in {c["id"] for c in queue["cases"]}
    assert queue["open_only_after_blind_grades_saved"] is True


def test_incomplete_reports_create_no_review_packet(completed_fictional_bundle, tmp_path):
    bundle = completed_fictional_bundle
    (bundle / "REPORT.md").unlink()
    with pytest.raises(ValueError, match="incomplete"):
        review_export.export_review(bundle, tmp_path / "review")
    assert not (tmp_path / "review").exists()


def test_export_refuses_overwrite_or_any_in_bundle_destination(completed_fictional_bundle, tmp_path):
    bundle = completed_fictional_bundle
    output = tmp_path / "review"
    review_export.export_review(bundle, output)
    before = (output / "BLIND_CASES.json").read_bytes()
    with pytest.raises(FileExistsError):
        review_export.export_review(bundle, output)
    assert (output / "BLIND_CASES.json").read_bytes() == before
    with pytest.raises(ValueError, match="outside"):
        review_export.export_review(bundle, bundle / "review")
