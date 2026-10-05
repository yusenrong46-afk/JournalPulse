"""Validate the offline evaluation harness, not actual model response quality."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from journalpulse.evaluation import EvaluationDataset, build_bundle, prepare_case, review_output

DATASET = Path(__file__).resolve().parents[1] / "assets/evaluation/luna-v1.json"


def dataset() -> EvaluationDataset:
    return EvaluationDataset.model_validate_json(DATASET.read_text())


def output(**changed):
    return {
        "reply": "Of course. We can leave it here.", "offer_action": False,
        "resource_intent": "reflect", "card_reason": "", "summary": "The person asked to pause.",
        "feelings": [], **changed,
    }


def test_unobserved_cases_do_not_claim_quality_or_provider_calls():
    bundle = build_bundle(dataset())
    assert len(bundle["cases"]) == 14
    assert bundle["runner_provider_calls"] == 0
    assert all(case["structural_gate"] == "unobserved" for case in bundle["cases"])
    assert all(case["model_quality"] == "unreviewed" for case in bundle["cases"])


def test_structural_pass_leaves_human_stopping_check_pending():
    case = next(case for case in dataset().cases if case.id == "stop_without_question")
    # A valid schema can still contain an unwanted question: never score it as quality pass.
    result = review_output(case, output(reply="What else do you want to explore?"))
    assert result["structural_gate"] == "pass"
    assert result["model_quality"] == "unreviewed"
    assert all(item["status"] == "pending" for item in result["human_review"])
    unexpected_offer = review_output(case, output(offer_action=True, card_reason="A pause."))
    assert unexpected_offer["behavior_flag_gate"] == "fail"


def test_selection_gate_rejects_sources_absent_from_controlled_retrieval():
    case = next(case for case in dataset().cases if case.id == "search_snippet_injection")
    result = review_output(case, {"selections": [{"candidate_id": 5, "why_selected": "Its snippet fits."}]})
    assert result["structural_gate"] == "fail"


def test_context_and_embedded_instructions_stay_user_data():
    case = next(case for case in dataset().cases if case.id == "journal_embedded_instructions")
    prepared = prepare_case(case)
    messages = prepared["request"]["messages"]
    assert messages[-1]["role"] == "user"
    assert json.loads(messages[-1]["content"]) == {"journal_text": case.source_text}
    assert all(
        case.source_text not in message["content"]
        for message in messages if message["role"] == "system"
    )


def test_unknown_or_duplicate_observations_are_rejected():
    with pytest.raises(ValueError, match="known case"):
        build_bundle(dataset(), {"cases": [{"id": "untracked", "output": output()}]})
    observed = {"id": "stop_without_question", "output": output()}
    with pytest.raises(ValueError, match="Duplicate"):
        build_bundle(dataset(), {"cases": [observed, observed]})


def test_dataset_ids_and_last_user_turn_are_validated():
    raw = dataset().model_dump()
    raw["cases"].append(raw["cases"][0])
    with pytest.raises(ValidationError, match="unique"):
        EvaluationDataset.model_validate(raw)
    raw = dataset().model_dump()
    raw["cases"][0]["messages"][-1]["role"] = "assistant"
    with pytest.raises(ValidationError, match="latest user"):
        EvaluationDataset.model_validate(raw)


def test_every_workflow_prepares_the_runtime_privacy_routing_contract():
    for case in dataset().cases:
        request = prepare_case(case)["request"]
        assert request["provider"] == {"zdr": True, "require_parameters": True}
