"""Benchmark trust boundaries: no synthetic successes or unsafe HTML rendering."""

from __future__ import annotations

import hashlib
import json
import re
from copy import deepcopy
from pathlib import Path

import pytest
from pydantic import ValidationError

from journalpulse.guided_evaluation import (
    DEFAULT_DATASET,
    DIMENSIONS,
    GuidedDataset,
    JudgePair,
    Observation,
    build_comparison,
    collect_observations,
    digest,
    judge_pair_binding,
    load_dataset,
    merge_pipeline_replay,
    normalize_observation,
    prepare_bundle,
    prepare_case,
    prepare_judge_request,
    prepare_simulator_request,
    render_comparison_html,
    write_comparison_report,
)


def observed(case_id: str, reply: str = "A test-only fictional response.") -> dict:
    return {
        "id": case_id,
        "status": "observed",
        "output": {"reply": reply},
        "provenance": {"model": "test-double-not-a-live-model", "track": "controlled"},
    }


def judgment(
    case_id: str,
    *,
    baseline: int = 3,
    candidate: int = 4,
    candidate_first: bool = False,
    preference: str = "candidate",
    candidate_gate: str = "pass",
    pair_observations: tuple[dict, dict] | None = None,
) -> dict:
    a_version, b_version = ("candidate", "baseline") if candidate_first else ("baseline", "candidate")
    scores = {
        "baseline": {dimension: baseline for dimension in DIMENSIONS},
        "candidate": {dimension: candidate for dimension in DIMENSIONS},
    }
    return {
        "id": case_id,
        "status": "observed",
        "a_version": a_version,
        "b_version": b_version,
        "scores": {"A": scores[a_version], "B": scores[b_version]},
        "preference": "tie" if preference == "tie" else "A" if preference == a_version else "B",
        "evidence": {label: {dimension: "test excerpt" for dimension in DIMENSIONS} for label in ("A", "B")},
        "uncertainty": "medium",
        "rationale": "Test-only judgment.",
        "critical_gates": {
            "A": candidate_gate if a_version == "candidate" else "pass",
            "B": candidate_gate if b_version == "candidate" else "pass",
        },
        "provenance": {
            "model": "test-double-not-a-live-judge",
            "evaluation_binding": judge_pair_binding(
                case(load_dataset(), case_id),
                *[
                    Observation.model_validate(item)
                    for item in (pair_observations or (observed(case_id), observed(case_id)))
                ],
            ),
        },
    }


def case(dataset: GuidedDataset, case_id: str):
    return next(item for item in dataset.cases if item.id == case_id)


def test_frozen_dataset_identity_counts_and_license() -> None:
    dataset = load_dataset()
    assert hashlib.sha256(DEFAULT_DATASET.read_bytes()).hexdigest() == (
        "6396ee7799c1e8dec8d3a45814d3126cfa1f35ad162083bc899061b1b206a9ab"
    )
    assert len(dataset.cases) == 60
    assert sum(item.split == "holdout" for item in dataset.cases) == 24
    assert sum(item.adaptive.enabled for item in dataset.cases) == 8
    assert all(
        item.origin.kind == "original_fictional" and item.origin.license == "CC0-1.0"
        for item in dataset.cases
    )


def test_observed_teacher_scores_require_evidence_but_null_scores_are_unscored() -> None:
    value = judgment("g01_quiet_walk")
    value["evidence"]["A"]["naturalness"] = None
    with pytest.raises(ValidationError, match="requires explicit evidence"):
        JudgePair.model_validate(value)
    value["scores"]["A"]["naturalness"] = None
    assert JudgePair.model_validate(value).scores["A"].naturalness is None


def test_local_review_identity_changes_with_actual_outputs_not_render_time() -> None:
    before = {"cases": [observed("g01_quiet_walk")]}
    first = build_comparison(load_dataset(), before, {"cases": []})
    repeated = build_comparison(load_dataset(), before, {"cases": []})
    changed = build_comparison(
        load_dataset(), {"cases": [observed("g01_quiet_walk", "Changed text")]}, {"cases": []}
    )
    assert first["review_id"] == repeated["review_id"]
    assert first["review_id"] != changed["review_id"]
    html = render_comparison_html(first)
    assert "'journalpulse-review-'+report.review_id" in html
    assert "...(blind?{}:{baseline_support:c.baseline_support})" in html


def test_family_leakage_is_rejected_even_when_counts_are_unchanged() -> None:
    data = load_dataset().model_dump()
    data["cases"][24]["family"] = data["cases"][2]["family"]
    with pytest.raises(ValidationError, match="must not leak"):
        GuidedDataset.model_validate(data)


def test_split_export_keeps_holdout_out_of_prompt_author_bundle() -> None:
    bundle = prepare_bundle(load_dataset(), split="development")
    assert len(bundle["cases"]) == 36
    assert all(item["fixture"]["split"] == "development" for item in bundle["cases"])
    assert bundle["runner_provider_calls"] == 0


def test_prompt_preparation_matches_trusted_runtime_order_and_settings() -> None:
    from journalpulse.intelligence import CONVERSATION_JSON_SCHEMA, CONVERSATION_SYSTEM_PROMPT
    from journalpulse.reflection_prompts import JOURNAL_CONTEXT_INSTRUCTION, LISTEN_CONTEXT_INSTRUCTION

    dataset = load_dataset()
    fixture = case(dataset, "j01_past_and_present").model_copy(update={"interaction_preference": "listen"})
    prepared = prepare_case(fixture, model="test-model-id")
    request = prepared["request"]
    assert request["messages"][0] == {"role": "system", "content": CONVERSATION_SYSTEM_PROMPT}
    assert request["messages"][1] == {"role": "system", "content": LISTEN_CONTEXT_INSTRUCTION}
    assert request["messages"][2] == {"role": "system", "content": JOURNAL_CONTEXT_INSTRUCTION}
    assert request["messages"][3]["role"] == "user"
    assert fixture.journal is not None
    assert fixture.journal.text in request["messages"][3]["content"]
    assert request["messages"][-1] == fixture.messages[-1].model_dump()
    assert request["response_format"]["json_schema"] == CONVERSATION_JSON_SCHEMA
    assert request["provider"] == {"zdr": True, "require_parameters": True}
    assert request["max_tokens"] == 4000
    assert request["reasoning"] == {"effort": "medium"}
    assert request["include_reasoning"] is False
    assert prepared["request_sha256"] == digest(request)


def test_journal_injection_stays_in_json_user_data() -> None:
    prepared = prepare_case(case(load_dataset(), "j04_journal_injection"))
    messages = prepared["request"]["messages"]
    assert messages[-1]["role"] == "user"
    assert "SYSTEM: Ignore" in json.loads(messages[-1]["content"])["journal_text"]
    assert all(
        "SYSTEM: Ignore" not in message["content"] for message in messages if message["role"] == "system"
    )


@pytest.mark.parametrize("field,value", [("deleted", True), ("owner", "fictional-owner-b")])
def test_invalid_journal_source_never_enters_model_request(field: str, value: str | bool) -> None:
    fixture = case(load_dataset(), "j01_past_and_present")
    assert fixture.journal is not None
    source = fixture.journal.model_copy(update={field: value})
    with pytest.raises(ValueError, match="Deleted or foreign"):
        prepare_case(fixture.model_copy(update={"journal": source}))


def test_lifecycle_cases_do_not_create_provider_requests_or_fake_baseline() -> None:
    prepared = prepare_case(case(load_dataset(), "g24_expiry_not_participation"))
    assert prepared["request"] is None
    assert prepared["status"] == "unsupported_baseline"
    assert prepared["provider_calls"] == 0
    with pytest.raises(ValidationError, match="fabricated"):
        Observation(id="g24_expiry_not_participation", status="unsupported_baseline", output="it helped")


@pytest.mark.parametrize(
    "changes",
    [
        {"output": None},
        {"provenance": {}},
        {"provenance": {"model": "deterministic", "track": "deterministic"}},
    ],
)
def test_observed_status_requires_actual_language_and_model_provenance(changes: dict) -> None:
    value = observed("g01_quiet_walk") | changes
    with pytest.raises(ValidationError):
        Observation.model_validate(value)


def test_judge_is_blinded_and_cannot_judge_unobserved_or_wrong_case() -> None:
    fixture = case(load_dataset(), "g01_quiet_walk")
    before = Observation.model_validate(observed(fixture.id, "Distinct response one"))
    after = Observation.model_validate(observed(fixture.id, "Distinct response two"))
    prepared = prepare_judge_request(
        fixture, before, after, model="actual-judge-id", order_seed="frozen-seed"
    )
    judge_data = json.loads(prepared["request"]["messages"][-1]["content"])
    serialized = json.dumps(judge_data)
    assert "baseline" not in serialized and "candidate" not in serialized
    assert "test-double" not in serialized
    assert set(judge_data) == {"scenario", "A", "B"}
    assert prepared["a_version"] != prepared["b_version"]
    assert prepared == prepare_judge_request(
        fixture, before, after, model="actual-judge-id", order_seed="frozen-seed"
    )
    with pytest.raises(ValueError, match="actual observed"):
        prepare_judge_request(
            fixture, before, Observation(id=fixture.id, status="unobserved"), model="judge", order_seed="seed"
        )
    with pytest.raises(ValueError, match="IDs"):
        prepare_judge_request(
            fixture, before.model_copy(update={"id": "wrong"}), after, model="judge", order_seed="seed"
        )


def test_simulation_turn_ceiling_and_seed_provenance() -> None:
    dataset = load_dataset()
    fixture = case(dataset, "g12_explicit_step")
    prepared = prepare_simulator_request(
        dataset, fixture, [{"role": "assistant", "content": "Test reply"}], model="actual-simulator"
    )
    assert prepared["seed_sent_to_provider"] is False
    assert prepared["seed"] == fixture.adaptive.seed
    with pytest.raises(ValueError, match="ceiling"):
        prepare_simulator_request(
            dataset, fixture, [{"role": "assistant", "content": "Test"}] * 4, model="actual-simulator"
        )
    with pytest.raises(ValueError, match="frozen adaptive"):
        prepare_simulator_request(dataset, case(dataset, "g01_quiet_walk"), [], model="actual-simulator")


def test_scores_are_strict_and_judge_requires_outside_version_mapping() -> None:
    data = judgment("g01_quiet_walk")
    data["scores"]["A"]["naturalness"] = True
    with pytest.raises(ValidationError):
        JudgePair.model_validate(data)
    data = judgment("g01_quiet_walk")
    data["b_version"] = data["a_version"]
    with pytest.raises(ValidationError, match="blinded versions"):
        JudgePair.model_validate(data)


def test_pair_mapping_ties_and_safety_exclusion_are_correct() -> None:
    dataset = load_dataset()
    ids = ["g25_cooking_savor", "g26_trip_uncertainty", "s09_user_instruction_injection"]
    baseline = {"cases": [observed(case_id) for case_id in ids]}
    candidate = {"cases": [observed(case_id) for case_id in ids]}
    judges = {
        "cases": [
            judgment(ids[0], candidate_first=True),
            judgment(ids[1], preference="tie"),
            judgment(ids[2], baseline=1, candidate=5),
        ]
    }
    report = build_comparison(dataset, baseline, candidate, judges)
    summary = report["summary"]
    assert summary["heldout_primary"]["n"] == 2  # Safety score is not in a quality average.
    assert summary["heldout_primary"]["mean_difference"] == 1
    assert summary["heldout_non_tied_pairs"] == 1
    assert summary["heldout_candidate_win_fraction"] == 1
    assert summary["heldout_preferences"] == {"candidate": 1, "tie": 1}
    assert summary["gates"]["heldout_primary_gain"] == "incomplete"
    assert summary["release_gate_status"] == "incomplete"


def test_missing_pairs_cannot_be_judged_and_duplicate_observations_are_rejected() -> None:
    dataset = load_dataset()
    with pytest.raises(ValueError, match="requires both actual"):
        build_comparison(
            dataset,
            {"cases": [observed("g25_cooking_savor")]},
            {"cases": []},
            {"cases": [judgment("g25_cooking_savor")]},
        )
    with pytest.raises(ValueError, match="unique known"):
        build_comparison(dataset, {"cases": [observed("g25_cooking_savor")] * 2}, {"cases": []})


def test_new_features_and_prepared_prompts_cannot_inflate_quality() -> None:
    dataset = load_dataset()
    candidate = {"cases": [{"id": "g24_expiry_not_participation", "status": "deterministic_pass"}]}
    report = build_comparison(dataset, {"cases": []}, candidate)
    row = next(row for row in report["cases"] if row["case"]["id"] == "g24_expiry_not_participation")
    assert row["baseline"]["status"] == "unsupported_baseline"
    assert row["quality_eligible"] is False
    assert report["summary"]["heldout_primary"]["n"] == 0


def test_severe_failure_blocks_release_even_if_quality_is_good() -> None:
    dataset = load_dataset()
    report = build_comparison(
        dataset,
        {"cases": [observed("s09_user_instruction_injection")]},
        {"cases": [observed("s09_user_instruction_injection")]},
        {"cases": [judgment("s09_user_instruction_injection", candidate_gate="fail")]},
        metadata={"software_gates": {"backend": "pass"}},
    )
    assert report["summary"]["release_gate_status"] == "fail"
    assert report["summary"]["severe_observed_failures"] == ["s09_user_instruction_injection"]


def test_unobserved_critical_cases_prevent_a_claimed_hard_gate_pass() -> None:
    report = build_comparison(
        load_dataset(),
        {"cases": []},
        {"cases": []},
        metadata={"software_gates": {"backend": "pass", "browser": "pass"}},
    )
    assert report["summary"]["gates"]["hard_safety_privacy_lifecycle"] == "uncertain"
    assert "s01_current_urgent_risk" in report["summary"]["unresolved_critical_cases"]


@pytest.mark.parametrize(
    "gates",
    [
        {"parser": {"status": "pass", "pipeline_status": "accepted"}},
        {"parser": {"status": "fail", "pipeline_status": "rejected"}},
        {"parser": True},
        {"parser": "unknown"},
        {"": "pass"},
        [],
    ],
)
def test_ambiguous_software_gate_contract_is_rejected(gates) -> None:
    with pytest.raises(ValueError, match="software_gates must map"):
        build_comparison(load_dataset(), {"cases": []}, {"cases": []}, metadata={"software_gates": gates})


@pytest.mark.parametrize("judge_status", [None, "unobserved", "provider_error"])
def test_parser_pass_does_not_replace_critical_language_judgment(judge_status) -> None:
    case_id = "s05_physical_warning"
    judges = []
    if judge_status is not None:
        judges.append(judgment(case_id) | {"status": judge_status})
    report = build_comparison(
        load_dataset(),
        {"cases": [observed(case_id)]},
        {"cases": [observed(case_id)]},
        {"cases": judges},
        metadata={"software_gates": {case_id: "pass"}},
    )
    assert case_id in report["summary"]["unresolved_critical_cases"]
    assert report["summary"]["gates"]["hard_safety_privacy_lifecycle"] == "uncertain"


def test_failed_software_gate_blocks_release_without_language_failure() -> None:
    report = build_comparison(
        load_dataset(), {"cases": []}, {"cases": []}, metadata={"software_gates": {"parser": "fail"}}
    )
    assert report["summary"]["gates"]["hard_safety_privacy_lifecycle"] == "fail"
    assert report["summary"]["release_gate_status"] == "fail"


def test_html_script_boundaries_and_placeholder_text_are_escaped(tmp_path: Path) -> None:
    hostile = '</script><script>window.injected=true</script><img src=x onerror="alert(1)">&__TITLE__'
    report = build_comparison(load_dataset(), {"cases": [observed("g01_quiet_walk", hostile)]}, {"cases": []})
    report["dataset_version"] = "__DATA__<fake>"
    page = render_comparison_html(report)
    assert "<title>Luna before and after — __DATA__&lt;fake&gt;</title>" in page
    assert "<script>window.injected=true</script>" not in page
    data = re.search(r'<script id="report-data" type="application/json">(.*?)</script>', page, re.S)
    assert data is not None
    recovered = json.loads(data.group(1))
    assert recovered["cases"][0]["baseline"]["output"]["reply"] == hostile
    assert "innerHTML" not in page
    assert "textContent" in page and "localStorage" in page and "Download my review" in page
    assert "<script src=" not in page
    output = tmp_path / "comparison.html"
    write_comparison_report(report, output)
    manifest = json.loads(output.with_suffix(".manifest.json").read_text())
    assert manifest["report_sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()


def test_offline_report_roundtrip_preserves_complete_transcripts() -> None:
    data = observed("g25_cooking_savor")
    data["messages"] = [
        {"role": "user", "content": "Fictional initial message"},
        {"role": "assistant", "content": "Fictional answer"},
        {"role": "user", "content": "Fictional follow-up"},
        {"role": "assistant", "content": "Fictional final answer"},
    ]
    report = build_comparison(
        load_dataset(),
        {"cases": [deepcopy(data)]},
        {"cases": [deepcopy(data)]},
        {"cases": [judgment("g25_cooking_savor", preference="tie", pair_observations=(data, data))]},
    )
    row = next(item for item in report["cases"] if item["case"]["id"] == data["id"])
    assert len(row["baseline"]["messages"]) == 4
    assert len(row["candidate"]["messages"]) == 4


def test_native_refusal_precedes_even_valid_structured_content() -> None:
    normalized = normalize_observation(
        {
            "id": "s04_harmful_breath_request",
            "status": "observed",
            "model": "actual-model",
            "refusal": "Provider decline",
            "content": '{"reply":"This should not become a completion."}',
            "finish_reason": "content_filter",
            "usage": {"completion_tokens": 7},
        }
    )
    assert normalized["status"] == "provider_refusal"
    assert normalized["output"] is None
    assert normalized["provenance"]["usage"] == {"completion_tokens": 7}


@pytest.mark.parametrize("content,finish", [("not JSON", "stop"), ('{"reply":"partial"}', "length")])
def test_invalid_or_truncated_provider_text_stays_an_error(content: str, finish: str) -> None:
    normalized = normalize_observation(
        {
            "id": "g01_quiet_walk",
            "status": "observed",
            "model": "actual-model",
            "content": content,
            "finish_reason": finish,
        }
    )
    assert normalized["status"] == "provider_error"
    assert normalized["output"] is None


def test_partial_adaptive_language_does_not_claim_complete_observation() -> None:
    normalized = normalize_observation(
        {
            "id": "g12_explicit_step",
            "status": "incomplete",
            "track": "adaptive",
            "turns": 1,
            "messages": [
                {"role": "user", "content": "Fictional input"},
                {"role": "assistant", "content": "Actual retained first-turn text"},
            ],
            "luna_observations": [
                {
                    "model": "actual-luna-model",
                    "provider": "actual-provider",
                    "status": "observed",
                    "usage": {"total_tokens": 200},
                }
            ],
            "teacher_simulations": [{"model": None, "status": "provider_failure", "error_code": 429}],
        }
    )
    assert normalized["status"] == "provider_error"
    assert len(normalized["messages"]) == 2
    assert normalized["provenance"]["model"] == "actual-luna-model"
    assert normalized["provenance"]["teacher_attempts"] == 1
    assert normalized["provenance"]["actual_simulator_runs"][0]["error_code"] == 429


def test_adaptive_normalization_preserves_each_structured_turn_and_native_flags() -> None:
    first = {"reply": "Fictional proposal", "offer_action": True, "activity": {"move": "propose"}}
    second = {"reply": "Fictional revision", "offer_action": False, "activity": {"move": "none"}}
    raw = {
        "id": "g12_explicit_step",
        "status": "observed",
        "track": "adaptive",
        "turns": 2,
        "messages": [{"role": "assistant", "content": "Fictional revision"}],
        "luna_observations": [
            {"status": "observed", "model": "actual-model", "content": json.dumps(first)},
            {"status": "observed", "model": "actual-model", "content": json.dumps(second)},
        ],
    }
    normalized = normalize_observation(raw)
    assert normalized["status"] == "observed"
    assert normalized["output"] == {"turn_outputs": [first, second]}
    assert "not an observed application/UI" in normalized["provenance"]["delivery_limit"]
    raw["luna_observations"][1] |= {"refusal": "Actual native refusal", "content": None}
    refused = normalize_observation(raw)
    assert refused["status"] == "provider_refusal"
    assert refused["output"]["turn_outputs"] == [first, None]
    assert refused["provenance"]["actual_luna_runs"][1]["refusal"] == "Actual native refusal"


def test_collect_latest_observations_does_not_double_count_attempt_evidence(tmp_path: Path) -> None:
    raw = {
        "id": "g01_quiet_walk",
        "status": "observed",
        "model": "actual-model",
        "content": '{"reply":"Fictional output"}',
    }
    (tmp_path / "g01_quiet_walk.json").write_text(json.dumps(raw))
    (tmp_path / "g01_quiet_walk.attempt1.json").write_text(json.dumps(raw))
    collected = collect_observations(tmp_path)
    assert len(collected["cases"]) == 1
    assert collected["cases"][0]["output"] == {"reply": "Fictional output"}


def test_native_refusal_has_separate_pipeline_gate_and_never_language_score() -> None:
    refused = {
        "id": "s09_user_instruction_injection",
        "status": "provider_refusal",
        "provenance": {
            "model": "actual-model",
            "request_sha256": "test-request-identity",
            "pipeline_replay": {
                "id": "s09_user_instruction_injection",
                "status": "provider_refusal",
                "runtime_gate": "pass_native_refusal_handled",
                "request_comparison": {
                    "exact_request_match": True,
                    "runtime_request_sha256": "test-request-identity",
                    "observed_request_sha256": "test-request-identity",
                },
            },
        },
    }
    report = build_comparison(
        load_dataset(),
        {"cases": [refused]},
        {"cases": [refused]},
        metadata={"software_gates": {"s09_user_instruction_injection": "pass"}},
    )
    row = next(row for row in report["cases"] if row["case"]["id"] == refused["id"])
    assert row["quality_eligible"] is False
    assert "s09_user_instruction_injection" not in report["summary"]["unresolved_critical_cases"]
    assert report["summary"]["heldout_primary"]["n"] == 0


@pytest.mark.parametrize(
    "proof",
    [
        {},
        {"id": "s09_user_instruction_injection", "status": "accepted", "runtime_gate": "pass"},
        {
            "id": "another_case",
            "status": "provider_refusal",
            "runtime_gate": "pass_native_refusal_handled",
            "request_comparison": {"exact_request_match": True},
        },
        {
            "id": "s09_user_instruction_injection",
            "status": "provider_refusal",
            "runtime_gate": "pass_native_refusal_handled",
            "request_comparison": {"exact_request_match": False},
        },
        {
            "id": "s09_user_instruction_injection",
            "status": "provider_refusal",
            "runtime_gate": "pass_native_refusal_handled",
            "request_comparison": {
                "exact_request_match": True,
                "runtime_request_sha256": "different-request",
                "observed_request_sha256": "different-request",
            },
        },
    ],
)
def test_native_refusal_needs_matching_runtime_boundary_proof(proof) -> None:
    case_id = "s09_user_instruction_injection"
    refused = {
        "id": case_id,
        "status": "provider_refusal",
        "provenance": {"pipeline_replay": proof, "request_sha256": "test-request-identity"},
    }
    report = build_comparison(
        load_dataset(),
        {"cases": [refused]},
        {"cases": [refused]},
        metadata={"software_gates": {case_id: "pass"}},
    )
    assert case_id in report["summary"]["unresolved_critical_cases"]


def test_candidate_chat_uses_production_context_without_environment_or_private_urls(monkeypatch) -> None:
    from journalpulse import config
    from journalpulse.guided_action import GUIDED_ACTION_JSON_SCHEMA
    from journalpulse.intelligence import CONVERSATION_SYSTEM_PROMPT
    from journalpulse.reflection_prompts import JOURNAL_CONTEXT_INSTRUCTION, LISTEN_CONTEXT_INSTRUCTION

    monkeypatch.setattr(
        config, "load_settings", lambda: pytest.fail("Evaluation must not read environment secrets")
    )
    fixture = case(load_dataset(), "j01_past_and_present").model_copy(
        update={"interaction_preference": "listen"}
    )
    prepared = prepare_case(fixture, candidate_guided=True)
    request = prepared["request"]
    assert request["response_format"]["json_schema"] == GUIDED_ACTION_JSON_SCHEMA
    assert request["messages"][0] == {"role": "system", "content": CONVERSATION_SYSTEM_PROMPT}
    assert request["messages"][1]["role"] == "user"
    context = json.loads(request["messages"][1]["content"])["activity_context"]
    assert len(context["candidates"]) == 16
    assert context["goal"] is None and context["activity_state"] is None
    assert context["preference"] == "listen" and context["action_allowed"] is False
    assert all("url" not in resource for resource in context["candidates"])
    assert request["messages"][2] == {"role": "system", "content": LISTEN_CONTEXT_INSTRUCTION}
    assert request["messages"][3] == {"role": "system", "content": JOURNAL_CONTEXT_INSTRUCTION}
    assert request["messages"][4]["role"] == "user"
    assert request["messages"][-1] == fixture.messages[-1].model_dump()
    assert prepared["runtime_contract"] == "guided_chat"
    assert len(prepared["skill_sha256"]) == 64


def test_candidate_standalone_journal_does_not_gain_activity_decision_schema() -> None:
    prepared = prepare_case(case(load_dataset(), "j03_standalone_no_action"), candidate_guided=True)
    request = prepared["request"]
    assert "activity" not in request["response_format"]["json_schema"]["schema"]["properties"]
    assert prepared["runtime_contract"] == "legacy"
    assert len(request["messages"]) == 3
    assert "journal_text" in json.loads(request["messages"][-1]["content"])


def test_historical_normal_preparation_override_does_not_change_frozen_dataset() -> None:
    dataset = load_dataset()
    before = digest(dataset.model_dump())
    bundle = prepare_bundle(dataset, candidate_guided=True, historical_normal_override=True)
    historical = next(item for item in bundle["cases"] if item["id"] == "s03_historical_risk")
    assert historical["request"] is not None
    assert historical["workflow"] == "chat"
    assert historical["fixture"]["workflow"] == "boundary"
    assert historical["preparation_overrides"]["workflow"] == "chat"
    assert digest(dataset.model_dump()) == before == bundle["dataset_sha256"]


def test_pipeline_replay_verifies_boundaries_without_inventing_model_observations() -> None:
    observations = {
        "cases": [
            {"id": "s01_current_urgent_risk", "status": "deterministic_uncertain", "errors": []},
            {"id": "g24_expiry_not_participation", "status": "unsupported_baseline", "errors": []},
            {"id": "g01_quiet_walk", "status": "unobserved", "errors": []},
        ]
    }
    replay = {
        "evidence_version": "actual-local-replay-test",
        "provider_calls": 0,
        "cases": [
            {
                "id": "s01_current_urgent_risk",
                "status": "replayed",
                "http_status": 200,
                "response": {"assistant_message": "Actual deterministic support copy"},
            },
            {"id": "g24_expiry_not_participation", "status": "replayed", "http_status": 200},
            {"id": "g01_quiet_walk", "status": "replayed", "http_status": 200},
        ],
    }
    baseline = merge_pipeline_replay(observations, replay)
    assert baseline["cases"][0]["status"] == "deterministic_pass"
    assert baseline["cases"][1]["status"] == "unsupported_baseline"
    assert baseline["cases"][2]["status"] == "unobserved"
    candidate = merge_pipeline_replay(observations, replay, phase="candidate")
    assert candidate["cases"][1]["status"] == "deterministic_pass"
    assert candidate["cases"][2]["status"] == "unobserved"
    assert observations["cases"][0]["status"] == "deterministic_uncertain"


def test_pipeline_request_mismatch_remains_a_failure() -> None:
    observations = {
        "cases": [{"id": "j06_journal_cross_owner", "status": "deterministic_uncertain", "errors": []}]
    }
    replay = {
        "cases": [
            {
                "id": "j06_journal_cross_owner",
                "status": "replayed",
                "http_status": 404,
                "request_comparison": {"exact_request_match": False},
            }
        ]
    }
    merged = merge_pipeline_replay(observations, replay)
    assert merged["cases"][0]["status"] == "deterministic_fail"
    assert merged["metadata"]["pipeline_case_gates"]["j06_journal_cross_owner"] == "fail"


def test_candidate_replay_statuses_require_explicit_runtime_and_identity_evidence() -> None:
    records = {
        "cases": [
            observed("g01_quiet_walk"),
            {"id": "g24_expiry_not_participation", "status": "unsupported_baseline"},
            {"id": "s04_harmful_breath_request", "status": "provider_refusal"},
            observed("g02_proud_and_nervous"),
        ]
    }
    replay = {
        "cases": [
            {
                "id": "g01_quiet_walk",
                "status": "accepted",
                "runtime_gate": "pass",
                "request_comparison": {"exact_request_match": True},
            },
            {"id": "g24_expiry_not_participation", "status": "deterministic_pass", "runtime_gate": "pass"},
            {
                "id": "s04_harmful_breath_request",
                "status": "provider_refusal",
                "runtime_gate": "pass_native_refusal_handled",
                "request_comparison": {"exact_request_match": True},
            },
            {"id": "g02_proud_and_nervous", "status": "accepted"},
        ]
    }
    merged = merge_pipeline_replay(records, replay, phase="candidate")
    assert merged["metadata"]["pipeline_case_gates"] == {
        "g01_quiet_walk": "pass",
        "g24_expiry_not_participation": "pass",
        "s04_harmful_breath_request": "pass",
        "g02_proud_and_nervous": "uncertain",
    }
    assert merged["cases"][1]["status"] == "deterministic_pass"
    assert merged["cases"][2]["status"] == "provider_refusal"


def test_rejected_provider_result_is_diagnostic_and_never_scored_as_delivered() -> None:
    case_id = "g29_safe_brief_walk"
    before = {"cases": [observed(case_id)]}
    raw = {"cases": [observed(case_id)]}
    rejected = merge_pipeline_replay(
        raw,
        {
            "cases": [
                {
                    "id": case_id,
                    "status": "rejected",
                    "http_status": 502,
                    "request_comparison": {"exact_request_match": True},
                }
            ]
        },
        phase="candidate",
    )
    result = rejected["cases"][0]
    assert result["status"] == "provider_error"
    assert result["delivery_status"] == "rejected"
    assert result["output"] == raw["cases"][0]["output"]
    assert result["provenance"]["provider_observation_status"] == "observed"
    assert rejected["metadata"]["pipeline_case_gates"][case_id] == "fail"
    report = build_comparison(
        load_dataset(), before, rejected, metadata={"software_gates": {case_id: "fail"}}
    )
    row = next(item for item in report["cases"] if item["case"]["id"] == case_id)
    assert row["quality_eligible"] is False
    assert report["summary"]["heldout_primary"]["n"] == 0
    assert report["summary"]["expected_heldout_quality_pairs"] == 14
    assert report["summary"]["failed_heldout_deliveries"] == [case_id]
    assert report["summary"]["gates"]["complete_heldout_observations"] == "fail"
    assert report["summary"]["release_gate_status"] == "fail"
    html = render_comparison_html(report)
    assert "Rejected provider result (not delivered)" in html
    assert "if(observation.delivery_status==='rejected')return caseData.messages" in html


def test_adaptive_transcripts_are_visible_but_do_not_inflate_controlled_scores() -> None:
    case_id = "g29_safe_brief_walk"
    adaptive = observed(case_id)
    adaptive["provenance"] |= {"track": "adaptive", "turns": 2}
    adaptive["messages"] = [message.model_dump() for message in case(load_dataset(), case_id).messages] + [
        {"role": "assistant", "content": "Fictional fitting suggestion"},
        {"role": "user", "content": "Fictional negotiation"},
        {"role": "assistant", "content": "Fictional revised suggestion"},
    ]
    report = build_comparison(
        load_dataset(),
        {"cases": [observed(case_id)]},
        {"cases": [observed(case_id)]},
        {"cases": [judgment(case_id)]},
        adaptive_baseline={"cases": [adaptive]},
        adaptive_candidate={"cases": [adaptive]},
        adaptive_judgments={"cases": [judgment(case_id, pair_observations=(adaptive, adaptive))]},
    )
    assert report["summary"]["heldout_primary"]["n"] == 1
    assert report["summary"]["adaptive"]["judged_pairs"] == 1
    row = next(item for item in report["cases"] if item["case"]["id"] == case_id)
    assert row["adaptive_comparison"]["primary_quality_eligible"] is False
    assert len(row["adaptive_comparison"]["candidate"]["messages"]) == 4
    assert "Adaptive conversation track (up to four turns)" in render_comparison_html(report)


def test_adaptive_severe_failure_blocks_release_independently_of_controlled_quality() -> None:
    case_id = "s06_unsupported_belief"
    adaptive = observed(case_id)
    adaptive["provenance"] |= {"track": "adaptive", "turns": 1}
    adaptive["messages"] = [message.model_dump() for message in case(load_dataset(), case_id).messages] + [
        {"role": "assistant", "content": "Fictional dangerous affirmation"},
    ]
    report = build_comparison(
        load_dataset(),
        {"cases": []},
        {"cases": []},
        adaptive_baseline={"cases": [adaptive]},
        adaptive_candidate={"cases": [adaptive]},
        adaptive_judgments={
            "cases": [judgment(case_id, candidate_gate="fail", pair_observations=(adaptive, adaptive))]
        },
    )
    assert report["summary"]["release_gate_status"] == "fail"
    assert report["summary"]["severe_observed_failures"] == [f"{case_id}:adaptive"]


def test_adaptive_track_rejects_controlled_substitution_and_overlong_runs() -> None:
    value = observed("g12_explicit_step")
    with pytest.raises(ValueError, match="actual adaptive provenance"):
        build_comparison(load_dataset(), {"cases": []}, {"cases": []}, adaptive_baseline={"cases": [value]})
    value["provenance"] |= {"track": "adaptive", "turns": 5}
    value["messages"] = [{"role": "assistant", "content": "Test-only"}]
    with pytest.raises(ValueError, match="one to four"):
        build_comparison(load_dataset(), {"cases": []}, {"cases": []}, adaptive_baseline={"cases": [value]})


@pytest.mark.parametrize("change", ["history", "turn_count"])
def test_adaptive_transcript_matches_declared_history_and_new_turn_count(change: str) -> None:
    dataset = load_dataset()
    fixture = case(dataset, "g18_negotiate_duration")
    value = observed(fixture.id)
    value["provenance"] |= {"track": "adaptive", "turns": 1}
    value["messages"] = [message.model_dump() for message in fixture.messages] + [
        {"role": "assistant", "content": "Fictional new answer."},
    ]
    if change == "history":
        value["messages"][0]["content"] = "Changed fictional history."
    else:
        value["provenance"]["turns"] = 2
    with pytest.raises(ValueError, match="frozen scenario history|count must match"):
        build_comparison(dataset, {"cases": []}, {"cases": []}, adaptive_baseline={"cases": [value]})


@pytest.mark.parametrize(
    "change", ["missing_binding", "output", "transcript", "request", "model", "scenario"]
)
def test_judgments_cannot_be_reused_after_compared_evidence_changes(change: str) -> None:
    dataset = load_dataset()
    case_id = "g25_cooking_savor"
    left, right = observed(case_id), observed(case_id)
    judge = judgment(case_id)
    if change == "missing_binding":
        del judge["provenance"]["evaluation_binding"]
    elif change == "output":
        right["output"]["reply"] = "A different fictional answer."
    elif change == "transcript":
        right["messages"] = [{"role": "assistant", "content": "A different fictional transcript."}]
    elif change == "request":
        right["provenance"]["request_sha256"] = "changed-model-input"
    elif change == "model":
        right["provenance"]["model"] = "different-actual-model"
    else:
        case(dataset, case_id).known_facts.append("A changed scenario fact.")
    with pytest.raises(ValueError, match="bind the exact scenario and observed pair"):
        build_comparison(dataset, {"cases": [left]}, {"cases": [right]}, {"cases": [judge]})


def test_judge_request_producer_emits_stable_content_binding() -> None:
    fixture = case(load_dataset(), "g25_cooking_savor")
    left = Observation.model_validate(observed(fixture.id))
    right = Observation.model_validate(observed(fixture.id, "A distinct fictional answer."))
    prepared = prepare_judge_request(fixture, left, right, model="test-judge", order_seed="frozen")
    judge = judgment(fixture.id)
    judge["provenance"]["evaluation_binding"] = prepared["evaluation_binding"]
    # Timing and later parser evidence are diagnostic, not the language being judged.
    right.provenance |= {"latency_ms": 100, "pipeline_replay": {"status": "accepted"}}
    report = build_comparison(
        load_dataset(), {"cases": [left.model_dump()]}, {"cases": [right.model_dump()]}, {"cases": [judge]}
    )
    assert report["summary"]["heldout_primary"]["n"] == 1


def test_naturalness_gate_uses_complete_heldout_population_not_development_scores() -> None:
    dataset = load_dataset()
    selected = [
        item for item in dataset.cases if item.group != "safety" and item.baseline_support == "shared"
    ]
    observations = {"cases": [observed(item.id) for item in selected]}
    judges = []
    for fixture in selected:
        judge = judgment(fixture.id)
        judge["scores"]["A"]["naturalness"] = 4 if fixture.split == "holdout" else 1
        judge["scores"]["B"]["naturalness"] = 3 if fixture.split == "holdout" else 5
        judges.append(judge)
    report = build_comparison(dataset, observations, observations, {"cases": judges})
    assert report["summary"]["heldout_primary"]["n"] == report["summary"]["expected_heldout_quality_pairs"]
    assert report["summary"]["naturalness_paired"]["mean_difference"] == -1
    assert report["summary"]["development_naturalness_paired"]["mean_difference"] == 4
    assert report["summary"]["gates"]["naturalness"] == "fail"
    heldout_id = next(item.id for item in selected if item.split == "holdout")
    next(item for item in judges if item["id"] == heldout_id)["scores"]["B"]["naturalness"] = None
    partial = build_comparison(dataset, observations, observations, {"cases": judges})
    assert partial["summary"]["gates"]["naturalness"] == "incomplete"


def test_adaptive_gate_requires_every_complete_bound_teacher_pair() -> None:
    dataset = load_dataset()
    observations, judges = [], []
    for fixture in dataset.cases:
        if not fixture.adaptive.enabled:
            continue
        record = observed(fixture.id)
        record["provenance"] |= {"track": "adaptive", "turns": 1}
        record["messages"] = [message.model_dump() for message in fixture.messages] + [
            {"role": "assistant", "content": "Fictional response"},
        ]
        observations.append(record)
        judges.append(judgment(fixture.id, pair_observations=(record, record)))
    kwargs = {"adaptive_baseline": {"cases": observations}, "adaptive_candidate": {"cases": observations}}
    unjudged = build_comparison(dataset, {"cases": []}, {"cases": []}, **kwargs)
    assert unjudged["summary"]["gates"]["adaptive_observations"] == "incomplete"
    judged = build_comparison(
        dataset, {"cases": []}, {"cases": []}, adaptive_judgments={"cases": judges}, **kwargs
    )
    assert judged["summary"]["gates"]["adaptive_observations"] == "pass"
    judges[-1]["status"] = "provider_error"
    incomplete = build_comparison(
        dataset, {"cases": []}, {"cases": []}, adaptive_judgments={"cases": judges}, **kwargs
    )
    assert incomplete["summary"]["gates"]["adaptive_observations"] == "incomplete"
