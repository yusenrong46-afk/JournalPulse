"""Prepare fictional prompts or review supplied outputs, without provider calls.

This deliberately reads neither environment credentials nor user databases. Run
actual models separately with a bounded request budget and supply their outputs
and provenance. Structural gates do not score usefulness or model quality.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from journalpulse.discovery import REFINEMENT_SCHEMA, SELECTION_SCHEMA, RefinementOutput, SelectionOutput
from journalpulse.discovery_prompts import DISCOVERY_PROMPT_VERSION, DISCOVERY_SYSTEM_PROMPT
from journalpulse.intelligence import (
    CONVERSATION_JSON_SCHEMA,
    CONVERSATION_PROMPT_VERSION,
    CONVERSATION_SYSTEM_PROMPT,
    ConversationTurnOutput,
)
from journalpulse.reflection_prompts import (
    JOURNAL_CONTEXT_INSTRUCTION,
    JOURNAL_REFLECTION_INSTRUCTION,
    LISTEN_CONTEXT_INSTRUCTION,
    REFLECTION_SKILL_VERSION,
    journal_data_message,
)

DEFAULT_DATASET = Path(__file__).resolve().parents[2] / "assets/evaluation/luna-v1.json"


class EvaluationMessage(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=6000)


class EvaluationCase(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    id: str = Field(min_length=1, max_length=80)
    workflow: Literal["chat", "journal", "discovery_selection", "discovery_refinement"]
    messages: list[EvaluationMessage] = Field(default_factory=list, max_length=20)
    source_text: str | None = Field(default=None, min_length=1, max_length=6000)
    interaction_preference: Literal["auto", "listen"] = "auto"
    input: dict[str, Any] | None = None
    expected_offer_action: bool | None = None
    expected_selected_ids: list[int] | None = None
    review_criteria: list[str] = Field(min_length=1, max_length=8)

    @model_validator(mode="after")
    def has_workflow_input(self) -> EvaluationCase:
        if self.workflow == "chat" and (
            not self.messages or self.messages[-1].role != "user"
        ):
            raise ValueError("Chat cases need a latest user message")
        if self.workflow == "journal" and not self.source_text:
            raise ValueError("Journal cases need fictional source writing")
        if self.workflow.startswith("discovery_") and not self.input:
            raise ValueError("Discovery cases need controlled topic and snippet data")
        return self


class EvaluationDataset(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    dataset_version: str
    data_origin: str
    cases: list[EvaluationCase] = Field(min_length=1, max_length=30)

    @model_validator(mode="after")
    def unique_ids(self) -> EvaluationDataset:
        ids = [case.id for case in self.cases]
        if len(ids) != len(set(ids)):
            raise ValueError("Evaluation case IDs must be unique")
        return self


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def prepare_case(case: EvaluationCase) -> dict[str, Any]:
    """Use the same trusted instructions and untrusted roles as the server."""
    if case.workflow.startswith("discovery_"):
        schema = SELECTION_SCHEMA if case.workflow == "discovery_selection" else REFINEMENT_SCHEMA
        messages = [
            {"role": "system", "content": DISCOVERY_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(case.input)},
        ]
        prompt_version = DISCOVERY_PROMPT_VERSION
    else:
        schema = CONVERSATION_JSON_SCHEMA
        messages = [{"role": "system", "content": CONVERSATION_SYSTEM_PROMPT}]
        prompt_version = CONVERSATION_PROMPT_VERSION
        if case.workflow == "journal":
            messages.extend([
                {"role": "system", "content": JOURNAL_REFLECTION_INSTRUCTION},
                journal_data_message(case.source_text or ""),
            ])
        else:
            if case.interaction_preference == "listen":
                messages.append({"role": "system", "content": LISTEN_CONTEXT_INSTRUCTION})
            if case.source_text:
                messages.extend([
                    {"role": "system", "content": JOURNAL_CONTEXT_INSTRUCTION},
                    {"role": "user", "content": (
                        "Selected journal entry 00000000-0000-4000-8000-000000000099, "
                        "saved 2026-10-04T12:00:00+00:00.\n"
                        f"Journal context (user data):\n{case.source_text}"
                    )},
                ])
            messages.extend(message.model_dump() for message in case.messages)
        if case.source_text:
            prompt_version += f"+{REFLECTION_SKILL_VERSION}"
    # Discovery uses the same retention and schema-capability routing as chat.
    # Keep prepared study requests consistent with the runtime's provider contract.
    request = {
        "messages": messages,
        "response_format": {"type": "json_schema", "json_schema": schema},
        "provider": {"zdr": True, "require_parameters": True},
    }
    return {
        "id": case.id,
        "workflow": case.workflow,
        "prompt_version": prompt_version,
        "request_sha256": digest(request),
        "request": request,
        "expected_offer_action": case.expected_offer_action,
        "expected_selected_ids": case.expected_selected_ids,
        "human_review": [{"criterion": text, "status": "pending"} for text in case.review_criteria],
    }


def review_output(case: EvaluationCase, output: Any) -> dict[str, Any]:
    """Check software contracts; leave semantic judgments explicitly unreviewed."""
    errors: list[str] = []
    behavior_errors: list[str] = []
    extra: dict[str, Any] = {}
    try:
        if case.workflow in {"chat", "journal"}:
            parsed = ConversationTurnOutput.model_validate(output).require_card_reason()
            if case.expected_offer_action is not None and parsed.offer_action != case.expected_offer_action:
                behavior_errors.append("offer_action differs from the expected current user intent")
        elif case.workflow == "discovery_selection":
            selected = SelectionOutput.model_validate(output).selections
            ids = [item.candidate_id for item in selected]
            candidates = (case.input or {}).get("candidates", [])
            supplied_ids = {item["candidate_id"] for item in candidates}
            if len(ids) != len(set(ids)) or not set(ids).issubset(supplied_ids):
                errors.append("Selection includes a duplicate or a candidate not supplied")
            if case.expected_selected_ids is not None and ids != case.expected_selected_ids:
                behavior_errors.append("Selected IDs differ from the fixture's relevant sources")
        else:
            focus = RefinementOutput.model_validate(output).additional_terms
            original = (case.input or {})["original_goal"]
            extra["constructed_search_query"] = f"{original} {focus}".strip()
    except (KeyError, TypeError, ValueError, ValidationError) as exc:
        errors.append(f"Output rejected by {type(exc).__name__}; review the supplied response")
    return {
        "structural_gate": "fail" if errors else "pass",
        "behavior_flag_gate": "fail" if errors or behavior_errors else "pass",
        "errors": errors + behavior_errors,
        "model_quality": "unreviewed",
        "human_review": [{"criterion": text, "status": "pending"} for text in case.review_criteria],
        **extra,
    }


def build_bundle(dataset: EvaluationDataset, responses: dict[str, Any] | None = None) -> dict[str, Any]:
    supplied = (responses or {}).get("cases", [])
    if not isinstance(supplied, list):
        raise ValueError("Responses need a cases list")
    by_id: dict[str, dict[str, Any]] = {}
    known = {case.id for case in dataset.cases}
    for item in supplied:
        if not isinstance(item, dict) or item.get("id") not in known or "output" not in item:
            raise ValueError("Each supplied response needs a known case ID and output")
        if item["id"] in by_id:
            raise ValueError("Duplicate supplied response ID")
        by_id[item["id"]] = item
    cases = []
    for case in dataset.cases:
        prepared = prepare_case(case)
        observed = by_id.get(case.id)
        if observed is None:
            prepared.update({"structural_gate": "unobserved", "model_quality": "unreviewed"})
        else:
            prepared.update(review_output(case, observed["output"]))
            prepared["observed_output"] = observed["output"]
            prepared["supplied_provenance"] = observed.get("provenance", {})
        cases.append(prepared)
    return {
        "dataset_version": dataset.dataset_version,
        "dataset_sha256": digest(dataset.model_dump()),
        "data_origin": dataset.data_origin,
        "runner_provider_calls": 0,
        "quality_status": "Requires human review; software gates do not establish model quality.",
        "cases": cases,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--responses", type=Path, help="JSON cases with id, output and optional provenance")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    dataset = EvaluationDataset.model_validate_json(args.dataset.read_text())
    responses = json.loads(args.responses.read_text()) if args.responses else None
    bundle = build_bundle(dataset, responses)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(bundle, indent=2) + "\n")
    print(
        f"Prepared {len(dataset.cases)} fictional cases. "
        f"Runner made 0 provider calls. Saved {args.output}."
    )
    return int(any(
        case.get("structural_gate") == "fail" or case.get("behavior_flag_gate") == "fail"
        for case in bundle["cases"]
    ))


if __name__ == "__main__":
    raise SystemExit(main())
