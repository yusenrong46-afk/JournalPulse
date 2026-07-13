from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator


class SafetyMode(StrEnum):
    NORMAL = "normal"
    SUPPORT = "support"


class SelectionSource(StrEnum):
    POLICY = "policy"
    POLICY_ACCEPTED = "policy_accepted"
    USER_OVERRIDE = "user_override"


class AffectiveState(BaseModel):
    valence: float = Field(ge=-1.0, le=1.0)
    arousal: float = Field(ge=0.0, le=1.0)
    agency: float = Field(ge=0.0, le=1.0)
    emotion_tags: list[str] = Field(default_factory=list, max_length=6)
    confidence: float = Field(ge=0.0, le=1.0)
    uncertainty: str | None = Field(default=None, max_length=240)


class TargetState(BaseModel):
    valence: float | None = Field(default=None, ge=-1.0, le=1.0)
    arousal: float | None = Field(default=None, ge=0.0, le=1.0)
    agency: float | None = Field(default=None, ge=0.0, le=1.0)
    goal: str = Field(min_length=1, max_length=80)


class SafetyResult(BaseModel):
    mode: SafetyMode
    reasons: list[str] = Field(default_factory=list)
    locale: str
    exploration_allowed: bool
    support_message: str | None = None
    resource_ids: list[str] = Field(default_factory=list)


class PolicyDecision(BaseModel):
    decision_id: UUID = Field(default_factory=uuid4)
    action_id: str
    propensity: float = Field(gt=0.0, le=1.0)
    policy_name: str
    policy_version: str
    safe_action_ids: list[str]
    context_snapshot: dict[str, Any]
    explanation: str
    recommended_action_id: str | None = None
    selection_source: SelectionSource = SelectionSource.POLICY
    eligible_for_ope: bool = True


class ReflectionCopy(BaseModel):
    summary: str = Field(min_length=1, max_length=420)
    interpretation: str = Field(min_length=1, max_length=420)
    reflection_question: str = Field(min_length=1, max_length=240)


class ModelRun(BaseModel):
    model: str
    provider: str = "openrouter"
    latency_ms: int = Field(ge=0)
    prompt_tokens: int | None = Field(default=None, ge=0)
    completion_tokens: int | None = Field(default=None, ge=0)
    schema_valid: bool
    used_fallback: bool = False
    fallback_reason: str | None = None


class ReflectionRecord(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    user_id: UUID
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    text: str | None = None
    text_retained: bool = False
    context: dict[str, str] = Field(default_factory=dict)
    state: AffectiveState
    target: TargetState
    reflection: ReflectionCopy
    safety: SafetyResult
    decision: PolicyDecision
    model_run: ModelRun | None = None


class PreparedAnalysis(BaseModel):
    state: AffectiveState
    reflection: ReflectionCopy
    safety: SafetyResult
    model_run: ModelRun
    resource_intent: str = Field(min_length=1, max_length=40)


class ActionPreviewRequest(BaseModel):
    state: AffectiveState
    target: TargetState
    context: dict[str, str] = Field(default_factory=dict)
    resource_intent: str = Field(default="reflect", min_length=1, max_length=40)


class ActionPreview(BaseModel):
    decision: PolicyDecision
    actions: list[dict[str, Any]]


class AnalysisRequest(BaseModel):
    text: str = Field(min_length=1, max_length=5000)
    context: dict[str, str] = Field(default_factory=dict)
    llm_consent: bool = False
    locale: str = Field(default="CA", min_length=2, max_length=8)

    @field_validator("text")
    @classmethod
    def strip_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("text must not be blank")
        return value


class OutcomeRecord(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    user_id: UUID
    decision_id: UUID
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    completed: bool
    post_state: AffectiveState | None = None
    helpfulness: int | None = Field(default=None, ge=1, le=5)
    effort: int | None = Field(default=None, ge=1, le=5)
    elapsed_minutes: int | None = Field(default=None, ge=0, le=10080)
    note: str | None = Field(default=None, max_length=1000)


class ReflectionRequest(BaseModel):
    client_request_id: UUID | None = None
    text: str = Field(min_length=1, max_length=5000)
    context: dict[str, str] = Field(default_factory=dict)
    self_report: AffectiveState | None = None
    target: TargetState
    llm_consent: bool = False
    retain_text: bool | None = None
    locale: str = Field(default="CA", min_length=2, max_length=8)
    prepared_analysis: PreparedAnalysis | None = None
    chosen_action_id: str | None = Field(default=None, min_length=1, max_length=120)

    @field_validator("text")
    @classmethod
    def strip_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("text must not be blank")
        return value


class OutcomeRequest(BaseModel):
    client_request_id: UUID | None = None
    decision_id: UUID
    completed: bool
    post_state: AffectiveState | None = None
    helpfulness: int | None = Field(default=None, ge=1, le=5)
    effort: int | None = Field(default=None, ge=1, le=5)
    elapsed_minutes: int | None = Field(default=None, ge=0, le=10080)
    note: str | None = Field(default=None, max_length=1000)
