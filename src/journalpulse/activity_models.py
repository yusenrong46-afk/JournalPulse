"""Owner-bound activity contracts, separate from legacy close-and-save choices.

A session records controls and a person's report, never a journal excerpt. Timer
expiry requests a check-in; only the participant can describe what they tried.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .discovery_models import DiscoveryProvenance, checked_source_url
from .domain import Goal, ModelRun


class ActivityStatus(StrEnum):
    OFFERED = "offered"
    ACTIVE = "active"
    PAUSED = "paused"
    AWAITING_REPORT = "awaiting_report"
    COMPLETED = "completed"
    STOPPED = "stopped"
    DECLINED = "declined"


NONTERMINAL_ACTIVITY_STATES = frozenset(
    {
        ActivityStatus.OFFERED,
        ActivityStatus.ACTIVE,
        ActivityStatus.PAUSED,
        ActivityStatus.AWAITING_REPORT,
    }
)
MAX_ACTIVITY_RECEIPTS = 64
MAX_FOLLOW_UP_ATTEMPTS = 3
# Exceeds the 100s provider budget and 120s host limit, with recovery/commit headroom.
FOLLOW_UP_LEASE_SECONDS = 180


class ActivityResource(BaseModel):
    """A server-approved descriptor. Search provenance never implies page review."""

    model_config = ConfigDict(extra="forbid")
    id: str = Field(min_length=1, max_length=120)
    title: str = Field(min_length=1, max_length=200)
    url: str | None = Field(default=None, max_length=2048)
    provider: str = Field(default="JournalPulse", min_length=1, max_length=120)
    resource_type: str = Field(default="activity", min_length=1, max_length=40)
    format: Literal["timer", "external", "manual"]
    kind: Literal["meditation", "movement", "reflection", "connection", "focus", "video", "reading", "other"]
    duration_seconds: int | None = Field(default=None, ge=1, le=3600)
    instructions: list[str] = Field(default_factory=list, max_length=8)
    provenance: Literal["builtin", "catalog", "search_snippet"]
    discovery_provenance: DiscoveryProvenance | None = None

    @field_validator("url")
    @classmethod
    def public_url(cls, value: str | None) -> str | None:
        return checked_source_url(value) if value is not None else None

    @field_validator("instructions")
    @classmethod
    def bounded_instructions(cls, values: list[str]) -> list[str]:
        if any(not value.strip() or len(value) > 240 for value in values):
            raise ValueError("Instructions must be nonblank and at most 240 characters each")
        return values

    @model_validator(mode="after")
    def coherent_format(self) -> ActivityResource:
        if self.format == "timer" and self.duration_seconds is None:
            raise ValueError("A timer needs a reviewed duration")
        if self.format == "external" and self.url is None:
            raise ValueError("An external activity needs a public resource link")
        return self


class ActivitySelectionProvenance(BaseModel):
    """These are LLM/user choices, not randomized research-policy assignments."""

    model_config = ConfigDict(extra="forbid")
    selection_source: Literal["llm", "guided", "user", "search"]
    recommended_resource_id: str = Field(min_length=1, max_length=120)
    selected_resource_id: str = Field(min_length=1, max_length=120)
    eligible_for_ope: Literal[False] = False
    propensity: None = None
    model_run: ModelRun | None = None


class ActivityReport(BaseModel):
    model_config = ConfigDict(extra="forbid")
    participation: Literal["completed", "partial", "not_tried", "stopped"]
    fit: Literal["good", "mixed", "poor", "unsure"] | None = None
    state_change: Literal["toward_target", "same", "away_from_target", "unsure"] | None = None
    goal_progress: Literal["closer", "same", "further", "unsure"] | None = None
    before_rating: int | None = Field(default=None, ge=1, le=5)
    after_rating: int | None = Field(default=None, ge=1, le=5)
    helpfulness: int | None = Field(default=None, ge=1, le=5)
    effort: int | None = Field(default=None, ge=1, le=5)
    note: str | None = Field(default=None, max_length=1000)


class ActivitySession(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: UUID = Field(default_factory=uuid4)
    user_id: UUID
    conversation_id: UUID
    source_entry_id: UUID | None = None
    offered_message_id: UUID | None = None
    revision: int = Field(default=0, ge=0)
    status: ActivityStatus = ActivityStatus.OFFERED
    resource: ActivityResource
    goal: Goal | None = None
    recommendation_reason: str | None = Field(default=None, max_length=240)
    selection: ActivitySelectionProvenance
    duration_seconds: int = Field(default=0, ge=0, le=3600)
    remaining_seconds: int = Field(default=0, ge=0, le=3600)
    expires_at: datetime | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    started_at: datetime | None = None
    check_in_issued: bool = False
    report: ActivityReport | None = None
    reported_at: datetime | None = None
    follow_up_status: Literal["none", "pending", "generating", "ready", "failed"] = "none"
    follow_up_reply: str | None = Field(default=None, max_length=1200)
    follow_up_model_run: ModelRun | None = None
    follow_up_message_id: UUID | None = None
    follow_up_request_id: UUID | None = None
    follow_up_lease_until: datetime | None = None
    follow_up_attempts: int = Field(default=0, ge=0, le=MAX_FOLLOW_UP_ATTEMPTS)
    final_follow_up: bool = False
    # Response clock/revision are fresh observations, not duplicated persisted facts.
    server_now: datetime | None = None
    conversation_revision: int | None = Field(default=None, ge=0)

    def storage_payload(self) -> dict:
        return self.model_dump(
            mode="json", exclude={"server_now", "conversation_revision", "follow_up_reply"}
        )


class ActivityHistoryItem(BaseModel):
    """One reported chat activity for the garden. It carries the person's own report
    choices only: no note, link, instructions or model text, and no claim of benefit."""

    model_config = ConfigDict(extra="forbid")
    id: UUID
    conversation_id: UUID
    title: str
    kind: str
    goal: Goal | None
    participation: Literal["completed", "partial", "not_tried", "stopped"]
    fit: Literal["good", "mixed", "poor", "unsure"] | None
    state_change: Literal["toward_target", "same", "away_from_target", "unsure"] | None
    helpfulness: int | None
    reported_at: datetime

    @classmethod
    def from_session(cls, session: ActivitySession) -> ActivityHistoryItem:
        if session.report is None or session.reported_at is None:
            raise ValueError("Only a reported activity belongs in the history")
        return cls(
            id=session.id,
            conversation_id=session.conversation_id,
            title=session.resource.title,
            kind=session.resource.kind,
            goal=session.goal,
            participation=session.report.participation,
            fit=session.report.fit,
            state_change=session.report.state_change,
            helpfulness=session.report.helpfulness,
            reported_at=session.reported_at,
        )


class ActivityHistoryPage(BaseModel):
    items: list[ActivityHistoryItem]


class CreateActivitySessionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    client_request_id: UUID
    expected_conversation_revision: int = Field(ge=0)
    resource_id: str = Field(min_length=1, max_length=120)
    duration_seconds: int | None = Field(default=None, ge=1, le=3600)
    resource_token: str | None = Field(default=None, min_length=1, max_length=12000)


class ActivityCommandRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    client_request_id: UUID
    expected_revision: int = Field(ge=0)
    expected_conversation_revision: int = Field(ge=0)
    command: Literal["start", "pause", "resume", "finish_early", "expire", "stop", "decline"]


class ActivityReportRequest(ActivityReport):
    client_request_id: UUID
    expected_revision: int = Field(ge=0)
    expected_conversation_revision: int = Field(ge=0)

    def participant_report(self) -> ActivityReport:
        return ActivityReport.model_validate(
            self.model_dump(
                exclude={
                    "client_request_id",
                    "expected_revision",
                    "expected_conversation_revision",
                }
            )
        )


class ActivityFollowUpRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    client_request_id: UUID
    expected_revision: int = Field(ge=0)
    expected_conversation_revision: int = Field(ge=0)
