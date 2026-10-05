"""Saved writing is independent of AI consent and action selection."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .domain import ModelRun, SafetyResult


def require_writing(text: str) -> str:
    """Reject whitespace-only writing while preserving the person's exact wording."""
    if not text.strip():
        raise ValueError("Journal text must not be blank")
    return text


class JournalEntry(BaseModel):
    """An immutable saved entry; AI output is never folded into its source text."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    id: UUID = Field(default_factory=uuid4)
    user_id: UUID
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    text: str = Field(min_length=1, max_length=5000)

    @field_validator("text")
    @classmethod
    def nonblank_text(cls, value: str) -> str:
        return require_writing(value)


class CreateJournalEntryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str = Field(min_length=1, max_length=5000)
    client_request_id: UUID | None = None

    @field_validator("text")
    @classmethod
    def nonblank_text(cls, value: str) -> str:
        return require_writing(value)


class JournalEntryPage(BaseModel):
    items: list[JournalEntry]
    limit: int
    offset: int


class ReflectJournalEntryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    llm_consent: bool
    locale: str = Field(default="CA", min_length=2, max_length=8)


class JournalReflectionResult(BaseModel):
    """A transient reflection. Deleting or exporting the source has no AI row to retain."""

    entry_id: UUID
    reply: str = Field(min_length=1, max_length=1200)
    model_run: ModelRun
    safety: SafetyResult
    generated_text_retained: Literal[False] = False

    @field_validator("reply")
    @classmethod
    def nonblank_reply(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Reflection reply must not be blank")
        return value
