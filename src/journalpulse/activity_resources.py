"""Trusted activity candidates and short-lived, owner-bound discovery receipts.

The model can choose a resource ID, but cannot invent an activity URL or bypass
the person's constraints. Inline search receives a small public vocabulary,
never a projection of private journal text. Existing standalone search is unchanged.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import json
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal, get_args
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .config import Settings
from .discovery_models import (
    DiscoveryCandidate,
    DiscoveryProvenance,
    checked_source_url,
    source_url_identity,
)
from .domain import ActivityConstraintInputs, Goal
from .resources import load_catalog

RESOURCE_TOKEN_TTL_SECONDS = 600
MAX_RESOURCE_TOKEN_BYTES = 12_000
_TOKEN_DOMAIN = b"journalpulse-inline-resource-v1:"
ACTIVITY_GOALS = ("settle", "move", "understand", "connect", "act")
ActivitySearchTopic = Literal[
    "meditation", "grounding", "movement", "walking", "stretching", "reflection",
    "communication", "focus", "reading", "puzzle", "video", "pause",
]
ACTIVITY_SEARCH_TOPICS: tuple[str, ...] = get_args(ActivitySearchTopic)


ActivityConstraints = ActivityConstraintInputs


def protected_activity_constraints(
    current: ActivityConstraints, proposed: ActivityConstraints,
) -> ActivityConstraints:
    """A model may add a restriction, but cannot revoke a saved one."""
    times = [v for v in (current.time_minutes, proposed.time_minutes) if v is not None]
    return ActivityConstraints(
        time_minutes=min(times) if times else None,
        no_audio=current.no_audio or proposed.no_audio,
        no_video=current.no_video or proposed.no_video,
        seated=current.seated or proposed.seated,
        avoid_breath_focus=current.avoid_breath_focus or proposed.avoid_breath_focus,
    )


class ActivityResource(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)

    id: str = Field(min_length=1, max_length=120)
    title: str = Field(min_length=1, max_length=200)
    url: str | None = Field(default=None, max_length=2048)
    summary: str = Field(min_length=1, max_length=800)
    provider: str = Field(min_length=1, max_length=120)
    resource_type: Literal["activity", "video", "website", "game"]
    coping_style: Literal["watch", "move", "read", "play", "connect", "ground", "reflect"]
    kind: Literal["meditation", "movement", "reading", "video", "social", "play", "reflection", "focus"]
    duration_minutes: int | None = Field(default=None, ge=1, le=20)
    duration_seconds: int | None = Field(default=None, ge=30, le=1200)
    timer_enabled: bool = False
    instructions: list[str] = Field(default_factory=list, max_length=8)
    source: Literal["builtin", "catalog", "search_snippet"]
    evidence_kind: Literal["app_guidance", "catalog_review", "search_snippet"]
    # These are format/accessibility metadata, not measurements of suitability.
    no_audio: bool = False
    no_video: bool = False
    seated: bool = False
    breath_focus: bool = False
    goal_tags: list[str] = Field(default_factory=list, max_length=8)
    emotion_tags: list[str] = Field(default_factory=list, max_length=8)
    reviewed_at: str | None = Field(default=None, max_length=80)
    source_tier: str | None = Field(default=None, max_length=80)
    why_selected: str | None = Field(default=None, max_length=420)
    discovery_provenance: DiscoveryProvenance | None = None

    @field_validator("url")
    @classmethod
    def public_source(cls, value: str | None) -> str | None:
        return checked_source_url(value) if value is not None else None

    @field_validator("instructions", "goal_tags", "emotion_tags")
    @classmethod
    def bounded_strings(cls, values: list[str]) -> list[str]:
        if any(not value.strip() or len(value) > 240 for value in values):
            raise ValueError("Resource text must be nonblank and bounded")
        return values

    @model_validator(mode="after")
    def consistent_activity(self) -> ActivityResource:
        if self.timer_enabled and (self.duration_seconds is None or not self.instructions):
            raise ValueError("Timed activities need a duration and local instructions")
        if self.source == "search_snippet":
            if (
                self.url is None
                or self.evidence_kind != "search_snippet"
                or self.timer_enabled
                or self.duration_seconds is not None
                or self.duration_minutes is not None
            ):
                raise ValueError("Search snippets cannot establish a timed activity")
        elif self.source == "catalog" and self.url is None:
            raise ValueError("Catalog resources need a checked source URL")
        return self


def _builtin(
    resource_id: str, title: str, summary: str, kind: str, style: str,
    goal_tags: list[str], instructions: list[str], *, timed: bool = True,
    seated: bool = True, duration_seconds: int = 120,
) -> dict[str, Any]:
    return ActivityResource.model_validate({
        "id": resource_id, "title": title, "summary": summary, "provider": "JournalPulse",
        "resource_type": "activity", "coping_style": style, "kind": kind,
        "duration_minutes": duration_seconds // 60 if timed else None,
        "duration_seconds": duration_seconds if timed else None,
        "timer_enabled": timed, "instructions": instructions, "source": "builtin",
        "evidence_kind": "app_guidance", "no_audio": True, "no_video": True,
        "seated": seated, "goal_tags": goal_tags,
    }).model_dump(mode="json")


def builtin_activities() -> list[dict[str, Any]]:
    """Return fresh descriptors so callers cannot mutate later people's choices."""
    return [
        _builtin(
            "guided_meditation_2m", "Two-minute quiet meditation",
            "A short, silent pause to notice your surroundings without forcing a change.",
            "meditation", "ground", ["settle", "ground"], [
                "Sit in a comfortable position. Keep your eyes open or closed, as you prefer.",
                "Notice the contact of the chair or floor, or a sound around you.",
                "When your attention wanders, gently return to that sensation. Breathe naturally.",
                "You can pause or stop at any point. The timer does not measure how you feel.",
            ],
        ),
        _builtin(
            "guided_meditation_1m", "One-minute quiet meditation",
            "A shorter, silent pause when only one minute is available.",
            "meditation", "ground", ["settle", "ground"], [
                "Sit comfortably, with your eyes open or closed as you prefer.",
                "Notice the contact of the chair or floor, or a sound around you. Breathe naturally.",
                "Gently return your attention when it wanders. You can stop at any point.",
            ], duration_seconds=60,
        ),
        _builtin(
            "guided_movement_2m", "Two minutes of comfortable movement",
            "A gentle seated movement break, within your own comfortable range.",
            "movement", "move", ["move", "movement", "act"], [
                "Sit comfortably and choose a small movement that feels easy for you.",
                "For example, gently move your hands or shoulders within a comfortable range.",
                "Keep the movement easy. Stop if it hurts, feels unsteady, or feels uncomfortable.",
                "You can finish early; doing more is not the goal.",
            ],
        ),
        _builtin(
            "guided_reflection_2m", "Two-minute reflection pause",
            "Make space for one concern and what matters to you about it.",
            "reflection", "reflect", ["understand", "reframing"], [
                "Notice one situation you want to understand; you do not need to solve it now.",
                "Separate what happened from what you think it might mean.",
                "Ask yourself what matters to you in this situation. You can jot a few words privately.",
                "Leave uncertainty open if you do not have an answer.",
            ],
        ),
        _builtin(
            "guided_connection_step", "One small connection step",
            "Choose a comfortable way to connect, with no obligation to send anything.",
            "social", "connect", ["connect", "connection"], [
                "Choose someone you feel comfortable contacting, if anyone comes to mind.",
                "Draft a brief hello or invitation. You can decide later whether to send it.",
                "Use Done or Not yet to report what you actually tried.",
            ], timed=False,
        ),
        _builtin(
            "guided_focus_2m", "Two minutes on one small task",
            "Try one manageable step toward the task you chose, without needing to finish it.",
            "focus", "move", ["act", "planning"], [
                "Choose a small part of your task that is possible right now.",
                "Start that one part, such as opening the document or writing one line.",
                "When the timer ends, decide whether to continue or pause. Finishing is optional.",
            ],
        ),
    ]


def _catalog_resource(resource: dict[str, Any]) -> dict[str, Any]:
    resource_type = resource["resource_type"]
    style = resource["coping_style"]
    kind = "video" if resource_type == "video" else "play" if resource_type == "game" else "reading"
    # A web link alone does not supply instructions for an in-app movement timer.
    return ActivityResource.model_validate({
        "id": resource["id"], "title": resource["title"], "url": resource["url"],
        "summary": resource["summary"], "provider": resource["provider"],
        "resource_type": resource_type, "coping_style": style, "kind": kind,
        "duration_minutes": resource.get("duration_minutes"), "duration_seconds": None,
        "timer_enabled": False, "source": "catalog", "evidence_kind": "catalog_review",
        # A website or game container can contain audio/video. Only explicit
        # reviewed capability metadata can satisfy a hard format preference.
        "no_audio": resource.get("no_audio", False), "no_video": resource.get("no_video", False),
        "seated": style != "move", "breath_focus": resource["id"] in {
            "site_nhs_breathing", "video_meditation_start_day",
        },
        "goal_tags": resource.get("goal_tags", []), "emotion_tags": resource.get("emotion_tags", []),
        "reviewed_at": resource.get("reviewed_at"), "source_tier": resource.get("source_tier"),
    }).model_dump(mode="json")


_GOAL_TAGS = {
    "settle": {"settle", "ground"}, "move": {"move", "movement"},
    "understand": {"understand", "reframing", "reading"},
    "connect": {"connect", "connection"}, "act": {"act", "planning", "movement", "play"},
}


def activity_candidates(
    path: Path, *, goal: str | None = None, constraints: ActivityConstraints | None = None,
    excluded_ids: tuple[str, ...] = (), limit: int = 12,
) -> list[dict[str, Any]]:
    """Filter hard preferences first; Luna then explains a choice among these IDs.

    The ordering is a transparent catalog heuristic, not a learned policy or
    randomized action probability. Emotion guesses do not become user reports.
    """
    if not 1 <= limit <= 16:
        raise ValueError("Choose between one and sixteen activity candidates")
    if goal is not None and goal not in ACTIVITY_GOALS:
        raise ValueError("Unknown activity goal")
    preferences = constraints or ActivityConstraints()
    all_resources = [*builtin_activities(), *[
        _catalog_resource(item) for item in load_catalog(path) if item["resource_type"] != "support"
    ]]
    eligible = []
    for resource in all_resources:
        if resource["id"] in excluded_ids or not activity_resource_matches_constraints(resource, preferences):
            continue
        eligible.append(resource)
    if goal is not None:
        tags = _GOAL_TAGS[goal]
        eligible.sort(key=lambda item: not bool(tags.intersection(item["goal_tags"])))
    return eligible[:limit]


def activity_resource_matches_constraints(
    resource: dict[str, Any], constraints: ActivityConstraints,
) -> bool:
    """Unknown external format/duration cannot become a claimed hard match."""
    checked = ActivityResource.model_validate(resource)
    duration = checked.duration_minutes
    return not (
        constraints.time_minutes is not None and (duration is None or duration > constraints.time_minutes)
        or constraints.no_audio and not checked.no_audio
        or constraints.no_video and not checked.no_video
        or constraints.seated and not checked.seated
        or constraints.avoid_breath_focus and (checked.breath_focus or checked.source == "search_snippet")
    )


def resolve_activity_resource(path: Path, resource_id: str) -> dict[str, Any] | None:
    """Resolve catalog/app IDs only; a discovered URL requires a signed receipt."""
    for resource in builtin_activities():
        if resource["id"] == resource_id:
            return resource
    for resource in load_catalog(path):
        if resource["id"] == resource_id and resource["resource_type"] != "support":
            return _catalog_resource(resource)
    return None


_GOAL_QUERY = {
    "settle": "brief grounding", "move": "gentle movement", "understand": "self reflection",
    "connect": "connection communication", "act": "small task focus",
}
_STYLE_QUERY = {
    "ground": "mindfulness meditation", "move": "gentle movement", "connect": "social connection",
    "reflect": "reflection", "read": "reading", "play": "creative puzzle", "watch": "video",
    "pause": "quiet pause", "meditation": "mindfulness meditation", "movement": "gentle movement",
    "reading": "reading", "video": "video", "social": "social connection",
}
_TOPIC_QUERY = {
    "meditation": "mindfulness meditation", "grounding": "brief grounding",
    "movement": "gentle movement", "walking": "gentle walking",
    "stretching": "gentle stretching", "reflection": "self reflection",
    "communication": "connection communication", "focus": "small task focus",
    "reading": "reading", "puzzle": "creative puzzle", "video": "video",
    "pause": "quiet pause",
}
_PUBLIC_QUERY_WORDS = frozenset(
    "brief grounding gentle movement self reflection connection communication small task focus "
    "mindfulness meditation social reading creative puzzle video quiet pause silent seated text "
    "minute minutes beginner free shorter written outdoors indoors walking walk stretching "
    "breathing exercise exercises music journaling no audio easy calm nature gratitude kindness "
    "attention body awareness comfortable low effort activity activities tutorial guide without "
    "two five ten one three four six seven eight nine twelve fifteen twenty".split()
)


def validate_general_search_topic(topic: str) -> str:
    """Accept public activity terms only; unknown/private words never reach Brave.

    This deliberately narrow first-slice boundary can reject uncommon wording.
    A prompt saying 'remove names' is not an enforceable disclosure boundary.
    """
    if not isinstance(topic, str) or not 3 <= len(topic) <= 160:
        raise ValueError("Use a short general activity topic")
    normalized = " ".join(topic.lower().split())
    if not re.fullmatch(r"[a-z0-9 ]+", normalized):
        raise ValueError("Use general activity words without links or identifiers")
    for word in normalized.split():
        if word not in _PUBLIC_QUERY_WORDS and not (word.isdigit() and 1 <= int(word) <= 20):
            raise ValueError("Use general activity words without personal details")
    return normalized


def general_search_topic(
    *, goal: str, style: str, constraints: ActivityConstraints,
    topic: ActivitySearchTopic | None = None,
) -> str:
    """Compile public query text; model prose never crosses the search boundary.

    Guided Luna selects a schema-enforced category. Existing user-edited search
    still passes the separate strict vocabulary validator before any provider call.
    """
    if goal not in _GOAL_QUERY or style not in _STYLE_QUERY:
        raise ValueError("Unknown general activity goal or style")
    if topic is not None and topic not in _TOPIC_QUERY:
        raise ValueError("Unknown public activity search category")
    terms = [_GOAL_QUERY[goal], _TOPIC_QUERY[topic] if topic else _STYLE_QUERY[style]]
    if constraints.time_minutes is not None:
        terms.append(f"{constraints.time_minutes} minute")
    if constraints.no_audio:
        terms.append("silent")
    if constraints.no_video:
        terms.append("text")
    if constraints.seated:
        terms.append("seated")
    if constraints.avoid_breath_focus:
        terms.append("without breathing exercises")
    return validate_general_search_topic(" ".join(terms))


def discovery_activity_resource(
    candidate: DiscoveryCandidate, provenance: DiscoveryProvenance | None = None,
) -> dict[str, Any]:
    identity = source_url_identity(candidate.url)
    host = candidate.url.split("/", 3)[2]
    is_video = host in {"www.youtube.com", "youtube.com", "youtu.be", "www.vimeo.com", "vimeo.com"}
    resource_id = "discovered_" + hashlib.sha256(identity.encode()).hexdigest()[:24]
    return ActivityResource(
        id=resource_id, title=candidate.title, url=candidate.url, summary=candidate.description,
        provider="Brave search", resource_type="video" if is_video else "website",
        coping_style="watch" if is_video else "read", kind="video" if is_video else "reading",
        source="search_snippet", evidence_kind="search_snippet", why_selected=candidate.why_selected,
        discovery_provenance=provenance,
        # Snippets do not establish audio/accessibility, duration or full-page safety.
        no_audio=False, no_video=False, seated=False,
    ).model_dump(mode="json")


class _ResourceReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    version: Literal[1, 2]
    goal: Goal | None = None
    user_id: str = Field(min_length=36, max_length=36)
    conversation_id: str = Field(min_length=36, max_length=36)
    conversation_revision: int = Field(ge=0)
    conversation_incarnation_id: str | None = Field(default=None, min_length=36, max_length=36)
    issued_at: int = Field(ge=0)
    expires_at: int = Field(ge=0)
    resource: ActivityResource


def _signing_key(settings: Settings) -> bytes:
    key = settings.write_signing_key
    if not key or len(key) < 32:
        raise ValueError("Inline resource receipts require configured server signing")
    return key.encode()


def _urlsafe_encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode().rstrip("=")


def issue_resource_token(
    settings: Settings, *, user_id: UUID, conversation_id: UUID, conversation_revision: int,
    resource: dict[str, Any], now: datetime | None = None, goal: Goal | None = None,
    conversation_incarnation_id: UUID | None = None,
) -> str:
    moment = now or datetime.now(UTC)
    descriptor = ActivityResource.model_validate(resource)
    if descriptor.source != "search_snippet":
        raise ValueError("Only retrieved snippets use resource receipts")
    receipt = _ResourceReceipt(
        version=2 if goal is not None else 1, goal=goal,
        user_id=str(user_id), conversation_id=str(conversation_id),
        conversation_incarnation_id=str(conversation_incarnation_id) if conversation_incarnation_id else None,
        conversation_revision=conversation_revision, issued_at=int(moment.timestamp()),
        expires_at=int((moment + timedelta(seconds=RESOURCE_TOKEN_TTL_SECONDS)).timestamp()),
        resource=descriptor,
    )
    payload = receipt.model_dump(mode="json")
    if receipt.version == 1:
        payload.pop("goal")
        payload.pop("conversation_incarnation_id")
    raw = json.dumps(payload, separators=(",", ":")).encode()
    signature = hmac.new(_signing_key(settings), _TOKEN_DOMAIN + raw, hashlib.sha256).digest()
    token = f"{_urlsafe_encode(raw)}.{_urlsafe_encode(signature)}"
    if len(token) > MAX_RESOURCE_TOKEN_BYTES:
        raise ValueError("Resource receipt exceeded its size limit")
    return token


def verify_resource_offer(
    settings: Settings, token: str, *, user_id: UUID, conversation_id: UUID,
    conversation_revision: int, now: datetime | None = None,
    conversation_incarnation_id: UUID | None = None,
) -> _ResourceReceipt:
    """Verify authenticity before decoding or trusting the resource descriptor.

    Binding to the current revision discards replies from old chat turns/tabs.
    This token grants neither journal access nor a generic database write.
    """
    if not isinstance(token, str) or len(token) > MAX_RESOURCE_TOKEN_BYTES:
        raise ValueError("Invalid resource receipt")
    try:
        encoded, encoded_signature = token.split(".")
        if not re.fullmatch(r"[A-Za-z0-9_-]+", encoded + encoded_signature):
            raise ValueError("Invalid resource receipt")
        raw = base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4))
        signature = base64.urlsafe_b64decode(encoded_signature + "=" * (-len(encoded_signature) % 4))
        expected = hmac.new(_signing_key(settings), _TOKEN_DOMAIN + raw, hashlib.sha256).digest()
        if not hmac.compare_digest(signature, expected):
            raise ValueError("Invalid resource receipt")
        receipt = _ResourceReceipt.model_validate_json(raw)
    except (ValueError, UnicodeError, binascii.Error, RecursionError) as exc:
        raise ValueError("Invalid or expired resource receipt") from exc
    timestamp = int((now or datetime.now(UTC)).timestamp())
    if (
        receipt.user_id != str(user_id)
        or receipt.conversation_id != str(conversation_id)
        or receipt.conversation_revision != conversation_revision
        or receipt.conversation_incarnation_id != (
            str(conversation_incarnation_id) if conversation_incarnation_id else None
        )
        or receipt.issued_at > timestamp + 30
        or receipt.expires_at <= timestamp
        or receipt.expires_at - receipt.issued_at != RESOURCE_TOKEN_TTL_SECONDS
        or receipt.resource.source != "search_snippet"
    ):
        raise ValueError("Invalid or expired resource receipt")
    if receipt.version == 2 and receipt.goal is None:
        raise ValueError("A versioned search offer needs its goal")
    return receipt


def verify_resource_token(
    settings: Settings, token: str, *, user_id: UUID, conversation_id: UUID,
    conversation_revision: int, now: datetime | None = None,
    conversation_incarnation_id: UUID | None = None,
) -> dict[str, Any]:
    """Compatibility descriptor view; saving uses the full verified receipt."""
    return verify_resource_offer(
        settings, token, user_id=user_id, conversation_id=conversation_id,
        conversation_revision=conversation_revision, now=now,
        conversation_incarnation_id=conversation_incarnation_id,
    ).resource.model_dump(mode="json")
