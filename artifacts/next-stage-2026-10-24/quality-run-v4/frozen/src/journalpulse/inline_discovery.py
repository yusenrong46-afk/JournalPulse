"""Conversation-scoped search with public topics and signed resource offers."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from typing import Literal
from uuid import UUID

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from .activity_resources import (
    ActivityResource,
    activity_resource_matches_constraints,
    discovery_activity_resource,
    general_search_topic,
    issue_resource_token,
    validate_general_search_topic,
)
from .auth import AuthContext
from .config import Settings
from .conversations import MAX_USER_MESSAGES, GenerationLimit, _require_source
from .discovery import DiscoveryClient, DiscoveryProviderError, OpenWebDiscoveryClient
from .discovery_models import MAX_EXCLUDED_URLS, DiscoveryRequest, DiscoveryResponse
from .domain import (
    ActivityConstraintInputs,
    Conversation,
    ConversationStatus,
    Goal,
    InteractionPreference,
    MessageRole,
    SafetyMode,
)
from .persistence import Repository


class InlineDiscoveryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    expected_revision: int = Field(ge=0)
    llm_consent: bool = False
    goal: Goal | None = None
    style: Literal["ground", "move", "connect", "reflect", "play", "watch", "read", "pause"] = "ground"
    constraints: ActivityConstraintInputs | None = None
    original_query: str | None = Field(default=None, max_length=160)
    previous_query: str | None = Field(default=None, max_length=400)
    feedback: str | None = Field(default=None, max_length=160)
    excluded_urls: list[str] = Field(default_factory=list, max_length=MAX_EXCLUDED_URLS)


class InlineResourceOffer(BaseModel):
    resource: ActivityResource
    resource_token: str


class InlineDiscoveryResponse(DiscoveryResponse):
    offers: list[InlineResourceOffer] = Field(max_length=3)
    conversation_revision: int


def public_search_query(value: str) -> str:
    """Validate longer refinements without splitting a word at a byte boundary."""
    if not 3 <= len(value) <= 400:
        raise ValueError("Search query exceeded its public-topic limit")
    words = value.lower().split()
    for word in words:
        validate_general_search_topic(f"activity {word}")
    return " ".join(words)


def register_inline_discovery_routes(
    app: FastAPI,
    *,
    settings: Settings,
    repositories: Callable[[AuthContext], Repository],
    enforce_generation_limit: GenerationLimit,
    auth_dependency: Callable[..., AuthContext],
    client: DiscoveryClient | None,
    clock: Callable[[], datetime],
) -> None:
    def require_context(repository: Repository, auth: AuthContext, identity: UUID) -> Conversation:
        conversation = repository.get_conversation(auth.user_id, identity)
        if conversation is None:
            raise HTTPException(404, "Conversation not found")
        if (
            conversation.status != ConversationStatus.OPEN
            or conversation.safety_mode == SafetyMode.SUPPORT
            or conversation.interaction_preference == InteractionPreference.LISTEN
            or conversation.activity_move == "pause"
        ):
            raise HTTPException(409, "This chat is not taking activity suggestions right now.")
        _require_source(
            repository,
            auth.user_id,
            conversation.source_entry_id,
            expected_created_at=conversation.source_entry_created_at,
        )
        if (
            sum(
                message.role == MessageRole.USER
                for message in repository.list_messages(auth.user_id, identity)
            )
            >= MAX_USER_MESSAGES
        ):
            raise HTTPException(
                409, "This chat reached its message limit. Existing activity check-ins still work."
            )
        return conversation

    @app.post("/v1/conversations/{conversation_id}/discover", response_model=InlineDiscoveryResponse)
    def discover(
        conversation_id: UUID,
        payload: InlineDiscoveryRequest,
        auth: AuthContext = Depends(auth_dependency),
    ) -> InlineDiscoveryResponse:
        repository = repositories(auth)
        repository.close_stale_conversations(auth.user_id, now=clock())
        conversation = require_context(repository, auth, conversation_id)
        if conversation.revision != payload.expected_revision:
            raise HTTPException(409, "This chat changed. Refresh before searching.")
        if not conversation.llm_consent or not payload.llm_consent:
            raise HTTPException(409, "Choose web-search consent before searching.")
        if not settings.discovery_enabled:
            raise HTTPException(
                503, "Web search is unavailable. You can keep talking or try an app activity."
            )
        if not settings.write_signing_key or len(settings.write_signing_key) < 32:
            raise HTTPException(503, "Inline search cannot safely save an offer right now.")
        goal = payload.goal or conversation.activity_goal or Goal.SETTLE
        constraints = payload.constraints or conversation.activity_constraints
        try:
            # A vocabulary validator is the disclosure boundary. Prompt-generated
            # topics and edited/refined queries receive exactly the same checks.
            original = (
                validate_general_search_topic(payload.original_query)
                if payload.original_query
                else (general_search_topic(goal=goal.value, style=payload.style, constraints=constraints))
            )
            previous = None
            if payload.previous_query:
                # Existing refinement can append terms. Validate each bounded
                # segment, rather than accepting arbitrary narrative as feedback.
                previous = public_search_query(payload.previous_query)
            feedback = validate_general_search_topic(payload.feedback) if payload.feedback else None
            request = DiscoveryRequest(
                original_query=original,
                previous_query=previous,
                feedback=feedback,
                excluded_urls=payload.excluded_urls,
                llm_consent=True,
                locale=conversation.locale,
            )
        except ValueError as exc:
            raise HTTPException(
                422,
                "Use general activity words such as 'quiet meditation' or 'shorter text', "
                "without personal details.",
            ) from exc
        enforce_generation_limit(auth, repository)
        try:
            result = (client or OpenWebDiscoveryClient(settings, query_validator=public_search_query)).search(
                request
            )
        except DiscoveryProviderError as exc:
            raise HTTPException(exc.status_code, str(exc)) from exc
        # Search/model latency cannot turn a stale journal or preference into a
        # new offer. Tokens bind the exact owner, chat revision and expiry.
        current = require_context(repository, auth, conversation_id)
        if (current.revision != conversation.revision
                or current.incarnation_id != conversation.incarnation_id
                or current.created_at != conversation.created_at):
            raise HTTPException(409, "This chat changed while searching. These results were not offered.")
        offers = []
        for candidate in result.candidates:
            descriptor = discovery_activity_resource(candidate, result.provenance)
            if not activity_resource_matches_constraints(descriptor, constraints):
                # Unknown snippet facts cannot bypass a confirmed hard limit.
                continue
            offers.append(
                InlineResourceOffer(
                    resource=ActivityResource.model_validate(descriptor),
                    resource_token=issue_resource_token(
                        settings,
                        user_id=auth.user_id,
                        conversation_id=conversation_id,
                        conversation_revision=current.revision,
                        conversation_incarnation_id=current.incarnation_id,
                        resource=descriptor,
                        goal=goal,
                        now=clock(),
                    ),
                )
            )
        limitations = result.limitations
        if len(offers) < len(result.candidates):
            limitations = [
                *limitations,
                "Some links were not offered because their snippets cannot verify your activity limits.",
            ]
        return InlineDiscoveryResponse(
            **result.model_dump(exclude={"limitations"}),
            limitations=limitations,
            offers=offers,
            conversation_revision=current.revision,
        )
