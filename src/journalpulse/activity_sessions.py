"""The in-chat activity loop: control time, save a report, then optionally reflect.

Starting an activity never calls legacy acceptance or closes/clears its chat. The
first check-in is deterministic; a model is used only for one bounded follow-up.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from typing import Any
from uuid import UUID, uuid5

from fastapi import Depends, FastAPI, HTTPException, Query, Response

from .activity_lifecycle import ActivityConflict, ActivityNotFound
from .activity_models import (
    ActivityCommandRequest,
    ActivityFollowUpRequest,
    ActivityHistoryItem,
    ActivityHistoryPage,
    ActivityReportRequest,
    ActivityResource,
    ActivitySelectionProvenance,
    ActivitySession,
    ActivityStatus,
    CreateActivitySessionRequest,
)
from .activity_resources import ActivityConstraints, activity_candidates
from .auth import AuthContext
from .config import Settings
from .domain import (
    ActivityFollowUpDirective,
    Conversation,
    ConversationMessage,
    ConversationMode,
    ConversationStatus,
    Goal,
    ModelRun,
    SafetyMode,
)
from .intelligence import ConversationProviderError, UnsupportedProviderResponse
from .persistence import (
    ConversationClosed,
    ConversationNotFound,
    ConversationStale,
    JournalEntryNotFound,
    Repository,
)
from .safety import SUPPORT_FALLBACK_MESSAGE, assess_safety

ActivityFollowUpResult = tuple[str, ModelRun | None] | tuple[str, ModelRun | None, ActivityFollowUpDirective]
ActivityFollowUpGenerator = Callable[
    [Conversation, list[ConversationMessage], ActivitySession], ActivityFollowUpResult
]
DefaultActivityFollowUpGenerator = Callable[
    [Repository, Conversation, list[ConversationMessage], ActivitySession], ActivityFollowUpResult
]


def session_resource(descriptor: dict[str, Any]) -> ActivityResource:
    """Keep only reviewed activity facts; catalog metadata is not private context."""
    source = descriptor.get("source", "catalog")
    kind: Any = descriptor.get("kind", "other")
    kind = {"social": "connection", "play": "other"}.get(str(kind), kind)
    if kind not in {
        "meditation",
        "movement",
        "reflection",
        "connection",
        "focus",
        "video",
        "reading",
        "other",
    }:
        kind = "video" if descriptor.get("coping_style") == "watch" else "reading"
    timer = bool(descriptor.get("timer_enabled", False))
    duration = descriptor.get("duration_seconds")
    instructions = descriptor.get("instructions", [])
    if isinstance(instructions, str):
        instructions = [instructions]
    return ActivityResource(
        id=descriptor["id"],
        title=descriptor["title"],
        url=descriptor.get("url"),
        provider=descriptor.get("provider", "JournalPulse"),
        resource_type=descriptor.get("resource_type", "activity"),
        format="timer" if timer else "external" if descriptor.get("url") else "manual",
        kind=kind,
        duration_seconds=duration if timer else None,
        instructions=instructions,
        discovery_provenance=descriptor.get("discovery_provenance"),
        provenance="search_snippet"
        if source == "search_snippet"
        else "builtin"
        if source == "builtin"
        else "catalog",
    )


def register_activity_routes(
    app: FastAPI,
    *,
    settings: Settings,
    repositories: Callable[[AuthContext], Repository],
    auth_dependency: Callable[..., AuthContext],
    clock: Callable[[], datetime],
    enforce_generation_limit: Callable[[AuthContext, Repository], None],
    follow_up: ActivityFollowUpGenerator | None = None,
    default_follow_up: DefaultActivityFollowUpGenerator | None = None,
) -> None:
    @app.get("/v1/activity-resources")
    def reviewed_resources(
        goal: Goal | None = None,
        time_minutes: int | None = Query(default=None, ge=1, le=20),
        no_audio: bool = False,
        no_video: bool = False,
        seated: bool = False,
        avoid_breath_focus: bool = False,
        auth: AuthContext = Depends(auth_dependency),
    ) -> dict[str, Any]:
        del auth
        constraints = ActivityConstraints(
            time_minutes=time_minutes, no_audio=no_audio, no_video=no_video,
            seated=seated, avoid_breath_focus=avoid_breath_focus,
        )
        return {"items": activity_candidates(
            settings.resource_catalog_path, goal=goal.value if goal else None,
            constraints=constraints, limit=16,
        )}

    def sweep(auth: AuthContext) -> Repository:
        repository = repositories(auth)
        repository.close_stale_conversations(auth.user_id, now=clock())
        return repository

    def owned_chat(
        repository: Repository, auth: AuthContext, conversation_id: UUID, *, open_only: bool
    ) -> Conversation:
        conversation = repository.get_conversation(auth.user_id, conversation_id)
        if conversation is None:
            raise HTTPException(404, "Conversation not found")
        if open_only and conversation.status != ConversationStatus.OPEN:
            raise HTTPException(409, "This conversation is closed.")
        if conversation.source_entry_id is not None:
            source = repository.get_journal_entry(auth.user_id, conversation.source_entry_id)
            if source is None or source.created_at != conversation.source_entry_created_at:
                raise HTTPException(404, "Journal entry not found")
        return conversation

    def owned_session(
        repository: Repository,
        auth: AuthContext,
        session_id: UUID,
        *,
        open_only: bool,
    ) -> tuple[Conversation, ActivitySession]:
        session = repository.get_activity_session(auth.user_id, session_id)
        if session is None:
            raise HTTPException(404, "Activity not found")
        return owned_chat(repository, auth, session.conversation_id, open_only=open_only), session

    def view(repository: Repository, auth: AuthContext, session: ActivitySession) -> ActivitySession:
        conversation = owned_chat(repository, auth, session.conversation_id, open_only=False)
        moment = clock()
        # A worker can disappear after claiming a follow-up. Expose an expired
        # lease as retryable without changing the stored claim: the explicit retry
        # still uses the repository's atomic takeover and three-attempt bound.
        expired_claim = (
            session.follow_up_status == "generating"
            and session.follow_up_lease_until is not None
            and moment >= session.follow_up_lease_until
        )
        return session.model_copy(
            update={
                "server_now": moment,
                "conversation_revision": conversation.revision,
                "follow_up_status": "failed" if expired_claim else session.follow_up_status,
            }
        )

    def translated(operation: Callable[[], Any]) -> Any:
        try:
            return operation()
        except (ActivityNotFound, ConversationNotFound, JournalEntryNotFound) as exc:
            raise HTTPException(404, "Activity or its source was deleted.") from exc
        except ConversationClosed as exc:
            raise HTTPException(409, "This conversation is closed.") from exc
        except ConversationStale as exc:
            raise HTTPException(409, "This chat changed. Refresh before trying again.") from exc
        except ActivityConflict as exc:
            raise HTTPException(409, str(exc)) from exc

    def synced(repository: Repository, auth: AuthContext, session: ActivitySession) -> ActivitySession:
        conversation = owned_chat(repository, auth, session.conversation_id, open_only=False)
        if (
            conversation.status == ConversationStatus.OPEN
            and session.status == ActivityStatus.ACTIVE
            and session.expires_at is not None
            and clock() >= session.expires_at
        ):
            # All tabs use the same receipt identity for this deadline. A GET after
            # background/refresh performs one short transaction, never a sleeping worker.
            request = ActivityCommandRequest(
                client_request_id=uuid5(
                    session.id, f"expiry:{session.revision}:{session.expires_at.isoformat()}"
                ),
                expected_revision=session.revision,
                expected_conversation_revision=conversation.revision,
                command="expire",
            )
            try:
                session = repository.command_activity_session(auth.user_id, session.id, request, now=clock())
            except (ActivityConflict, ConversationStale, ConversationClosed):
                _, session = owned_session(repository, auth, session.id, open_only=False)
        return view(repository, auth, session)

    @app.get("/v1/conversations/{conversation_id}/activity-sessions", response_model=ActivitySession | None)
    def latest_activity(
        conversation_id: UUID, auth: AuthContext = Depends(auth_dependency)
    ) -> ActivitySession | None:
        repository = sweep(auth)
        owned_chat(repository, auth, conversation_id, open_only=False)
        sessions = repository.list_activity_sessions(auth.user_id, conversation_id)
        return synced(repository, auth, sessions[0]) if sessions else None

    @app.get("/v1/activity-history", response_model=ActivityHistoryPage)
    def activity_history(
        limit: int = Query(default=50, ge=1, le=100),
        auth: AuthContext = Depends(auth_dependency),
    ) -> ActivityHistoryPage:
        # Read-only and owner-scoped (PostgreSQL RLS also applies). Only the person's own
        # report reaches the garden; timer expiry or an unstarted offer adds nothing.
        sessions = sweep(auth).list_activity_sessions(auth.user_id)
        reported = sorted(
            (item for item in sessions if item.report is not None and item.reported_at is not None),
            key=lambda item: (item.reported_at, str(item.id)),
            reverse=True,
        )
        return ActivityHistoryPage(
            items=[ActivityHistoryItem.from_session(item) for item in reported[:limit]]
        )

    @app.get("/v1/activity-sessions/{session_id}", response_model=ActivitySession)
    def read_activity(session_id: UUID, auth: AuthContext = Depends(auth_dependency)) -> ActivitySession:
        repository = sweep(auth)
        _, session = owned_session(repository, auth, session_id, open_only=False)
        return synced(repository, auth, session)

    @app.post(
        "/v1/conversations/{conversation_id}/activity-sessions",
        response_model=ActivitySession,
        status_code=201,
    )
    def offer_activity(
        conversation_id: UUID,
        payload: CreateActivitySessionRequest,
        auth: AuthContext = Depends(auth_dependency),
    ) -> ActivitySession:
        # This helper is imported at use to keep route registration independent
        # from the provider. It resolves static resources or verifies a signed snippet.
        from .activity_resources import (
            activity_resource_matches_constraints,
            resolve_activity_resource,
            verify_resource_offer,
        )

        repository = sweep(auth)
        conversation = owned_chat(repository, auth, conversation_id, open_only=True)
        active_card = conversation.activity_card or conversation.card
        search_goal = conversation.activity_goal or Goal.SETTLE
        previous = repository.get_activity_session(auth.user_id, payload.client_request_id)
        if previous is not None and previous.conversation_id == conversation_id:
            candidate = previous.resource
        else:
            try:
                if payload.resource_token is not None:
                    receipt = verify_resource_offer(
                        settings,
                        payload.resource_token,
                        user_id=auth.user_id,
                        conversation_id=conversation_id,
                        conversation_revision=conversation.revision,
                        conversation_incarnation_id=conversation.incarnation_id,
                        now=clock(),
                    )
                    descriptor = receipt.resource.model_dump(mode="json")
                    search_goal = Goal(receipt.goal) if receipt.goal else search_goal
                    if descriptor.get("id") != payload.resource_id:
                        raise ValueError("Resource identity changed")
                else:
                    if active_card is None or payload.resource_id not in {
                        item.get("id") for item in active_card.actions
                    }:
                        raise ValueError("Choose from the current Luna recommendation")
                    resolved = resolve_activity_resource(settings.resource_catalog_path, payload.resource_id)
                    if resolved is None:
                        raise ValueError("The recommended resource is unavailable")
                    descriptor = resolved
                if not activity_resource_matches_constraints(descriptor, conversation.activity_constraints):
                    raise ValueError("Resource does not fit the current activity preferences")
                candidate = session_resource(descriptor)
            except (ValueError, KeyError, TypeError) as exc:
                raise HTTPException(409, "This recommendation changed. Ask Luna for another option.") from exc
        if candidate.id != payload.resource_id:
            raise HTTPException(409, "Activity request ID is already in use")
        if payload.duration_seconds is not None and candidate.format != "timer":
            raise HTTPException(422, "Only a timed activity can have a countdown duration")
        duration = (
            payload.duration_seconds
            if payload.duration_seconds is not None
            else candidate.duration_seconds or 0
        )
        selection_card = None if payload.resource_token else active_card
        recommended = selection_card.decision_preview.recommended_action_id if selection_card else None
        selection = ActivitySelectionProvenance(
            selection_source="search"
            if payload.resource_token
            else "guided"
            if conversation.mode == ConversationMode.GUIDED
            else "user",
            recommended_resource_id=recommended or payload.resource_id,
            selected_resource_id=payload.resource_id,
            model_run=(
                selection_card.decision_preview.context_snapshot.get("model_run") if selection_card else None
            ),
        )
        session = ActivitySession(
            id=payload.client_request_id,
            user_id=auth.user_id,
            conversation_id=conversation_id,
            source_entry_id=conversation.source_entry_id,
            offered_message_id=active_card.offered_message_id if active_card else None,
            resource=candidate,
            goal=(previous.goal if previous is not None else search_goal)
            if payload.resource_token else active_card.goal if active_card else None,
            # The visible card may include private context; never copy its freeform
            # reason (or journal quotations) into an activity row.
            recommendation_reason=None,
            selection=selection,
            duration_seconds=duration,
            remaining_seconds=duration,
            created_at=clock(),
            updated_at=clock(),
        )
        saved = translated(
            lambda: repository.offer_activity_session(
                session,
                request_id=payload.client_request_id,
                expected_conversation_revision=payload.expected_conversation_revision,
                now=clock(),
            )
        )
        return view(repository, auth, saved)

    @app.post("/v1/activity-sessions/{session_id}/commands", response_model=ActivitySession)
    def control_activity(
        session_id: UUID,
        payload: ActivityCommandRequest,
        auth: AuthContext = Depends(auth_dependency),
    ) -> ActivitySession:
        repository = sweep(auth)
        owned_session(repository, auth, session_id, open_only=True)
        saved = translated(
            lambda: repository.command_activity_session(auth.user_id, session_id, payload, now=clock())
        )
        return view(repository, auth, saved)

    @app.post("/v1/activity-sessions/{session_id}/report", response_model=ActivitySession)
    def report_activity(
        session_id: UUID,
        payload: ActivityReportRequest,
        auth: AuthContext = Depends(auth_dependency),
    ) -> ActivitySession:
        repository = sweep(auth)
        conversation, _ = owned_session(repository, auth, session_id, open_only=True)
        safety = assess_safety(payload.note or "", conversation.locale)
        saved = translated(
            lambda: repository.report_activity_session(
                auth.user_id,
                session_id,
                payload,
                now=clock(),
                safety=safety,
            )
        )
        return view(repository, auth, saved)

    @app.post("/v1/activity-sessions/{session_id}/follow-up", response_model=ActivitySession)
    def follow_up_activity(
        session_id: UUID,
        payload: ActivityFollowUpRequest,
        response: Response,
        auth: AuthContext = Depends(auth_dependency),
    ) -> ActivitySession:
        repository = sweep(auth)
        conversation, current = owned_session(repository, auth, session_id, open_only=True)
        if current.report is None:
            raise HTTPException(409, "Save a check-in before asking Luna to reflect")
        support = conversation.safety_mode == SafetyMode.SUPPORT
        use_model = not support and conversation.mode == ConversationMode.AI
        if use_model and current.follow_up_status != "ready":
            if not conversation.llm_consent or not settings.openrouter_enabled or not settings.openrouter_zdr:
                raise HTTPException(409, "Private AI help needs consent and provider retention disabled")
            if follow_up is None and default_follow_up is None:
                raise HTTPException(
                    503, "Your check-in is saved. Luna's follow-up is temporarily unavailable."
                )
            # Avoid spending a budget slot for the duplicate of a live claim.
            if (
                current.follow_up_status != "generating"
                or current.follow_up_lease_until is None
                or clock() >= current.follow_up_lease_until
            ):
                enforce_generation_limit(auth, repository)
        claimed, owns_generation = translated(
            lambda: repository.claim_activity_follow_up(
                auth.user_id,
                session_id,
                payload,
                now=clock(),
            )
        )
        if not owns_generation:
            response.status_code = 202 if claimed.follow_up_status == "generating" else 200
            return view(repository, auth, claimed)
        conversation = owned_chat(repository, auth, claimed.conversation_id, open_only=True)
        if conversation.revision != payload.expected_conversation_revision:
            raise HTTPException(409, "This chat changed before Luna could reflect. Your check-in is saved.")
        canonical = repository.get_activity_session(auth.user_id, session_id)
        if canonical is None or canonical.revision != claimed.revision:
            raise HTTPException(
                409, "This activity changed before Luna could reflect. Your check-in is saved."
            )
        support = conversation.safety_mode == SafetyMode.SUPPORT
        generation_revision = conversation.revision
        directive: ActivityFollowUpDirective | None = None
        try:
            if support:
                reply = conversation.safety.support_message if conversation.safety else None
                reply = reply or SUPPORT_FALLBACK_MESSAGE
                model_run = None
            elif not use_model:
                reply = "Your check-in is saved. You can keep talking about what changed, or leave it here."
                model_run = None
            else:
                messages = repository.list_messages(auth.user_id, conversation.id)
                if follow_up is not None:
                    completion = follow_up(conversation, messages, claimed)
                else:
                    assert default_follow_up is not None
                    completion = default_follow_up(repository, conversation, messages, claimed)
                reply, model_run = completion[0], completion[1]
                if len(completion) == 3:
                    directive = completion[2]
                if not reply.strip() or len(reply) > 1200:
                    raise ConversationProviderError("Invalid activity follow-up")
        except (ConversationProviderError, UnsupportedProviderResponse):
            translated(
                lambda: repository.finish_activity_follow_up(
                    auth.user_id,
                    session_id,
                    request_id=payload.client_request_id,
                    expected_revision=claimed.revision,
                    expected_conversation_revision=generation_revision,
                    expected_session_created_at=claimed.created_at,
                    expected_conversation_incarnation_id=conversation.incarnation_id,
                    reply=None,
                    model_run=None,
                    now=clock(),
                )
            )
            raise HTTPException(
                503, "Your check-in is saved. Luna could not reply; you can retry the follow-up."
            ) from None
        # Re-read source identity after generation too. The conditional transaction
        # is the final authority if deletion/preferences/support raced this check.
        owned_chat(repository, auth, claimed.conversation_id, open_only=True)
        saved = translated(
            lambda: repository.finish_activity_follow_up(
                auth.user_id,
                session_id,
                request_id=payload.client_request_id,
                expected_revision=claimed.revision,
                expected_conversation_revision=generation_revision,
                expected_session_created_at=claimed.created_at,
                expected_conversation_incarnation_id=conversation.incarnation_id,
                reply=reply,
                model_run=model_run,
                now=clock(),
                directive=directive,
            )
        )
        return view(repository, auth, saved)
