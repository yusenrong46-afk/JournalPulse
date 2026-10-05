from __future__ import annotations

import threading
from collections.abc import Callable
from datetime import datetime, timedelta
from typing import Protocol
from uuid import UUID, uuid4

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel

from .auth import AuthContext
from .config import Settings
from .domain import (
    AcceptConversationRequest,
    ActionCard,
    AffectiveState,
    ChangeConversationPreferenceRequest,
    Conversation,
    ConversationMessage,
    ConversationMode,
    ConversationRequestInputs,
    ConversationStatus,
    ConversationTurnRequest,
    GenerationErrorResponse,
    Goal,
    InteractionPreference,
    MessageRole,
    ModelRun,
    PolicyDecision,
    ReflectionCopy,
    ReflectionRecord,
    SafetyMode,
    SafetyResult,
    SelectionSource,
    SelfReportInput,
    StartConversationRequest,
    TargetState,
)
from .guided import (
    GUIDED_PROMPT_VERSION,
    GUIDED_SUMMARY,
    goal_card_reason,
    goal_reply,
    guided_completion,
    guided_model_run,
)
from .intelligence import (
    CONVERSATION_PROMPT_VERSION,
    ConversationCompletion,
    ConversationProviderError,
    OpenRouterConversationClient,
    UnsupportedProviderResponse,
)
from .journal_models import JournalEntry
from .persistence import (
    ConversationAlreadyAccepted,
    ConversationClosed,
    ConversationNotFound,
    ConversationStale,
    JournalEntryNotFound,
    Repository,
    require_same_turn_request,
)
from .policy import ReflectionPolicy, apply_user_choice
from .reflection_prompts import (
    JOURNAL_CONTEXT_INSTRUCTION,
    LISTEN_CONTEXT_INSTRUCTION,
    REFLECTION_SKILL_VERSION,
)
from .resources import action_intent, approved_actions, goal_for_intent
from .safety import SUPPORT_FALLBACK_MESSAGE, assess_safety
from .self_report import derive_state

MAX_USER_MESSAGES = 20
PREVIEW_STATE = AffectiveState(
    valence=0.0,
    arousal=0.5,
    agency=0.5,
    emotion_tags=[],
    confidence=None,
    uncertainty="A self-report is collected when an action is accepted.",
)
STALE_TURN = "This chat changed while Luna was replying, so that reply was not saved. Please send it again."
STALE_ACCEPT = "This chat changed before your choice was saved. Please look at the options again."
CLOSED = "This conversation is closed."
ALREADY_ACCEPTED = "This conversation already has a saved choice."


class ConversationClient(Protocol):
    def complete(self, messages: list[dict[str, str]]) -> ConversationCompletion: ...


class ConversationDetail(BaseModel):
    conversation: Conversation
    messages: list[ConversationMessage]


class ConversationTurnResult(BaseModel):
    conversation: Conversation
    user_message: ConversationMessage
    assistant_message: ConversationMessage


GenerationLimit = Callable[[AuthContext, Repository], None]


def register_conversation_routes(
    app: FastAPI,
    *,
    settings: Settings,
    repositories: Callable[[AuthContext], Repository],
    policy: ReflectionPolicy,
    enforce_generation_limit: GenerationLimit,
    auth_dependency: Callable[..., AuthContext],
    conversation_client: ConversationClient | None,
    clock: Callable[[], datetime],
) -> None:
    # Fast rejection of a second concurrent reply within one process. Correctness does
    # not depend on it: the database commit is conditional on the revision.
    active_turns: set[UUID] = set()
    lock_guard = threading.Lock()

    def sweep(auth: AuthContext) -> Repository:
        repository = repositories(auth)
        repository.close_stale_conversations(auth.user_id, now=clock())
        return repository

    def require_owned(repository: Repository, auth: AuthContext, conversation_id: UUID) -> Conversation:
        conversation = repository.get_conversation(auth.user_id, conversation_id)
        if conversation is None:
            raise HTTPException(status_code=404, detail="Conversation not found")
        return conversation

    def acquire(conversation_id: UUID) -> None:
        with lock_guard:
            if conversation_id in active_turns:
                raise HTTPException(
                    status_code=409,
                    detail="A reply is already in progress for this conversation.",
                )
            # Track only requests currently running. Keeping one lock per seen ID
            # leaks memory, including for guesses that immediately return HTTP 404.
            active_turns.add(conversation_id)

    @app.post("/v1/conversations", response_model=Conversation, status_code=201)
    def start_conversation(
        payload: StartConversationRequest,
        auth: AuthContext = Depends(auth_dependency),
    ) -> Conversation:
        repository = sweep(auth)
        use_model = payload.llm_consent and settings.openrouter_enabled
        source_entry = _require_source(repository, auth.user_id, payload.source_entry_id)
        if source_entry is not None and (not use_model or not settings.openrouter_zdr):
            raise HTTPException(
                status_code=409,
                detail=(
                    "Using a journal entry in chat requires AI consent and private AI help "
                    "with provider data retention disabled. "
                    "You can start a guided chat without sending the entry."
                ),
            )
        moment = clock()
        conversation = Conversation(
            id=payload.client_request_id or uuid4(),
            user_id=auth.user_id,
            created_at=moment,
            updated_at=moment,
            llm_consent=payload.llm_consent,
            source_entry_id=source_entry.id if source_entry is not None else None,
            source_entry_created_at=source_entry.created_at if source_entry is not None else None,
            mode=ConversationMode.AI if use_model else ConversationMode.GUIDED,
            retain_text=payload.retain_text,
            locale=payload.locale.upper(),
            prompt_version=(
                f"{CONVERSATION_PROMPT_VERSION}+{REFLECTION_SKILL_VERSION}"
                if source_entry is not None
                else CONVERSATION_PROMPT_VERSION if use_model else GUIDED_PROMPT_VERSION
            ),
            safety=SafetyResult(
                mode=SafetyMode.NORMAL,
                locale=payload.locale.upper(),
                exploration_allowed=True,
            ),
        )
        try:
            return repository.create_conversation(conversation)
        except JournalEntryNotFound as exc:
            raise HTTPException(status_code=404, detail="Journal entry not found") from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail="Conversation request ID is already in use") from exc

    @app.get("/v1/conversations/{conversation_id}", response_model=ConversationDetail)
    def read_conversation(
        conversation_id: UUID,
        auth: AuthContext = Depends(auth_dependency),
    ) -> ConversationDetail:
        repository = sweep(auth)
        conversation = require_owned(repository, auth, conversation_id)
        return ConversationDetail(
            conversation=conversation,
            messages=repository.list_messages(auth.user_id, conversation_id),
        )

    @app.post(
        "/v1/conversations/{conversation_id}/messages", response_model=ConversationTurnResult,
        responses={422: {
            "model": GenerationErrorResponse, "description": "Invalid input or provider decline",
        }},
    )
    def continue_conversation(
        conversation_id: UUID,
        payload: ConversationTurnRequest,
        auth: AuthContext = Depends(auth_dependency),
    ) -> ConversationTurnResult:
        repository = sweep(auth)
        acquire(conversation_id)
        try:
            conversation = require_owned(repository, auth, conversation_id)
            messages = repository.list_messages(auth.user_id, conversation_id)
            stored = _stored_turn(messages, payload.client_message_id)
            if stored is not None:
                try:
                    require_same_turn_request(
                        stored[0], text=payload.text, inputs=_request_inputs(payload),
                    )
                except ValueError as exc:
                    raise HTTPException(status_code=409, detail=str(exc)) from exc
                return ConversationTurnResult(
                    conversation=conversation, user_message=stored[0], assistant_message=stored[1]
                )
            if conversation.status != ConversationStatus.OPEN:
                raise HTTPException(status_code=409, detail=CLOSED)
            user_count = sum(message.role == MessageRole.USER for message in messages)
            if user_count >= MAX_USER_MESSAGES:
                raise HTTPException(
                    status_code=409,
                    detail="This conversation has reached its 20-message limit.",
                )
            return _take_turn(
                settings=settings,
                repository=repository,
                policy=policy,
                conversation=conversation,
                messages=messages,
                payload=payload,
                auth=auth,
                moment=clock(),
                enforce_generation_limit=enforce_generation_limit,
                conversation_client=conversation_client,
            )
        finally:
            with lock_guard:
                active_turns.discard(conversation_id)

    @app.post("/v1/conversations/{conversation_id}/accept", response_model=ReflectionRecord, status_code=201)
    def accept_card(
        conversation_id: UUID,
        payload: AcceptConversationRequest,
        auth: AuthContext = Depends(auth_dependency),
    ) -> ReflectionRecord:
        repository = sweep(auth)
        conversation = require_owned(repository, auth, conversation_id)
        if conversation.reflection_id is not None:
            return _already_accepted(repository, auth, conversation, payload)
        if conversation.status != ConversationStatus.OPEN:
            raise HTTPException(status_code=409, detail=CLOSED)
        if (
            payload.expected_revision is not None and payload.expected_revision != conversation.revision
        ) or (
            conversation.interaction_preference != InteractionPreference.AUTO
            and payload.expected_revision is None
        ):
            # Older clients are compatible with automatic chats, but cannot accept a
            # card after explicit preference changes without identifying its revision.
            raise HTTPException(status_code=409, detail=STALE_ACCEPT)
        if (
            conversation.interaction_preference == InteractionPreference.LISTEN
            and conversation.safety_mode != SafetyMode.SUPPORT
        ):
            raise HTTPException(status_code=409, detail="Please request a new action before choosing one.")
        if conversation.activity_card is not None:
            raise HTTPException(
                status_code=409,
                detail="This Luna activity stays in your chat. Use Start activity to try it.",
            )
        if conversation.card is None:
            raise HTTPException(status_code=409, detail="There is no action card to accept yet.")
        if payload.action_id not in conversation.card.decision_preview.safe_action_ids:
            raise HTTPException(status_code=422, detail="Chosen action is not in the safe set")
        messages = repository.list_messages(auth.user_id, conversation_id)
        record = _reflection_from_card(
            settings=settings,
            policy=policy,
            conversation=conversation,
            messages=messages,
            payload=payload,
            user_id=auth.user_id,
        )
        try:
            return repository.accept_conversation(
                auth.user_id, conversation.id, record, expected_revision=conversation.revision
            )
        except ConversationAlreadyAccepted:
            current = require_owned(repository, auth, conversation_id)
            return _already_accepted(repository, auth, current, payload)
        except ConversationStale as exc:
            raise HTTPException(status_code=409, detail=STALE_ACCEPT) from exc
        except ConversationClosed as exc:
            raise HTTPException(status_code=409, detail=CLOSED) from exc
        except ConversationNotFound as exc:
            raise HTTPException(status_code=404, detail="Conversation not found") from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail="Reflection request ID is already in use") from exc

    @app.post("/v1/conversations/{conversation_id}/preference", response_model=Conversation)
    def change_preference(
        conversation_id: UUID,
        payload: ChangeConversationPreferenceRequest,
        auth: AuthContext = Depends(auth_dependency),
    ) -> Conversation:
        # This route deliberately avoids the generation lock: changing preference
        # advances the database revision and invalidates an in-flight reply.
        repository = sweep(auth)
        try:
            return repository.change_preference(
                auth.user_id, conversation_id, request_id=payload.client_request_id,
                preference=InteractionPreference(payload.preference),
                expected_revision=payload.expected_revision, now=clock(),
            )
        except ConversationNotFound as exc:
            raise HTTPException(status_code=404, detail="Conversation not found") from exc
        except ConversationClosed as exc:
            raise HTTPException(status_code=409, detail=CLOSED) from exc
        except ConversationStale as exc:
            raise HTTPException(
                status_code=409, detail="This chat changed. Please reload and try again."
            ) from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/v1/conversations/{conversation_id}/close", response_model=Conversation)
    def close_conversation(
        conversation_id: UUID,
        auth: AuthContext = Depends(auth_dependency),
    ) -> Conversation:
        repository = sweep(auth)
        closed = repository.close_conversation(auth.user_id, conversation_id)
        if closed is None:
            raise HTTPException(status_code=404, detail="Conversation not found")
        return closed

    @app.delete("/v1/conversations/{conversation_id}", status_code=204)
    def delete_conversation(
        conversation_id: UUID,
        auth: AuthContext = Depends(auth_dependency),
    ) -> None:
        repository = sweep(auth)
        if not repository.delete_conversation(auth.user_id, conversation_id):
            raise HTTPException(status_code=404, detail="Conversation not found")


def _already_accepted(
    repository: Repository,
    auth: AuthContext,
    conversation: Conversation,
    payload: AcceptConversationRequest,
) -> ReflectionRecord:
    """Idempotent retry of the same request returns the saved record; any other is refused."""
    assert conversation.reflection_id is not None
    if payload.client_request_id is not None and payload.client_request_id == conversation.reflection_id:
        saved = repository.get_reflection(auth.user_id, conversation.reflection_id)
        if saved is not None:
            state, _ = _self_report(conversation, payload)
            if (
                saved.context.get("conversation_id") != str(conversation.id)
                or saved.decision.action_id != payload.action_id
                or saved.state != state
                or (
                    payload.expected_revision is not None
                    and payload.expected_revision != conversation.revision - 1
                )
            ):
                raise HTTPException(status_code=409, detail="Reflection request ID is already in use")
            return saved
    raise HTTPException(status_code=409, detail=ALREADY_ACCEPTED)


def _stored_turn(
    messages: list[ConversationMessage], client_message_id: UUID
) -> tuple[ConversationMessage, ConversationMessage] | None:
    user_message = next(
        (message for message in messages if message.client_message_id == client_message_id),
        None,
    )
    if user_message is None:
        return None
    assistant_message = next(
        (
            message
            for message in messages
            if message.role == MessageRole.ASSISTANT and message.created_at >= user_message.created_at
        ),
        None,
    )
    if assistant_message is None:
        return None
    return user_message, assistant_message


def _take_turn(
    *,
    settings: Settings,
    repository: Repository,
    policy: ReflectionPolicy,
    conversation: Conversation,
    messages: list[ConversationMessage],
    payload: ConversationTurnRequest,
    auth: AuthContext,
    moment: datetime,
    enforce_generation_limit: GenerationLimit,
    conversation_client: ConversationClient | None,
) -> ConversationTurnResult:
    # The turn is computed against this revision and only committed if it still holds.
    expected_revision = conversation.revision
    source_entry = _require_source(
        repository, auth.user_id, conversation.source_entry_id,
        expected_created_at=conversation.source_entry_created_at,
    )
    reported = _with_reported_inputs(conversation, payload)
    # A clause boundary prevents a negation in the current message from suppressing
    # risk in the selected entry (or the other way around).
    safety_text = f"{source_entry.text}\n.\n{payload.text}" if source_entry is not None else payload.text
    assessed = assess_safety(safety_text, conversation.locale)
    already_support = conversation.safety_mode == SafetyMode.SUPPORT
    safety = conversation.safety if already_support and conversation.safety is not None else assessed
    entering_support = already_support or safety.mode == SafetyMode.SUPPORT
    user_message = ConversationMessage(
        conversation_id=conversation.id,
        client_message_id=payload.client_message_id,
        role=MessageRole.USER,
        content=payload.text,
        created_at=moment,
        safety_mode=SafetyMode.SUPPORT if entering_support else SafetyMode.NORMAL,
        request_inputs=_request_inputs(payload),
    )
    assistant_created = moment + timedelta(microseconds=1)
    if entering_support:
        model_run = ModelRun(
            model="safety-router",
            provider="safety-router",
            latency_ms=0,
            schema_valid=True,
            used_fallback=True,
            fallback_reason="support_mode_llm_bypassed",
            prompt_version=CONVERSATION_PROMPT_VERSION,
        )
        assistant_message = ConversationMessage(
            conversation_id=conversation.id,
            role=MessageRole.ASSISTANT,
            content=_support_text(safety),
            created_at=assistant_created,
            safety_mode=SafetyMode.SUPPORT,
            model_run=model_run,
        )
        updated = reported.model_copy(
            update={
                "updated_at": assistant_created,
                "safety_mode": SafetyMode.SUPPORT,
                "safety": safety,
                "summary": conversation.summary or "This conversation moved to support mode.",
                "card": _support_card(settings, safety, assistant_message.id),
            }
        )
        return _commit(repository, updated, user_message, assistant_message, expected_revision)

    if payload.goal is not None:
        if conversation.interaction_preference == InteractionPreference.LISTEN:
            raise HTTPException(status_code=409, detail="Please request a new action before choosing a goal.")
        assistant_message = ConversationMessage(
            conversation_id=conversation.id,
            role=MessageRole.ASSISTANT,
            content=goal_reply(payload.goal),
            created_at=assistant_created,
            safety_mode=SafetyMode.NORMAL,
            model_run=guided_model_run("goal_card"),
        )
        card = _catalog_card(
            settings,
            policy,
            intent=action_intent("reflect", payload.goal.value),
            reason=goal_card_reason(payload.goal),
            message_id=assistant_message.id,
            state=PREVIEW_STATE,
            goal=payload.goal,
        )
        updated = reported.model_copy(
            update={
                "updated_at": assistant_created,
                "safety": safety,
                "summary": conversation.summary or GUIDED_SUMMARY,
                "card": card,
                "ready_for_action": True,
            }
        )
        return _commit(repository, updated, user_message, assistant_message, expected_revision)

    if conversation.mode == ConversationMode.GUIDED:
        user_texts = [
            message.content for message in messages if message.role == MessageRole.USER and message.content
        ]
        completion = guided_completion(
            [*user_texts, payload.text],
            listening=conversation.interaction_preference == InteractionPreference.LISTEN,
        )
    else:
        if source_entry is not None and (
            not conversation.llm_consent or not settings.openrouter_enabled or not settings.openrouter_zdr
        ):
            raise HTTPException(
                status_code=409,
                detail=(
                    "AI help is unavailable for this journal-linked chat. "
                    "Start a guided chat without the entry."
                ),
            )
        enforce_generation_limit(auth, repository)
        history = [
            {"role": message.role.value, "content": message.content}
            for message in messages
            if message.content
        ]
        history.append({"role": "user", "content": payload.text})
        if source_entry is not None:
            # The trusted instruction is fixed. Even journal text that resembles
            # instructions remains a separate user message, never system content.
            history[0:0] = [
                {"role": "system", "content": JOURNAL_CONTEXT_INSTRUCTION},
                {"role": "user", "content": (
                    f"Selected journal entry {source_entry.id}, "
                    f"saved {source_entry.created_at.isoformat()}.\n"
                    f"Journal context (user data):\n{source_entry.text}"
                )},
            ]
        if conversation.interaction_preference == InteractionPreference.LISTEN:
            # Only the server's validated enum controls this instruction. Journal
            # text remains user data and cannot become a system instruction.
            history.insert(0, {"role": "system", "content": LISTEN_CONTEXT_INSTRUCTION})
        try:
            from .activity_chat import chat_activity_context, complete_chat

            activity_context = chat_activity_context(settings, repository, conversation)
            completion = complete_chat(_client(settings, conversation_client), history, activity_context)
        except UnsupportedProviderResponse as exc:
            raise HTTPException(
                status_code=502,
                detail="The model response was not a supported text format.",
                headers={"X-JournalPulse-Error-Stage": "content_format"},
            ) from exc
        except ConversationProviderError as exc:
            raise HTTPException(
                status_code=exc.status_code, detail=str(exc), headers=exc.diagnostic_headers,
            ) from exc
    assistant_message = ConversationMessage(
        conversation_id=conversation.id,
        role=MessageRole.ASSISTANT,
        content=completion.reply,
        created_at=assistant_created,
        safety_mode=SafetyMode.NORMAL,
        model_run=(
            completion.model_run.model_copy(update={
                # A resumed chat uses today's instructions, even if its creation
                # record still describes an older release.
                "prompt_version": (
                    f"{completion.model_run.prompt_version or CONVERSATION_PROMPT_VERSION}"
                    f"+{REFLECTION_SKILL_VERSION}"
                ),
            })
            if source_entry is not None else completion.model_run
        ),
    )
    updated = reported.model_copy(
        update={
            "updated_at": assistant_created,
            "safety": safety,
            "summary": completion.summary,
            "feelings": list(completion.feelings),
            "ready_for_action": (
                conversation.interaction_preference != InteractionPreference.LISTEN
                # Readiness describes this turn. A previous invitation must not
                # survive a correction, refusal or request to keep reflecting.
                and completion.offer_action
            ),
        }
    )
    if conversation.mode == ConversationMode.AI:
        from .activity_chat import validated_activity_update

        # Resolve model IDs against this turn's server candidates before saving.
        # Existing injected clients with no activity directive keep their contract.
        update = validated_activity_update(completion, activity_context, assistant_message.id)
        if update:
            updated = updated.model_copy(update=update)
    return _commit(repository, updated, user_message, assistant_message, expected_revision)


def _request_inputs(payload: ConversationTurnRequest) -> ConversationRequestInputs:
    return ConversationRequestInputs(
        goal=payload.goal, mood_score=payload.mood_score, confirmed_feelings=payload.confirmed_feelings,
    )


def _with_reported_inputs(conversation: Conversation, payload: ConversationTurnRequest) -> Conversation:
    """Store what the person tapped: the first mood face, and feelings confirmed with a goal."""
    update: dict[str, object] = {}
    if payload.mood_score is not None and conversation.reported_mood is None:
        update["reported_mood"] = payload.mood_score
    if payload.goal is not None and payload.confirmed_feelings is not None:
        update["confirmed_feelings"] = payload.confirmed_feelings
    return conversation.model_copy(update=update) if update else conversation


def _client(settings: Settings, conversation_client: ConversationClient | None) -> ConversationClient:
    if conversation_client is not None:
        return conversation_client
    return OpenRouterConversationClient(settings)


def _require_source(
    repository: Repository, user_id: UUID, entry_id: UUID | None,
    *, expected_created_at: datetime | None = None,
) -> JournalEntry | None:
    """Resolve only the explicitly linked, owned entry; hide absent and foreign IDs alike."""
    if entry_id is None:
        return None
    entry = repository.get_journal_entry(user_id, entry_id)
    if (
        entry is None or entry.user_id != user_id or entry.id != entry_id
        or (expected_created_at is not None and entry.created_at != expected_created_at)
    ):
        raise HTTPException(status_code=404, detail="Journal entry not found")
    return entry


def _commit(
    repository: Repository,
    conversation: Conversation,
    user_message: ConversationMessage,
    assistant_message: ConversationMessage,
    expected_revision: int,
) -> ConversationTurnResult:
    # Source deletion invalidates a pending response. The repository also commits
    # conditionally, so a deletion racing this check cannot recreate derived data.
    _require_source(
        repository, conversation.user_id, conversation.source_entry_id,
        expected_created_at=conversation.source_entry_created_at,
    )
    try:
        stored_conversation, stored_user, stored_assistant = repository.commit_turn(
            conversation, user_message, assistant_message, expected_revision=expected_revision
        )
    except ConversationStale as exc:
        raise HTTPException(status_code=409, detail=STALE_TURN) from exc
    except ConversationClosed as exc:
        raise HTTPException(status_code=409, detail=CLOSED) from exc
    except ConversationNotFound as exc:
        raise HTTPException(status_code=404, detail="Conversation not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return ConversationTurnResult(
        conversation=stored_conversation,
        user_message=stored_user,
        assistant_message=stored_assistant,
    )


def _support_text(safety: SafetyResult) -> str:
    """Every support-mode reply is the support message recorded when the chat entered
    support mode. It never reuses an earlier ordinary Luna reply."""
    return safety.support_message or SUPPORT_FALLBACK_MESSAGE


def _support_card(settings: Settings, safety: SafetyResult, message_id: UUID) -> ActionCard:
    actions = approved_actions(
        settings.resource_catalog_path,
        intent="pause",
        support_ids=safety.resource_ids,
    )[:3]
    safe_ids = [item["id"] for item in actions] or ["contact-local-support"]
    decision = PolicyDecision(
        action_id=safe_ids[0],
        propensity=1.0,
        policy_name="safety-router",
        policy_version="1.0.0",
        safe_action_ids=safe_ids,
        context_snapshot={"safety_mode": True, "locale": safety.locale},
        explanation="Support mode disables adaptive exploration and prioritizes human help.",
    )
    return ActionCard(
        resource_intent="pause",
        card_reason="Human support comes before another journaling turn.",
        decision_preview=decision,
        actions=actions,
        offered_message_id=message_id,
    )


def _catalog_card(
    settings: Settings,
    policy: ReflectionPolicy,
    *,
    intent: str,
    reason: str,
    message_id: UUID,
    state: AffectiveState,
    goal: Goal | None = None,
) -> ActionCard:
    actions = approved_actions(settings.resource_catalog_path, intent=intent)
    decision = policy.decide(
        state=state,
        target=TargetState(goal=goal.value if goal else goal_for_intent(intent)),
        actions=actions,
        context={"source": "conversation"},
    )
    visible_ids = decision.safe_action_ids[:3]
    visible = [item for item in actions if item["id"] in set(visible_ids)]
    if visible_ids and decision.action_id not in visible_ids:
        decision = decision.model_copy(update={"action_id": visible_ids[0]})
    decision = decision.model_copy(update={"safe_action_ids": visible_ids or decision.safe_action_ids})
    return ActionCard(
        resource_intent=intent,
        card_reason=reason,
        decision_preview=decision,
        actions=visible[:3],
        offered_message_id=message_id,
        goal=goal,
    )


def _self_report(
    conversation: Conversation, payload: AcceptConversationRequest
) -> tuple[AffectiveState, SelfReportInput | None]:
    """The person's confirmed taps win. A client-computed state is only accepted from
    older clients that never sent confirmed feelings."""
    if conversation.confirmed_feelings is not None:
        report = SelfReportInput(
            feelings=conversation.confirmed_feelings, mood_score=conversation.reported_mood
        )
        return derive_state(report), report
    if payload.self_report is not None:
        return payload.self_report, None
    report = SelfReportInput(feelings=[], mood_score=conversation.reported_mood)
    return derive_state(report), report


def _reflection_from_card(
    *,
    settings: Settings,
    policy: ReflectionPolicy,
    conversation: Conversation,
    messages: list[ConversationMessage],
    payload: AcceptConversationRequest,
    user_id: UUID,
) -> ReflectionRecord:
    card = conversation.card
    assert card is not None
    state, report = _self_report(conversation, payload)
    context = {"source": "conversation", "conversation_id": str(conversation.id)}
    target = TargetState(goal=card.goal.value if card.goal else goal_for_intent(card.resource_intent))
    if conversation.safety_mode == SafetyMode.SUPPORT:
        decision = _accept_support_choice(card.decision_preview, payload.action_id)
    else:
        actions = approved_actions(settings.resource_catalog_path, intent=card.resource_intent)
        decision = policy.decide(state=state, target=target, actions=actions, context=context)
        try:
            decision = apply_user_choice(decision, payload.action_id)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail="Chosen action is not in the safe set") from exc
    offered = next((item for item in messages if item.id == card.offered_message_id), None)
    model_run = offered.model_run if offered is not None else None
    # A goal card is built locally; credit the model that actually heard the person.
    model_turns = [
        item.model_run
        for item in messages
        if item.role == MessageRole.ASSISTANT
        and item.model_run is not None
        and item.model_run.provider != "local"
        and item.model_run.model != "safety-router"
    ]
    if model_run is not None and model_run.provider == "local" and model_turns:
        model_run = model_turns[-1]
    safety = conversation.safety or SafetyResult(
        mode=conversation.safety_mode,
        locale=conversation.locale,
        exploration_allowed=conversation.safety_mode == SafetyMode.NORMAL,
    )
    return ReflectionRecord(
        id=payload.client_request_id or uuid4(),
        user_id=user_id,
        text=None,
        text_retained=False,
        context=context,
        state=state,
        target=target,
        reflection=ReflectionCopy(
            summary=conversation.summary or card.card_reason,
            interpretation=card.card_reason,
            reflection_question="What changed after you tried it?",
        ),
        safety=safety,
        decision=decision,
        model_run=model_run,
        self_report_input=report,
    )


def _accept_support_choice(preview: PolicyDecision, action_id: str) -> PolicyDecision:
    decision = preview.model_copy(update={"decision_id": uuid4()})
    if action_id == decision.action_id:
        return decision.model_copy(
            update={
                "recommended_action_id": decision.action_id,
                "selection_source": SelectionSource.POLICY_ACCEPTED,
            }
        )
    return decision.model_copy(
        update={
            "action_id": action_id,
            "recommended_action_id": preview.action_id,
            "selection_source": SelectionSource.USER_OVERRIDE,
            "eligible_for_ope": False,
        }
    )
