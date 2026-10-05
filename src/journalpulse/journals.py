"""Save writing without a task, then explicitly request a transient AI reflection."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from uuid import UUID, uuid4

from fastapi import Depends, FastAPI, HTTPException, Query, Response
from pydantic import ValidationError

from .auth import AuthContext
from .config import Settings
from .conversations import ConversationClient, GenerationLimit
from .domain import GenerationErrorResponse, ModelRun, SafetyMode
from .intelligence import (
    CONVERSATION_PROMPT_VERSION,
    ConversationProviderError,
    OpenRouterConversationClient,
    UnsupportedProviderResponse,
)
from .journal_models import (
    CreateJournalEntryRequest,
    JournalEntry,
    JournalEntryPage,
    JournalReflectionResult,
    ReflectJournalEntryRequest,
)
from .persistence import Repository
from .reflection_prompts import JOURNAL_REFLECTION_INSTRUCTION, REFLECTION_SKILL_VERSION, journal_data_message
from .safety import SUPPORT_FALLBACK_MESSAGE, assess_safety


def register_journal_routes(
    app: FastAPI,
    *,
    settings: Settings,
    repositories: Callable[[AuthContext], Repository],
    enforce_generation_limit: GenerationLimit,
    auth_dependency: Callable[..., AuthContext],
    conversation_client: ConversationClient | None,
    clock: Callable[[], datetime],
) -> None:
    """Register owner-scoped CRUD and opt-in reflection using the shared model limiter.

    Entries retain text because saving writing is the requested behavior. AI consent
    is checked separately on each reflection request; generated replies are transient.
    """

    def require_owned(repository: Repository, auth: AuthContext, entry_id: UUID) -> JournalEntry:
        entry = repository.get_journal_entry(auth.user_id, entry_id)
        if entry is None:
            raise HTTPException(status_code=404, detail="Journal entry not found")
        return entry

    def private_response(response: Response) -> None:
        response.headers["Cache-Control"] = "no-store"

    @app.post(
        "/v1/journal/entries",
        response_model=JournalEntry,
        status_code=201,
        dependencies=[Depends(private_response)],
    )
    def create_entry(
        payload: CreateJournalEntryRequest,
        auth: AuthContext = Depends(auth_dependency),
    ) -> JournalEntry:
        entry = JournalEntry(
            id=payload.client_request_id or uuid4(),
            user_id=auth.user_id,
            created_at=clock(),
            text=payload.text,
        )
        try:
            return repositories(auth).save_journal_entry(entry)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail="Journal request ID is already in use") from exc

    @app.get("/v1/journal/entries", response_model=JournalEntryPage, dependencies=[Depends(private_response)])
    def list_entries(
        limit: int = Query(default=50, ge=1, le=100),
        offset: int = Query(default=0, ge=0),
        auth: AuthContext = Depends(auth_dependency),
    ) -> JournalEntryPage:
        return JournalEntryPage(
            items=repositories(auth).list_journal_entries(auth.user_id, limit=limit, offset=offset),
            limit=limit,
            offset=offset,
        )

    @app.get(
        "/v1/journal/entries/{entry_id}",
        response_model=JournalEntry,
        dependencies=[Depends(private_response)],
    )
    def read_entry(entry_id: UUID, auth: AuthContext = Depends(auth_dependency)) -> JournalEntry:
        return require_owned(repositories(auth), auth, entry_id)

    @app.delete("/v1/journal/entries/{entry_id}", status_code=204, dependencies=[Depends(private_response)])
    def delete_entry(entry_id: UUID, auth: AuthContext = Depends(auth_dependency)) -> None:
        if not repositories(auth).delete_journal_entry(auth.user_id, entry_id):
            raise HTTPException(status_code=404, detail="Journal entry not found")

    @app.post(
        "/v1/journal/entries/{entry_id}/reflect",
        response_model=JournalReflectionResult,
        responses={422: {
            "model": GenerationErrorResponse, "description": "Invalid input or provider decline",
        }},
        dependencies=[Depends(private_response)],
    )
    def reflect_entry(
        entry_id: UUID,
        payload: ReflectJournalEntryRequest,
        auth: AuthContext = Depends(auth_dependency),
    ) -> JournalReflectionResult:
        repository = repositories(auth)
        entry = require_owned(repository, auth, entry_id)
        if not payload.llm_consent:
            raise HTTPException(
                status_code=409, detail="Allow AI for this entry before requesting a reflection."
            )
        safety = assess_safety(entry.text, payload.locale)
        if safety.mode == SafetyMode.SUPPORT:
            # The safety router takes precedence even when a provider is unavailable.
            # This is labelled support information, never presented as a model reply.
            return JournalReflectionResult(
                entry_id=entry.id,
                reply=safety.support_message or SUPPORT_FALLBACK_MESSAGE,
                safety=safety,
                model_run=ModelRun(
                    model="safety-router",
                    provider="safety-router",
                    latency_ms=0,
                    schema_valid=True,
                    used_fallback=True,
                    fallback_reason="support_mode_llm_bypassed",
                    prompt_version=REFLECTION_SKILL_VERSION,
                ),
            )
        if not settings.openrouter_enabled or not settings.chat_model or not settings.openrouter_zdr:
            raise HTTPException(
                status_code=503,
                detail="AI reflection is unavailable. Your entry is saved and you can return to it anytime.",
            )
        enforce_generation_limit(auth, repository)
        client = conversation_client or OpenRouterConversationClient(settings)
        try:
            completion = client.complete(
                [
                    {"role": "system", "content": JOURNAL_REFLECTION_INSTRUCTION},
                    # JSON escaping keeps embedded quotes/markers inside the data
                    # value. The model still needs instructions and strict validation;
                    # a data container alone is not a prompt-injection guarantee.
                    journal_data_message(entry.text),
                ]
            )
            if (
                completion.offer_action
                or completion.model_run.used_fallback
                or not completion.model_run.schema_valid
            ):
                raise ValueError("Unsupported journal reflection output")
            result = JournalReflectionResult(
                entry_id=entry.id,
                reply=completion.reply,
                safety=safety,
                model_run=completion.model_run.model_copy(
                    update={"prompt_version": (
                        f"{completion.model_run.prompt_version or CONVERSATION_PROMPT_VERSION}"
                        f"+{REFLECTION_SKILL_VERSION}"
                    )}
                ),
            )
        except UnsupportedProviderResponse as exc:
            raise HTTPException(
                status_code=502,
                detail="AI reflection could not be shown. Your entry is still saved. Please try again.",
                headers={"X-JournalPulse-Error-Stage": "content_format"},
            ) from exc
        except ConversationProviderError as exc:
            # The entry was saved before this optional request. Chat-oriented
            # provider copy saying 'Nothing was saved' would misrepresent it.
            declined = exc.diagnostic is not None and exc.diagnostic.stage == "provider_refusal"
            detail = (
                "Luna's AI provider declined this reflection. Your entry is still saved. "
                "You can keep journaling without an AI reply."
                if declined else
                "AI reflection could not be generated. Your entry is still saved. Please try again."
            )
            raise HTTPException(
                status_code=exc.status_code,
                detail=detail,
                headers=exc.diagnostic_headers,
            ) from exc
        except (ValueError, ValidationError) as exc:
            raise HTTPException(
                status_code=502,
                detail="AI reflection could not be shown. Your entry is still saved. Please try again.",
            ) from exc
        # A deletion during generation invalidates the pending response. No derived
        # text is persisted, and the browser receives no delayed reflection.
        if require_owned(repository, auth, entry_id) != entry:
            raise HTTPException(status_code=404, detail="Journal entry not found")
        return result
