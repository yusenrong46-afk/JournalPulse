from __future__ import annotations

import os
from collections import Counter, defaultdict
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.middleware.gzip import GZipMiddleware

from .auth import AuthContext, resolve_auth
from .config import Settings, load_settings
from .conversations import ConversationClient, register_conversation_routes
from .domain import (
    ActionPreview,
    ActionPreviewRequest,
    AffectiveState,
    AnalysisRequest,
    ModelRun,
    OutcomeRecord,
    OutcomeRequest,
    PolicyDecision,
    PreparedAnalysis,
    ReflectionCopy,
    ReflectionRecord,
    ReflectionRequest,
    SafetyMode,
)
from .intelligence import OpenRouterReflectionClient, UnsupportedProviderResponse, safe_analyze
from .middleware import RequestContextMiddleware, SlidingWindowRateLimiter
from .persistence import DuplicateOutcomeError, Repository, SQLiteRepository, SupabaseRepository
from .policy import FixedBaselinePolicy, ReflectionPolicy, apply_user_choice
from .resources import action_intent, approved_actions, load_catalog
from .safety import assess_safety


class ReflectionPage(BaseModel):
    items: list[ReflectionRecord]
    limit: int
    offset: int


class OutcomePage(BaseModel):
    items: list[OutcomeRecord]


class StatePoint(BaseModel):
    reflection_id: UUID
    created_at: str
    valence: float
    arousal: float
    agency: float


class InsightsResponse(BaseModel):
    reflection_count: int
    completed_outcomes: int
    action_counts: dict[str, int]
    average_helpfulness_by_action: dict[str, float]
    average_state_change: dict[str, float] | None
    completion_rate: float
    pending_decision_ids: list[UUID]
    state_trajectory: list[StatePoint]
    note: str = "These are descriptive personal patterns, not causal or clinical conclusions."


class ReadinessResponse(BaseModel):
    status: str
    checks: dict[str, str]


class DeletionResponse(BaseModel):
    deleted_records: int
    auth_identity_deleted: bool = False
    note: str = "Journal data was deleted. Your sign-in identity remains active."


class SystemStatusResponse(BaseModel):
    analysis_mode: str
    persistence_mode: str
    message: str


def create_app(
    *,
    settings: Settings | None = None,
    repository_factory: Callable[[AuthContext], Repository] | None = None,
    policy: ReflectionPolicy | None = None,
    intelligence_client: OpenRouterReflectionClient | None = None,
    conversation_client: ConversationClient | None = None,
    clock: Callable[[], datetime] | None = None,
) -> FastAPI:
    settings = settings or load_settings()
    policy = policy or FixedBaselinePolicy()
    analysis_limiter = SlidingWindowRateLimiter(limit=settings.analysis_rate_limit_per_minute)

    def default_repository_factory(auth: AuthContext) -> Repository:
        if settings.supabase_enabled and auth.access_token:
            return SupabaseRepository(settings, auth.access_token)
        return SQLiteRepository(settings.database_path)

    repositories = repository_factory or default_repository_factory
    app = FastAPI(title="JournalPulse Research Beta API", version="1.0.0")
    app.add_middleware(RequestContextMiddleware, max_request_bytes=settings.max_request_bytes)
    app.add_middleware(GZipMiddleware, minimum_size=1000, compresslevel=6)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_origins),
        allow_credentials=True,
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type", "X-JournalPulse-User", "X-Request-ID"],
        expose_headers=["X-Request-ID"],
    )

    def enforce_generation_limit(user_id: UUID) -> None:
        allowed, retry_after = analysis_limiter.check(str(user_id))
        if not allowed:
            raise HTTPException(
                status_code=429,
                detail="Please wait before requesting another analysis",
                headers={"Retry-After": str(retry_after)},
            )

    def auth_dependency(
        authorization: str | None = Header(default=None),
        development_user: str | None = Header(default=None, alias="X-JournalPulse-User"),
    ) -> AuthContext:
        return resolve_auth(
            settings,
            authorization=authorization,
            development_user=development_user,
        )

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.get("/ready", response_model=ReadinessResponse)
    def ready(response: Response) -> ReadinessResponse:
        checks: dict[str, str] = {}
        issues = settings.configuration_issues
        checks["configuration"] = "ready" if not issues else f"not_ready:{','.join(issues)}"
        try:
            load_catalog(settings.resource_catalog_path)
            checks["resources"] = "ready"
        except Exception as exc:
            checks["resources"] = f"not_ready:{exc.__class__.__name__}"
        if not settings.llm_feature_enabled:
            checks["llm"] = "disabled"
        elif settings.openrouter_enabled:
            checks["llm"] = "configured:not_probed"
        else:
            checks["llm"] = "not_ready:not_configured"
        checks["persistence"] = "supabase" if settings.supabase_enabled else "server_sqlite"
        required_ready = (
            checks["configuration"] == "ready"
            and checks["resources"] == "ready"
            and checks["llm"] != "not_ready:not_configured"
        )
        if settings.environment == "production":
            required_ready = required_ready and settings.supabase_enabled
        status = "ready" if required_ready else "not_ready"
        if status != "ready":
            response.status_code = 503
        return ReadinessResponse(status=status, checks=checks)

    @app.get("/v1/system/status", response_model=SystemStatusResponse)
    def system_status(auth: AuthContext = Depends(auth_dependency)) -> SystemStatusResponse:
        del auth
        if not settings.llm_feature_enabled:
            analysis_mode = "local_only"
            message = "Private AI analysis is turned off. Your corrections remain the source of truth."
        elif settings.openrouter_enabled:
            analysis_mode = "ai_configured"
            message = "Private AI analysis is configured. A safe local fallback remains available."
        else:
            analysis_mode = "local_fallback"
            message = "AI analysis is unavailable, so entries use the local reflection fallback."
        return SystemStatusResponse(
            analysis_mode=analysis_mode,
            persistence_mode="account" if settings.supabase_enabled else "server_sqlite",
            message=message,
        )

    @app.post("/v1/reflections", response_model=ReflectionRecord, status_code=201)
    def create_reflection(
        payload: ReflectionRequest, auth: AuthContext = Depends(auth_dependency)
    ) -> ReflectionRecord:
        safety = assess_safety(payload.text, payload.locale)
        if safety.mode == SafetyMode.SUPPORT:
            state = payload.self_report or AffectiveState(
                valence=-0.8,
                arousal=0.8,
                agency=0.1,
                emotion_tags=["acute_distress"],
                confidence=1.0,
                uncertainty=None,
            )
            reflection = ReflectionCopy(
                summary="This entry triggered support mode.",
                interpretation="Immediate human support matters more than automated reflection right now.",
                reflection_question="Can you contact a trusted person or crisis service now?",
            )
            model_run = ModelRun(
                model="safety-router",
                latency_ms=0,
                schema_valid=True,
                used_fallback=True,
                fallback_reason="support_mode_llm_bypassed",
            )
            actions = approved_actions(
                settings.resource_catalog_path,
                intent="pause",
                support_ids=safety.resource_ids,
            )
            decision = PolicyDecision(
                action_id=actions[0]["id"] if actions else "contact-local-support",
                propensity=1.0,
                policy_name="safety-router",
                policy_version="1.0.0",
                safe_action_ids=[item["id"] for item in actions] or ["contact-local-support"],
                context_snapshot={"safety_mode": True, "locale": safety.locale},
                explanation="Support mode disables adaptive exploration and prioritizes human help.",
            )
        else:
            prepared = payload.prepared_analysis
            if prepared is None:
                enforce_generation_limit(auth.user_id)
                try:
                    analysis = safe_analyze(
                        settings,
                        text=payload.text,
                        context=payload.context,
                        consent=payload.llm_consent,
                        self_report=payload.self_report,
                        client=intelligence_client,
                    )
                except UnsupportedProviderResponse as exc:
                    raise HTTPException(
                        status_code=502,
                        detail="The model response was not a supported text format.",
                    ) from exc
                prepared = PreparedAnalysis(
                    state=analysis.state,
                    reflection=analysis.reflection,
                    safety=safety,
                    model_run=analysis.model_run,
                    resource_intent=analysis.resource_intent,
                )
                model_run = analysis.model_run
            else:
                # Client echo: keep the corrected copy, but not its model or safety provenance.
                model_run = ModelRun(
                    model="unverified-client-analysis",
                    provider="client",
                    latency_ms=0,
                    schema_valid=False,
                    used_fallback=True,
                    fallback_reason="unverified_client_analysis",
                )
            state = payload.self_report or prepared.state
            reflection = prepared.reflection
            intent = action_intent(prepared.resource_intent, payload.target.goal)
            actions = approved_actions(settings.resource_catalog_path, intent=intent)
            decision = policy.decide(
                state=state,
                target=payload.target,
                actions=actions,
                context=payload.context,
            )
            try:
                decision = apply_user_choice(decision, payload.chosen_action_id)
            except ValueError as exc:
                raise HTTPException(status_code=422, detail="Chosen action is not in the safe set") from exc

        retain_text = (
            payload.retain_text if payload.retain_text is not None else settings.raw_text_retention_default
        )
        record_data = {
            "user_id": auth.user_id,
            "text": payload.text if retain_text else None,
            "text_retained": retain_text,
            "context": payload.context,
            "state": state,
            "target": payload.target,
            "reflection": reflection,
            "safety": safety,
            "decision": decision,
            "model_run": model_run,
        }
        if payload.client_request_id is not None:
            record_data["id"] = payload.client_request_id
        record = ReflectionRecord.model_validate(record_data)
        try:
            return repositories(auth).save_reflection(record)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail="Reflection request ID is already in use") from exc

    @app.post("/v1/reflections/analyze", response_model=PreparedAnalysis)
    def analyze_reflection(
        payload: AnalysisRequest, auth: AuthContext = Depends(auth_dependency)
    ) -> PreparedAnalysis:
        enforce_generation_limit(auth.user_id)
        safety = assess_safety(payload.text, payload.locale)
        if safety.mode == SafetyMode.SUPPORT:
            return PreparedAnalysis(
                state=AffectiveState(
                    valence=-0.8,
                    arousal=0.8,
                    agency=0.1,
                    emotion_tags=["acute_distress"],
                    confidence=1.0,
                ),
                reflection=ReflectionCopy(
                    summary="This entry triggered support mode.",
                    interpretation=(
                        "Immediate human support matters more than automated reflection right now."
                    ),
                    reflection_question="Can you contact a trusted person or crisis service now?",
                ),
                safety=safety,
                model_run=ModelRun(
                    model="safety-router",
                    latency_ms=0,
                    schema_valid=True,
                    used_fallback=True,
                    fallback_reason="support_mode_llm_bypassed",
                ),
                resource_intent="pause",
            )
        try:
            analysis = safe_analyze(
                settings,
                text=payload.text,
                context=payload.context,
                consent=payload.llm_consent,
                self_report=None,
                client=intelligence_client,
            )
        except UnsupportedProviderResponse as exc:
            raise HTTPException(
                status_code=502,
                detail="The model response was not a supported text format.",
            ) from exc
        return PreparedAnalysis(
            state=analysis.state,
            reflection=analysis.reflection,
            safety=safety,
            model_run=analysis.model_run,
            resource_intent=analysis.resource_intent,
        )

    @app.post("/v1/actions/preview", response_model=ActionPreview)
    def preview_actions(
        payload: ActionPreviewRequest, auth: AuthContext = Depends(auth_dependency)
    ) -> ActionPreview:
        del auth
        intent = action_intent(payload.resource_intent, payload.target.goal)
        actions = approved_actions(settings.resource_catalog_path, intent=intent)
        decision = policy.decide(
            state=payload.state,
            target=payload.target,
            actions=actions,
            context=payload.context,
        )
        visible_ids = set(decision.safe_action_ids[:3])
        return ActionPreview(
            decision=decision,
            actions=[item for item in actions if item["id"] in visible_ids],
        )

    @app.post("/v1/outcomes", response_model=OutcomeRecord, status_code=201)
    def create_outcome(
        payload: OutcomeRequest, auth: AuthContext = Depends(auth_dependency)
    ) -> OutcomeRecord:
        record_data = {"user_id": auth.user_id, **payload.model_dump(exclude={"client_request_id"})}
        if payload.client_request_id is not None:
            record_data["id"] = payload.client_request_id
        record = OutcomeRecord.model_validate(record_data)
        try:
            return repositories(auth).save_outcome(record)
        except DuplicateOutcomeError as exc:
            raise HTTPException(status_code=409, detail="Check-in already recorded") from exc
        except ValueError as exc:
            raise HTTPException(status_code=404, detail="Policy decision not found") from exc

    @app.get("/v1/outcomes", response_model=OutcomePage)
    def outcomes(auth: AuthContext = Depends(auth_dependency)) -> OutcomePage:
        return OutcomePage(items=repositories(auth).list_outcomes(auth.user_id))

    @app.get("/v1/reflections", response_model=ReflectionPage)
    def reflections(
        limit: int = Query(default=25, ge=1, le=100),
        offset: int = Query(default=0, ge=0),
        auth: AuthContext = Depends(auth_dependency),
    ) -> ReflectionPage:
        return ReflectionPage(
            items=repositories(auth).list_reflections(auth.user_id, limit=limit, offset=offset),
            limit=limit,
            offset=offset,
        )

    @app.get("/v1/insights", response_model=InsightsResponse)
    def insights(auth: AuthContext = Depends(auth_dependency)) -> InsightsResponse:
        repository = repositories(auth)
        reflections = repository.list_reflections(auth.user_id, limit=10000, offset=0)
        outcomes = repository.list_outcomes(auth.user_id)
        action_by_decision = {str(item.decision.decision_id): item.decision.action_id for item in reflections}
        action_counts = Counter(item.decision.action_id for item in reflections)
        helpfulness: dict[str, list[int]] = defaultdict(list)
        deltas: dict[str, list[float]] = defaultdict(list)
        state_by_decision = {str(item.decision.decision_id): item.state for item in reflections}
        completed_decisions = {str(item.decision_id) for item in outcomes}
        for outcome in outcomes:
            action_id = action_by_decision.get(str(outcome.decision_id), "unknown")
            if outcome.helpfulness is not None:
                helpfulness[action_id].append(outcome.helpfulness)
            before = state_by_decision.get(str(outcome.decision_id))
            if before and outcome.post_state:
                for dimension in ("valence", "arousal", "agency"):
                    deltas[dimension].append(
                        getattr(outcome.post_state, dimension) - getattr(before, dimension)
                    )
        return InsightsResponse(
            reflection_count=len(reflections),
            completed_outcomes=sum(item.completed for item in outcomes),
            action_counts=dict(action_counts),
            average_helpfulness_by_action={
                key: round(sum(values) / len(values), 3) for key, values in helpfulness.items()
            },
            average_state_change=(
                {key: round(sum(values) / len(values), 3) for key, values in deltas.items()}
                if deltas
                else None
            ),
            completion_rate=round(len(completed_decisions) / len(reflections), 3) if reflections else 0.0,
            pending_decision_ids=[
                item.decision.decision_id
                for item in reflections
                if str(item.decision.decision_id) not in completed_decisions
            ],
            state_trajectory=[
                StatePoint(
                    reflection_id=item.id,
                    created_at=item.created_at.isoformat(),
                    valence=item.state.valence,
                    arousal=item.state.arousal,
                    agency=item.state.agency,
                )
                for item in reversed(reflections)
            ],
        )

    @app.get("/v1/resources")
    def resources(auth: AuthContext = Depends(auth_dependency)) -> dict:
        del auth
        return {"items": load_catalog(settings.resource_catalog_path)}

    @app.delete("/v1/reflections/{reflection_id}", status_code=204)
    def delete_reflection(reflection_id: UUID, auth: AuthContext = Depends(auth_dependency)) -> None:
        if not repositories(auth).delete_reflection(auth.user_id, reflection_id):
            raise HTTPException(status_code=404, detail="Reflection not found")

    @app.get("/v1/export")
    def export(auth: AuthContext = Depends(auth_dependency)) -> dict:
        return repositories(auth).export_user_data(auth.user_id)

    @app.delete("/v1/account/data", response_model=DeletionResponse)
    def delete_account_data(auth: AuthContext = Depends(auth_dependency)) -> DeletionResponse:
        deleted = repositories(auth).delete_user_data(auth.user_id)
        return DeletionResponse(deleted_records=deleted)

    register_conversation_routes(
        app,
        settings=settings,
        repositories=repositories,
        policy=policy,
        enforce_generation_limit=enforce_generation_limit,
        auth_dependency=auth_dependency,
        conversation_client=conversation_client,
        clock=clock or (lambda: datetime.now(UTC)),
    )

    web_dist_value = os.getenv("JOURNALPULSE_WEB_DIST", "").strip()
    if web_dist_value:
        web_dist = Path(web_dist_value).expanduser().resolve()
        if web_dist.is_dir():
            app.mount("/", StaticFiles(directory=web_dist, html=True), name="journalpulse-web")

    return app


app = create_app()
