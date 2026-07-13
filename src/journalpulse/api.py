from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable
from uuid import UUID

from fastapi import Depends, FastAPI, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .auth import AuthContext, resolve_auth
from .config import Settings, load_settings
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
    SelectionSource,
)
from .intelligence import OpenRouterReflectionClient, safe_analyze
from .persistence import DuplicateOutcomeError, Repository, SQLiteRepository, SupabaseRepository
from .policy import FixedBaselinePolicy, ReflectionPolicy
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


def create_app(
    *,
    settings: Settings | None = None,
    repository_factory: Callable[[AuthContext], Repository] | None = None,
    policy: ReflectionPolicy | None = None,
    intelligence_client: OpenRouterReflectionClient | None = None,
) -> FastAPI:
    settings = settings or load_settings()
    policy = policy or FixedBaselinePolicy()

    def default_repository_factory(auth: AuthContext) -> Repository:
        if settings.supabase_enabled and auth.access_token:
            return SupabaseRepository(settings, auth.access_token)
        return SQLiteRepository(settings.database_path)

    repositories = repository_factory or default_repository_factory
    app = FastAPI(title="JournalPulse Research Beta API", version="1.0.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
        allow_credentials=True,
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type", "X-JournalPulse-User"],
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
    def ready() -> ReadinessResponse:
        checks: dict[str, str] = {}
        try:
            load_catalog(settings.resource_catalog_path)
            checks["resources"] = "ready"
        except Exception as exc:
            checks["resources"] = f"not_ready:{exc.__class__.__name__}"
        checks["llm"] = "ready" if settings.openrouter_enabled else "optional:not_configured"
        checks["persistence"] = "supabase" if settings.supabase_enabled else "local_sqlite"
        required_ready = checks["resources"] == "ready"
        if settings.environment == "production":
            required_ready = required_ready and settings.openrouter_enabled and settings.supabase_enabled
        return ReadinessResponse(status="ready" if required_ready else "not_ready", checks=checks)

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
                analysis = safe_analyze(
                    settings,
                    text=payload.text,
                    context=payload.context,
                    consent=payload.llm_consent,
                    self_report=payload.self_report,
                    client=intelligence_client,
                )
                prepared = PreparedAnalysis(
                    state=analysis.state,
                    reflection=analysis.reflection,
                    safety=safety,
                    model_run=analysis.model_run,
                    resource_intent=analysis.resource_intent,
                )
            state = payload.self_report or prepared.state
            reflection = prepared.reflection
            model_run = prepared.model_run
            intent = action_intent(prepared.resource_intent, payload.target.goal)
            actions = approved_actions(settings.resource_catalog_path, intent=intent)
            decision = policy.decide(
                state=state,
                target=payload.target,
                actions=actions,
                context=payload.context,
            )
            if payload.chosen_action_id:
                if payload.chosen_action_id not in decision.safe_action_ids:
                    raise HTTPException(status_code=422, detail="Chosen action is not in the safe set")
                recommended = decision.action_id
                if payload.chosen_action_id == recommended:
                    decision = decision.model_copy(
                        update={
                            "recommended_action_id": recommended,
                            "selection_source": SelectionSource.POLICY_ACCEPTED,
                        }
                    )
                else:
                    decision = decision.model_copy(
                        update={
                            "action_id": payload.chosen_action_id,
                            "recommended_action_id": recommended,
                            "propensity": 1.0,
                            "policy_name": "user-choice",
                            "policy_version": "1.0.0",
                            "selection_source": SelectionSource.USER_OVERRIDE,
                            "eligible_for_ope": False,
                            "explanation": (
                                "You chose a safe alternative. This decision is recorded as a user "
                                "override and excluded from off-policy evaluation."
                            ),
                        }
                    )

        retain_text = (
            payload.retain_text if payload.retain_text is not None else settings.raw_text_retention_default
        )
        record = ReflectionRecord(
            user_id=auth.user_id,
            text=payload.text if retain_text else None,
            text_retained=retain_text,
            context=payload.context,
            state=state,
            target=payload.target,
            reflection=reflection,
            safety=safety,
            decision=decision,
            model_run=model_run,
        )
        return repositories(auth).save_reflection(record)

    @app.post("/v1/reflections/analyze", response_model=PreparedAnalysis)
    def analyze_reflection(payload: AnalysisRequest) -> PreparedAnalysis:
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
        analysis = safe_analyze(
            settings,
            text=payload.text,
            context=payload.context,
            consent=payload.llm_consent,
            self_report=None,
            client=intelligence_client,
        )
        return PreparedAnalysis(
            state=analysis.state,
            reflection=analysis.reflection,
            safety=safety,
            model_run=analysis.model_run,
            resource_intent=analysis.resource_intent,
        )

    @app.post("/v1/actions/preview", response_model=ActionPreview)
    def preview_actions(payload: ActionPreviewRequest) -> ActionPreview:
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
        record = OutcomeRecord(user_id=auth.user_id, **payload.model_dump())
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

    return app


app = create_app()
