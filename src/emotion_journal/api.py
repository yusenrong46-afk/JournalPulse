from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query

from .coach import coach_available, llm_mode_available, respond_with_coach
from .config import (
    CRISIS_DISCLAIMER,
    DEFAULT_DISCLAIMER,
    app_environment,
    database_path,
    deployment_mode,
)
from .db import (
    get_analytics,
    initialize_database,
    insert_entry,
    list_entries,
    record_resource_interaction,
    update_feedback,
)
from .experience import build_prediction_experience
from .model import get_default_predictor
from .resources import filter_resources, get_resource_lookup, load_resource_catalog, recommend_resources, resource_catalog_summary
from .schemas import (
    AnalyticsResponse,
    CoachTurnRequest,
    CoachTurnResponse,
    EntriesResponse,
    FeedbackUpdate,
    HealthResponse,
    JournalEntryCreate,
    JournalEntryResponse,
    JournalInput,
    PredictionResponse,
    ReadinessResponse,
    ResourceInteractionCreate,
    ResourceInteractionResponse,
    ResourceSummaryResponse,
    ResourcesResponse,
)


def create_app(*, predictor=None, db_path: Path = None) -> FastAPI:
    resolved_db_path = db_path or database_path()

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        initialize_database(resolved_db_path)
        yield

    app = FastAPI(title="JournalPulse API", version="0.4.0", lifespan=lifespan)

    def resolve_predictor():
        return predictor or get_default_predictor()

    def summarize_coach_state(coach_state: dict) -> str:
        parts = [f"step={coach_state.get('step', 'opening')}"]
        if coach_state.get("framing_emotion"):
            parts.append(f"emotion={coach_state['framing_emotion']}")
        if coach_state.get("selected_coping_style"):
            parts.append(f"style={coach_state['selected_coping_style']}")
        return "|".join(parts)

    def safe_coach_summary(coach_state: dict, resources: list, *, used_llm: bool = False) -> dict:
        return {
            "turn_count": int(coach_state.get("turns", 0) or 0),
            "final_step": coach_state.get("step", "opening"),
            "framing_emotion": coach_state.get("framing_emotion"),
            "selected_coping_style": coach_state.get("selected_coping_style"),
            "resource_ids": [resource["id"] for resource in resources],
            "used_llm": bool(used_llm),
            "safety_mode": bool(coach_state.get("is_crisis")),
        }

    def experience_payload(experience: dict) -> dict:
        result = experience["prediction"]
        resources = experience["resources"]
        coach = experience["coach"]
        recommendation_meta = experience.get("recommendation_meta", {})
        fields = (
            "emotion",
            "confidence",
            "recommendation",
            "disclaimer",
            "is_crisis",
            "scores",
            "support_message",
            "model_name",
            "confidence_band",
            "reflection_summary",
            "interpretation",
            "follow_up_prompts",
            "explanation_phrases",
            "secondary_emotions",
            "emotion_tags",
            "top_margin",
            "is_mixed",
            "uncertainty_reason",
            "calibration_notes",
            "classifier_mode",
            "classifier_source",
            "classifier_fallback_reason",
        )
        payload = {field: getattr(result, field) for field in fields}
        payload.update(
            {
                "resources": resources,
                "suggested_resource_ids": [resource["id"] for resource in resources],
                "used_llm_recommender": recommendation_meta.get("used_llm_recommender", False),
                "generated_resource_count": recommendation_meta.get("generated_count", 0),
                "recommender_fallback_reason": recommendation_meta.get("recommender_fallback_reason"),
                "recommender_model": recommendation_meta.get("recommender_model"),
                "coach_opening": coach["assistant_message"],
                "coach_state": coach["coach_state"],
                "suggested_replies": coach["suggested_replies"],
                "tips": coach.get("tips", []),
                "practical_steps": coach.get("practical_steps", []),
                "reflection_question": coach.get("reflection_question"),
                "communication_draft": coach.get("communication_draft"),
                "confidence_note": coach.get("confidence_note"),
                "used_llm": coach.get("used_llm", False),
                "coach_mode": coach.get("coach_mode", "deterministic"),
                "agent_mode": coach.get("agent_mode", "deterministic"),
                "agent_model": coach.get("agent_model"),
                "agent_fallback_reason": coach.get("agent_fallback_reason"),
                "fallback_reason": coach.get("fallback_reason"),
                "resource_intent": coach.get("resource_intent"),
                "resource_rationales": coach.get("resource_rationales", {}),
                "coach_available": coach_available(),
            }
        )
        return payload

    def entry_response_payload(entry: dict) -> dict:
        resource_lookup = get_resource_lookup()
        resource_ids = entry.get("suggested_resource_ids", [])
        resources = [
            resource_lookup[resource_id]
            for resource_id in resource_ids
            if resource_id in resource_lookup
        ]
        is_crisis = bool(entry.get("support_message"))
        response = dict(entry)
        response.update(
            {
                "confidence": round(float(entry["confidence"]), 4),
                "disclaimer": CRISIS_DISCLAIMER if is_crisis else DEFAULT_DISCLAIMER,
                "is_crisis": is_crisis,
                "scores": {},
                "resources": resources,
                "coach_opening": None,
                "coach_state": {},
                "suggested_replies": [],
                "coach_available": coach_available(),
            }
        )
        return response

    @app.get("/health", response_model=HealthResponse)
    def healthcheck() -> HealthResponse:
        try:
            resolve_predictor()
            model_ready = True
        except Exception:
            model_ready = False
        initialize_database(resolved_db_path)
        return HealthResponse(
            status="ok",
            model_ready=model_ready,
            llm_mode_available=llm_mode_available(),
            db_path=str(resolved_db_path),
        )

    @app.get("/ready", response_model=ReadinessResponse)
    def readiness() -> ReadinessResponse:
        checks = {}
        model_ready = False
        database_ready = False
        resources_ready = False
        model_name = None
        model_artifact_source = "local"
        resource_count = 0

        try:
            predictor_instance = resolve_predictor()
            model_ready = True
            model_name = predictor_instance.metadata.get("model_name")
            model_artifact_source = predictor_instance.metadata.get("artifact_source", "local")
            checks["model"] = "ready"
        except Exception as exc:
            checks["model"] = f"not_ready:{exc.__class__.__name__}"

        try:
            initialize_database(resolved_db_path)
            database_ready = True
            checks["database"] = "ready"
        except Exception as exc:
            checks["database"] = f"not_ready:{exc.__class__.__name__}"

        try:
            resource_count = len(load_resource_catalog())
            resources_ready = resource_count > 0
            checks["resources"] = "ready" if resources_ready else "not_ready:empty_catalog"
        except Exception as exc:
            checks["resources"] = f"not_ready:{exc.__class__.__name__}"

        ready = model_ready and database_ready and resources_ready
        return ReadinessResponse(
            status="ready" if ready else "not_ready",
            app_environment=app_environment(),
            deployment_mode=deployment_mode(),
            model_ready=model_ready,
            database_ready=database_ready,
            resources_ready=resources_ready,
            model_name=model_name,
            model_artifact_source=model_artifact_source,
            resource_count=resource_count,
            db_path=str(resolved_db_path),
            checks=checks,
        )

    @app.post("/predict", response_model=PredictionResponse)
    def predict(payload: JournalInput) -> PredictionResponse:
        try:
            experience = build_prediction_experience(
                resolve_predictor(),
                payload.text,
                location=payload.location,
                activity=payload.activity,
                db_path=resolved_db_path,
                use_llm=payload.use_llm,
            )
        except Exception as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return PredictionResponse(**experience_payload(experience))

    @app.post("/entries", response_model=JournalEntryResponse)
    def create_entry(payload: JournalEntryCreate) -> JournalEntryResponse:
        try:
            experience = build_prediction_experience(
                resolve_predictor(),
                payload.text,
                location=payload.location,
                activity=payload.activity,
                db_path=resolved_db_path,
                use_llm=payload.use_llm,
            )
        except Exception as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

        result = experience["prediction"]
        coach = experience["coach"]
        resources = experience["resources"]
        entry = insert_entry(
            text=payload.text,
            emotion=result.emotion,
            confidence=result.confidence,
            recommendation=result.recommendation,
            location=payload.location,
            activity=payload.activity,
            feedback=payload.feedback,
            reflection_summary=result.reflection_summary,
            interpretation=result.interpretation,
            confidence_band=result.confidence_band,
            model_name=result.model_name,
            classifier_mode=result.classifier_mode,
            classifier_source=result.classifier_source,
            classifier_fallback_reason=result.classifier_fallback_reason,
            support_message=result.support_message,
            follow_up_prompts=result.follow_up_prompts,
            explanation_phrases=result.explanation_phrases,
            coach_state_summary=summarize_coach_state(coach["coach_state"]),
            coach_summary=(
                payload.coach_summary.model_dump()
                if payload.coach_summary is not None
                else safe_coach_summary(
                    coach["coach_state"],
                    resources,
                    used_llm=coach.get("used_llm", False),
                )
            ),
            suggested_resource_ids=[
                resource["id"] for resource in resources if resource.get("source") != "ai_suggested"
            ],
            db_path=resolved_db_path,
        )
        combined = dict(entry)
        combined.update(experience_payload(experience))
        combined["coach_state_summary"] = summarize_coach_state(coach["coach_state"])
        combined["coach_summary"] = entry.get("coach_summary")
        return JournalEntryResponse(**combined)

    @app.patch("/entries/{entry_id}/feedback", response_model=JournalEntryResponse)
    def patch_feedback(entry_id: int, payload: FeedbackUpdate) -> JournalEntryResponse:
        try:
            entry = update_feedback(entry_id, payload.feedback, db_path=resolved_db_path)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return JournalEntryResponse(**entry_response_payload(entry))

    @app.get("/entries", response_model=EntriesResponse)
    def read_entries(
        emotion: Optional[str] = Query(default=None),
        start_date: Optional[str] = Query(default=None),
        end_date: Optional[str] = Query(default=None),
    ) -> EntriesResponse:
        entries = list_entries(
            emotion=emotion,
            start_date=start_date,
            end_date=end_date,
            db_path=resolved_db_path,
        )
        return EntriesResponse(
            entries=[JournalEntryResponse(**entry_response_payload(entry)) for entry in entries]
        )

    @app.get("/analytics", response_model=AnalyticsResponse)
    def analytics(
        start_date: Optional[str] = Query(default=None),
        end_date: Optional[str] = Query(default=None),
    ) -> AnalyticsResponse:
        data = get_analytics(start_date=start_date, end_date=end_date, db_path=resolved_db_path)
        return AnalyticsResponse(**data)

    @app.get("/resources", response_model=ResourcesResponse)
    def read_resources(
        emotion: Optional[str] = Query(default=None),
        resource_type: Optional[str] = Query(default=None),
        coping_style: Optional[str] = Query(default=None),
        limit: Optional[int] = Query(default=None, ge=1, le=20),
    ) -> ResourcesResponse:
        resources = filter_resources(
            emotion=emotion,
            resource_type=resource_type,
            coping_style=coping_style,
            limit=limit,
        )
        return ResourcesResponse(resources=resources)

    @app.get("/resources/summary", response_model=ResourceSummaryResponse)
    def read_resource_summary() -> ResourceSummaryResponse:
        return ResourceSummaryResponse(**resource_catalog_summary())

    @app.get("/resources/recommendations", response_model=ResourcesResponse)
    def read_resource_recommendations(
        emotion: Optional[str] = Query(default=None),
        coping_style: Optional[str] = Query(default=None),
        goal: Optional[str] = Query(default=None),
        is_crisis: bool = Query(default=False),
    ) -> ResourcesResponse:
        resources = recommend_resources(
            emotion,
            coping_style=coping_style,
            goal=goal,
            is_crisis=is_crisis,
            db_path=resolved_db_path,
        )
        return ResourcesResponse(resources=resources)

    @app.post("/resource-interactions", response_model=ResourceInteractionResponse)
    def create_resource_interaction(payload: ResourceInteractionCreate) -> ResourceInteractionResponse:
        interaction = record_resource_interaction(
            resource_id=payload.resource_id,
            action=payload.action,
            emotion=payload.emotion,
            entry_id=payload.entry_id,
            db_path=resolved_db_path,
        )
        return ResourceInteractionResponse(**interaction)

    @app.post("/coach/respond", response_model=CoachTurnResponse)
    def coach_respond(payload: CoachTurnRequest) -> CoachTurnResponse:
        response = respond_with_coach(
            entry_text=payload.text,
            emotion=payload.emotion,
            confidence_band=payload.confidence_band,
            coach_state=payload.coach_state,
            user_message=payload.user_message,
            is_crisis=payload.is_crisis,
            use_llm=payload.use_llm,
            db_path=resolved_db_path,
        )
        return CoachTurnResponse(**response)

    return app


app = create_app()
