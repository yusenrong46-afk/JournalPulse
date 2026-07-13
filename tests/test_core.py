import sqlite3
from pathlib import Path

import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from emotion_journal.analytics import build_analytics
from emotion_journal.coach import build_initial_coach_turn, respond_with_coach
from emotion_journal.classification import EmotionAnalysis, maybe_generate_llm_emotion_analysis
from emotion_journal.db import (
    get_analytics,
    initialize_database,
    insert_entry,
    list_entries,
    record_resource_interaction,
    update_feedback,
)
from emotion_journal.llm import (
    OpenAICompatibleCoachAdapter,
    validate_structured_coach_payload,
    validate_structured_emotion_payload,
)
from emotion_journal.model import BaselineExplainer
from emotion_journal.preprocessing import contains_crisis_language, normalize_text
from emotion_journal.recommendations import build_support_response, confidence_band_for_score
from emotion_journal.resources import (
    build_resource_draft,
    filter_resources,
    recommend_resources,
    resource_admin_snapshot,
    resource_catalog_summary,
    validate_resource_catalog,
)


def test_normalize_text_removes_punctuation_and_urls():
    text = "I feel GREAT! Visit https://example.com right now."
    assert normalize_text(text) == "i feel great visit right now"


def test_contains_crisis_language_detects_high_risk_text():
    assert contains_crisis_language("I want to die and I do not feel safe")


def test_contains_crisis_language_handles_negation_and_broad_distress():
    assert not contains_crisis_language("I am not suicidal and do not want to hurt myself")
    assert not contains_crisis_language("I feel hopeless about the project, but I am safe")
    assert contains_crisis_language("I feel hopeless and cannot go on")
    assert contains_crisis_language("I am not safe tonight")
    assert contains_crisis_language("I do not feel safe tonight and I need help")
    assert contains_crisis_language("I feel unsafe and need someone nearby")


def test_confidence_band_thresholds():
    assert confidence_band_for_score(0.81) == "high"
    assert confidence_band_for_score(0.55) == "medium"
    assert confidence_band_for_score(0.32) == "low"


def test_support_response_switches_to_crisis_message():
    response = build_support_response("sadness", "I want to die", 0.9)
    assert response["is_crisis"] is True
    assert response["follow_up_prompts"] == []
    assert "urgent human support" in response["interpretation"].lower()


def test_baseline_explainer_returns_non_empty_phrases():
    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
            ("classifier", LogisticRegression(max_iter=500)),
        ]
    )
    texts = [
        "i feel amazing and grateful today",
        "i am furious about the meeting",
        "i am scared of what comes next",
        "i feel close to my family",
        "this caught me off guard completely",
        "i feel exhausted and low",
    ]
    labels = [1, 3, 4, 2, 5, 0]
    pipeline.fit(texts, labels)

    explainer = BaselineExplainer(pipeline)
    phrases = explainer.explain("i feel amazing and grateful today", "joy")
    assert phrases


def test_database_crud_migration_and_resource_analytics(tmp_path: Path):
    db_path = tmp_path / "journal.db"
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            CREATE TABLE journal_entries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                text TEXT NOT NULL,
                emotion TEXT NOT NULL,
                confidence REAL NOT NULL,
                recommendation TEXT NOT NULL,
                location TEXT,
                activity TEXT,
                feedback TEXT
            )
            """
        )
        connection.commit()

    initialize_database(db_path)

    first = insert_entry(
        text="I had a calm and joyful afternoon",
        emotion="joy",
        confidence=0.93,
        recommendation="Capture the highlight",
        reflection_summary="The entry feels light and open.",
        interpretation="The language sounds appreciative and energized.",
        confidence_band="high",
        model_name="distilroberta-base",
        classifier_mode="llm",
        classifier_source="llm",
        classifier_fallback_reason=None,
        support_message=None,
        follow_up_prompts=["a", "b", "c"],
        explanation_phrases=["joyful afternoon", "calm"],
        suggested_resource_ids=["video_meditation_start_day", "game_autodraw"],
        coach_state_summary="step=opening|emotion=joy|style=none",
        coach_summary={
            "turn_count": 2,
            "final_step": "opening",
            "framing_emotion": "joy",
            "selected_coping_style": None,
            "resource_ids": ["video_meditation_start_day", "game_autodraw"],
            "used_llm": False,
            "safety_mode": False,
        },
        feedback="helpful",
        db_path=db_path,
    )
    insert_entry(
        text="I am angry about the meeting",
        emotion="anger",
        confidence=0.61,
        recommendation="Separate the trigger from the next move.",
        reflection_summary="There is direct friction in the entry.",
        interpretation="The model sees blocked, provoked language.",
        confidence_band="medium",
        model_name="distilroberta-base",
        support_message=None,
        follow_up_prompts=["x", "y", "z"],
        explanation_phrases=["angry", "meeting"],
        suggested_resource_ids=["site_nhs_breathing"],
        coach_state_summary="step=resource_follow_up|emotion=anger|style=move",
        db_path=db_path,
    )

    record_resource_interaction(
        resource_id="site_nhs_breathing",
        action="helpful",
        emotion="anger",
        entry_id=first["id"],
        db_path=db_path,
    )
    record_resource_interaction(
        resource_id="game_autodraw",
        action="opened",
        emotion="joy",
        db_path=db_path,
    )

    updated = update_feedback(first["id"], "not_helpful", db_path=db_path)
    assert updated["feedback"] == "not_helpful"
    assert updated["follow_up_prompts"] == ["a", "b", "c"]
    assert updated["suggested_resource_ids"] == ["video_meditation_start_day", "game_autodraw"]
    assert updated["coach_summary"]["turn_count"] == 2
    assert "coach_transcript" not in updated["coach_summary"]
    assert "user_message" not in updated["coach_summary"]
    assert updated["classifier_mode"] == "llm"
    assert updated["classifier_source"] == "llm"

    entries = list_entries(db_path=db_path)
    assert len(entries) == 2
    legacy_entry = next(entry for entry in entries if entry["emotion"] == "anger")
    assert legacy_entry["coach_summary"]["final_step"] == "resource_follow_up"
    assert legacy_entry["coach_summary"]["selected_coping_style"] == "move"
    assert legacy_entry["classifier_mode"] == "calibrated"
    assert legacy_entry["classifier_source"] == "artifact"
    analytics = get_analytics(db_path=db_path)
    assert analytics["counts_by_emotion"]["anger"] == 1
    assert analytics["counts_by_emotion"]["joy"] == 1
    assert analytics["feedback_counts"]["not_helpful"] == 1
    assert analytics["confidence_band_counts"]["high"] == 1
    assert analytics["resource_action_counts"]["helpful"] == 1
    assert analytics["top_helpful_resources"][0]["resource_id"] == "site_nhs_breathing"


def test_validate_structured_emotion_payload_normalizes_llm_output():
    payload = validate_structured_emotion_payload(
        {
            "primary_emotion": "fear",
            "secondary_emotions": ["fear", "sadness", "anger"],
            "emotion_tags": ["anxiety", "work_stress", "anxiety"],
            "intensity": 0.72,
            "confidence": "high",
            "is_mixed": False,
            "themes": ["uncertainty", "work", "uncertainty"],
            "rationale": "The entry focuses on uncertainty and future-oriented worry.",
        }
    )

    assert payload["primary_emotion"] == "fear"
    assert payload["secondary_emotions"] == ["sadness", "anger"]
    assert payload["emotion_tags"] == ["anxiety", "work_stress"]
    assert payload["is_mixed"] is True


def test_llm_classifier_can_replace_artifact_prediction(monkeypatch):
    fallback = EmotionAnalysis(
        emotion="joy",
        confidence=0.52,
        scores={
            "sadness": 0.08,
            "joy": 0.52,
            "love": 0.04,
            "anger": 0.24,
            "fear": 0.08,
            "surprise": 0.04,
        },
    )

    class FakeAdapter:
        model = "gemma-test"

        def classify_emotion(self, *, text, calibrated_context):
            assert "dismissed" in text
            assert calibrated_context["primary_emotion"] == "joy"
            return {
                "primary_emotion": "anger",
                "secondary_emotions": ["sadness"],
                "emotion_tags": ["frustration", "work_stress"],
                "intensity": 0.78,
                "confidence": "high",
                "is_mixed": True,
                "themes": ["work meeting"],
                "rationale": "The entry centers on being dismissed and frustrated.",
            }

    monkeypatch.setenv("JOURNALPULSE_CLASSIFIER_MODE", "llm")
    monkeypatch.setattr(
        "emotion_journal.classification.OpenAICompatibleCoachAdapter.from_env",
        lambda: FakeAdapter(),
    )

    analysis = maybe_generate_llm_emotion_analysis(
        "My manager dismissed my idea and I am still frustrated.",
        fallback,
    )

    assert analysis.emotion == "anger"
    assert analysis.classifier_mode == "llm"
    assert analysis.classifier_source == "llm"
    assert "structured_llm:gemma-test" in analysis.calibration_notes


def test_hybrid_mode_blends_model_and_llm(monkeypatch):
    # Model leans joy; LLM is confident it is anger. Hybrid should agree with the LLM
    # on the label but keep meaningful weight on the model's joy signal (a true blend,
    # not a full replacement).
    fallback = EmotionAnalysis(
        emotion="joy",
        confidence=0.52,
        scores={
            "sadness": 0.08,
            "joy": 0.52,
            "love": 0.04,
            "anger": 0.24,
            "fear": 0.08,
            "surprise": 0.04,
        },
        emotion_tags=["work_stress"],
    )

    class FakeAdapter:
        model = "gemma-test"

        def classify_emotion(self, *, text, calibrated_context):
            return {
                "primary_emotion": "anger",
                "secondary_emotions": ["sadness"],
                "emotion_tags": ["frustration"],
                "intensity": 0.78,
                "confidence": "high",
                "is_mixed": True,
                "themes": ["work meeting"],
                "rationale": "The entry centers on being dismissed and frustrated.",
            }

    monkeypatch.setenv("JOURNALPULSE_CLASSIFIER_MODE", "hybrid")
    monkeypatch.setattr(
        "emotion_journal.classification.OpenAICompatibleCoachAdapter.from_env",
        lambda: FakeAdapter(),
    )

    analysis = maybe_generate_llm_emotion_analysis(
        "My manager dismissed my idea and I am still frustrated.",
        fallback,
    )

    assert analysis.emotion == "anger"
    assert analysis.classifier_mode == "hybrid"
    assert analysis.classifier_source == "hybrid"
    # Blend keeps the model's joy weight alive (pure LLM mode would zero it out).
    assert analysis.scores["joy"] > 0.2
    # Tags from both sources are merged.
    assert "frustration" in analysis.emotion_tags
    assert "work_stress" in analysis.emotion_tags
    assert any(note.startswith("hybrid_blend:") for note in analysis.calibration_notes)


def test_llm_classifier_bypasses_crisis_text(monkeypatch):
    fallback = EmotionAnalysis(
        emotion="joy",
        confidence=0.92,
        scores={
            "sadness": 0.01,
            "joy": 0.92,
            "love": 0.01,
            "anger": 0.02,
            "fear": 0.02,
            "surprise": 0.02,
        },
    )

    def fail_if_called():
        raise AssertionError("LLM classifier should not be created in crisis mode")

    monkeypatch.setenv("JOURNALPULSE_CLASSIFIER_MODE", "llm")
    monkeypatch.setattr(
        "emotion_journal.classification.OpenAICompatibleCoachAdapter.from_env",
        fail_if_called,
    )

    analysis = maybe_generate_llm_emotion_analysis(
        "I want to die and I do not feel safe.",
        fallback,
    )

    assert analysis.classifier_source == "artifact"
    assert analysis.classifier_fallback_reason == "crisis_mode_llm_bypassed"


def test_resource_recommendations_and_coach_flow(tmp_path: Path):
    db_path = tmp_path / "journal.db"
    initialize_database(db_path)
    resources = recommend_resources("joy", db_path=db_path)
    assert resources
    assert any(resource["coping_style"] == "play" for resource in resources)
    assert all(resource["resource_type"] != "support" for resource in resources)

    crisis_resources = recommend_resources("sadness", is_crisis=True, db_path=db_path)
    assert crisis_resources
    assert any(resource["resource_type"] == "support" for resource in crisis_resources)

    read_resources = filter_resources(emotion="fear", resource_type="website", coping_style="move")
    assert read_resources

    opening = build_initial_coach_turn(
        entry_text="I feel mixed and uncertain after the call.",
        emotion="fear",
        confidence_band="low",
        is_crisis=False,
        use_llm=False,
    )
    assert opening["coach_state"]["step"] == "clarify"
    assert opening["suggested_replies"]

    response = respond_with_coach(
        entry_text="I feel mixed and uncertain after the call.",
        emotion="fear",
        confidence_band="low",
        coach_state=opening["coach_state"],
        user_message="watch",
        is_crisis=False,
        use_llm=False,
        db_path=db_path,
    )
    assert response["coach_state"]["selected_coping_style"] == "watch"
    assert response["resource_ids"]

    tip_response = respond_with_coach(
        entry_text="I feel mixed and uncertain after the call.",
        emotion="fear",
        confidence_band="low",
        coach_state=response["coach_state"],
        user_message="give me tips",
        is_crisis=False,
        use_llm=False,
        db_path=db_path,
    )
    assert tip_response["tips"]
    assert "practical" in tip_response["assistant_message"].lower()

    plan_response = respond_with_coach(
        entry_text="I feel mixed and uncertain after the call.",
        emotion="fear",
        confidence_band="low",
        coach_state=tip_response["coach_state"],
        user_message="help me make a plan",
        is_crisis=False,
        use_llm=False,
        db_path=db_path,
    )
    assert plan_response["tips"]
    assert plan_response["resource_ids"]
    assert plan_response["resource_rationales"]
    assert plan_response["resource_intent"] == "plan"


def test_resource_ranking_uses_goal_metadata(tmp_path: Path):
    db_path = tmp_path / "journal.db"
    initialize_database(db_path)

    grounding = recommend_resources("fear", goal="ground", db_path=db_path)
    assert grounding
    assert "ground" in grounding[0]["goal_tags"]
    assert grounding[0]["rationale"].startswith("Chosen because")

    planning_reads = recommend_resources("fear", coping_style="read", goal="plan", db_path=db_path)
    assert planning_reads
    assert all(resource["coping_style"] == "read" for resource in planning_reads)
    assert any("planning" in resource["goal_tags"] for resource in planning_reads)

    anger_plan = recommend_resources("anger", coping_style="read", goal="plan", db_path=db_path)
    assert anger_plan[0]["id"] == "site_mind_manage_anger"


def test_structured_coach_payload_validation():
    payload = validate_structured_coach_payload(
        {
            "assistant_message": "Try one small reset before deciding what comes next.",
            "tips": ["Breathe out slowly", "Name one next step"],
            "practical_steps": ["Put the next action in one sentence."],
            "suggested_replies": ["Show resources", "Done"],
            "resource_intent": "ground",
            "resource_ids": ["site_nhs_breathing"],
            "reflection_question": "What is one fact you know right now?",
            "communication_draft": "I need a little time before I respond clearly.",
            "confidence_note": "This is a practical suggestion, not a certainty.",
        },
        allowed_resource_ids=["site_nhs_breathing"],
        fallback_replies=["Done"],
    )

    assert payload["resource_ids"] == ["site_nhs_breathing"]
    assert payload["resource_intent"] == "ground"
    assert payload["practical_steps"] == ["Put the next action in one sentence."]
    assert payload["reflection_question"] == "What is one fact you know right now?"
    assert payload["communication_draft"].startswith("I need")

    with pytest.raises(ValueError):
        validate_structured_coach_payload(
            {
                "assistant_message": "Try one small reset.",
                "suggested_replies": ["Done"],
                "resource_intent": "ground",
                "resource_ids": ["invented_resource"],
            },
            allowed_resource_ids=["site_nhs_breathing"],
            fallback_replies=["Done"],
        )


def test_structured_llm_mode_can_shape_valid_coach_turn(monkeypatch, tmp_path: Path):
    class GoodAdapter:
        model = "openrouter/test-model"

        def structured(self, *, deterministic_payload, allowed_resources, context):
            return {
                "assistant_message": "A short video reset fits this moment; keep it simple and choose one thing to notice.",
                "tips": ["Let the video run for one minute before analyzing anything."],
                "practical_steps": ["Start the video.", "Name one physical sensation.", "Choose one next action."],
                "suggested_replies": ["Read", "Move", "Done"],
                "resource_intent": "watch",
                "resource_ids": [allowed_resources[0]["id"]],
                "reflection_question": "What would feel one notch easier after this?",
                "communication_draft": "I need a few minutes, then I can come back with a clearer answer.",
                "confidence_note": "The suggestion is based on the fear signal and the user's request to watch.",
            }

    monkeypatch.setenv("JOURNALPULSE_LLM_MODE", "structured")
    monkeypatch.setattr(OpenAICompatibleCoachAdapter, "from_env", classmethod(lambda cls: GoodAdapter()))

    db_path = tmp_path / "journal.db"
    initialize_database(db_path)
    response = respond_with_coach(
        entry_text="I feel nervous and need to settle down.",
        emotion="fear",
        confidence_band="high",
        coach_state={"step": "opening", "framing_emotion": "fear"},
        user_message="watch",
        is_crisis=False,
        use_llm=True,
        db_path=db_path,
    )

    assert response["coach_mode"] == "structured"
    assert response["agent_mode"] == "structured"
    assert response["agent_model"] == "openrouter/test-model"
    assert response["used_llm"] is True
    assert response["tips"] == ["Let the video run for one minute before analyzing anything."]
    assert response["practical_steps"][0] == "Start the video."
    assert response["reflection_question"].startswith("What would feel")
    assert response["communication_draft"].startswith("I need")
    assert response["resource_ids"]


def test_invalid_structured_llm_falls_back(monkeypatch, tmp_path: Path):
    class BadAdapter:
        def structured(self, *, deterministic_payload, allowed_resources, context):
            return {
                "assistant_message": "I invented something.",
                "suggested_replies": ["Done"],
                "resource_intent": "watch",
                "resource_ids": ["not_in_catalog"],
            }

    monkeypatch.setenv("JOURNALPULSE_LLM_MODE", "structured")
    monkeypatch.setattr(OpenAICompatibleCoachAdapter, "from_env", classmethod(lambda cls: BadAdapter()))

    db_path = tmp_path / "journal.db"
    initialize_database(db_path)
    response = respond_with_coach(
        entry_text="I feel nervous and need to settle down.",
        emotion="fear",
        confidence_band="high",
        coach_state={"step": "opening", "framing_emotion": "fear"},
        user_message="watch",
        is_crisis=False,
        use_llm=True,
        db_path=db_path,
    )

    assert response["coach_mode"] == "deterministic"
    assert response["agent_mode"] == "fallback"
    assert response["used_llm"] is False
    assert response["fallback_reason"].startswith("structured_llm_invalid")
    assert response["agent_fallback_reason"].startswith("structured_llm_invalid")


def test_crisis_mode_bypasses_llm(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("JOURNALPULSE_LLM_MODE", "structured")

    db_path = tmp_path / "journal.db"
    initialize_database(db_path)
    response = respond_with_coach(
        entry_text="I want to die and I am not safe.",
        emotion="sadness",
        confidence_band="high",
        coach_state={"step": "opening", "framing_emotion": "sadness"},
        user_message="show resources",
        is_crisis=True,
        use_llm=True,
        db_path=db_path,
    )

    assert response["coach_mode"] == "deterministic"
    assert response["used_llm"] is False
    assert response["fallback_reason"] == "crisis_mode_llm_bypassed"
    assert response["agent_fallback_reason"] == "crisis_mode_llm_bypassed"
    assert response["resource_ids"]


def test_openrouter_adapter_env_and_headers(monkeypatch):
    monkeypatch.setenv("JOURNALPULSE_LLM_API_KEY", "test-key")
    monkeypatch.setenv("JOURNALPULSE_LLM_MODEL", "google/gemma-3-27b-it")
    monkeypatch.delenv("JOURNALPULSE_LLM_BASE_URL", raising=False)
    monkeypatch.setenv("JOURNALPULSE_LLM_APP_URL", "https://example.com/journalpulse")
    monkeypatch.setenv("JOURNALPULSE_LLM_APP_TITLE", "JournalPulse")

    adapter = OpenAICompatibleCoachAdapter.from_env()

    assert adapter.base_url == "https://openrouter.ai/api/v1"
    assert adapter.model == "google/gemma-3-27b-it"
    assert adapter.headers()["HTTP-Referer"] == "https://example.com/journalpulse"
    assert adapter.headers()["X-Title"] == "JournalPulse"
    assert adapter.headers()["X-OpenRouter-Title"] == "JournalPulse"


def test_resource_catalog_is_valid_and_covers_all_emotion_styles():
    errors = validate_resource_catalog()
    summary = resource_catalog_summary()

    assert errors == []
    assert summary["total_resources"] >= 10
    assert summary["crisis_safe_count"] >= 2
    assert summary["coverage_gaps"] == []
    assert summary["counts_by_coping_style"]["watch"] >= 1
    assert summary["counts_by_source_tier"]["official"] >= 1


def test_resource_admin_helpers_build_valid_preview():
    snapshot = resource_admin_snapshot()
    draft = build_resource_draft(
        title="Grounding Walk",
        url="https://example.com/grounding-walk",
        resource_type="website",
        coping_style="move",
        provider="Example",
        embed_kind="link",
        summary="A short grounding exercise for getting oriented.",
        emotion_tags=["fear", "sadness"],
        goal_tags=["ground", "movement"],
        source_tier="educational",
        reviewed_at="2026-05-09",
        tone_tags=["grounding", "short"],
        duration_minutes=5,
    )

    assert draft["id"].startswith("website_example_grounding_walk")
    assert draft["emotion_tags"] == ["fear", "sadness"]
    assert draft["goal_tags"] == ["ground", "movement"]
    assert "summary" in snapshot
    assert validate_resource_catalog(snapshot["resources"] + [draft]) == []


def test_build_analytics_handles_empty_sequence():
    analytics = build_analytics([])
    assert analytics["total_entries"] == 0
    assert analytics["counts_by_emotion"] == {}
