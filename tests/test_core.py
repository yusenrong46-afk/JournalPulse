import sqlite3
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from emotion_journal.analytics import build_analytics
from emotion_journal.coach import build_initial_coach_turn, respond_with_coach
from emotion_journal.db import (
    get_analytics,
    initialize_database,
    insert_entry,
    list_entries,
    record_resource_interaction,
    update_feedback,
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

    entries = list_entries(db_path=db_path)
    assert len(entries) == 2
    legacy_entry = next(entry for entry in entries if entry["emotion"] == "anger")
    assert legacy_entry["coach_summary"]["final_step"] == "resource_follow_up"
    assert legacy_entry["coach_summary"]["selected_coping_style"] == "move"
    analytics = get_analytics(db_path=db_path)
    assert analytics["counts_by_emotion"]["anger"] == 1
    assert analytics["counts_by_emotion"]["joy"] == 1
    assert analytics["feedback_counts"]["not_helpful"] == 1
    assert analytics["confidence_band_counts"]["high"] == 1
    assert analytics["resource_action_counts"]["helpful"] == 1
    assert analytics["top_helpful_resources"][0]["resource_id"] == "site_nhs_breathing"


def test_resource_recommendations_and_coach_flow(tmp_path: Path):
    db_path = tmp_path / "journal.db"
    initialize_database(db_path)
    resources = recommend_resources("joy", db_path=db_path)
    assert resources
    assert any(resource["coping_style"] == "play" for resource in resources)

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


def test_resource_catalog_is_valid_and_covers_all_emotion_styles():
    errors = validate_resource_catalog()
    summary = resource_catalog_summary()

    assert errors == []
    assert summary["total_resources"] >= 10
    assert summary["crisis_safe_count"] >= 2
    assert summary["coverage_gaps"] == []
    assert summary["counts_by_coping_style"]["watch"] >= 1


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
        tone_tags=["grounding", "short"],
        duration_minutes=5,
    )

    assert draft["id"].startswith("website_example_grounding_walk")
    assert draft["emotion_tags"] == ["fear", "sadness"]
    assert "summary" in snapshot
    assert validate_resource_catalog(snapshot["resources"] + [draft]) == []


def test_build_analytics_handles_empty_sequence():
    analytics = build_analytics([])
    assert analytics["total_entries"] == 0
    assert analytics["counts_by_emotion"] == {}
