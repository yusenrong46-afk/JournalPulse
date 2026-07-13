import httpx
import pytest

from emotion_journal.api import create_app
from emotion_journal.llm import OpenAICompatibleCoachAdapter
from emotion_journal.schemas import CoachTurnResponse, ResourceCard


class DummyPredictor:
    def predict(self, text: str, *, location=None, activity=None):
        return type(
            "Prediction",
            (),
            {
                "emotion": "joy",
                "confidence": 0.88,
                "recommendation": "Turn the good energy into one deliberate next step.",
                "disclaimer": "Testing only",
                "is_crisis": False,
                "scores": {
                    "sadness": 0.02,
                    "joy": 0.88,
                    "love": 0.02,
                    "anger": 0.03,
                    "fear": 0.03,
                    "surprise": 0.02,
                },
                "support_message": None,
                "model_name": "distilroberta-base",
                "confidence_band": "high",
                "reflection_summary": "This reads like genuine lift rather than just brief relief.",
                "interpretation": "The model is reacting to language that sounds appreciative and energized.",
                "follow_up_prompts": [
                    "What created the lift most clearly?",
                    "How do you want to use it?",
                    "How could you recreate ten percent of it later?",
                ],
                "explanation_phrases": ["feel great", "good day"],
                "secondary_emotions": ["love"],
                "emotion_tags": ["gratitude"],
                "top_margin": 0.8,
                "is_mixed": False,
                "uncertainty_reason": None,
                "calibration_notes": ["joy_journal_cues"],
                "classifier_mode": "llm",
                "classifier_source": "llm",
                "classifier_fallback_reason": None,
            },
        )()


def test_extended_response_schemas_accept_older_shapes():
    resource = ResourceCard(
        id="old",
        title="Old card",
        url="https://example.com",
        resource_type="website",
        coping_style="read",
        provider="Example",
        embed_kind="link",
        summary="A previous client shape without new metadata.",
    )
    coach = CoachTurnResponse(
        assistant_message="Try one small next step.",
        coach_state={"step": "opening"},
    )

    assert resource.goal_tags == []
    assert resource.source_tier is None
    assert coach.coach_mode == "deterministic"
    assert coach.agent_mode == "deterministic"
    assert coach.practical_steps == []
    assert coach.resource_rationales == {}


@pytest.mark.anyio
async def test_api_happy_path(tmp_path):
    db_path = tmp_path / "api.db"
    app = create_app(predictor=DummyPredictor(), db_path=db_path)
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        predict_response = await client.post(
            "/predict",
            json={"text": "I feel great", "location": "Vancouver", "activity": "walking"},
        )
        health_response = await client.get("/health")
        ready_response = await client.get("/ready")
        create_response = await client.post(
            "/entries",
            json={
                "text": "I feel great",
                "location": "Vancouver",
                "activity": "walking",
                "feedback": "helpful",
                "coach_summary": {
                    "turn_count": 1,
                    "final_step": "opening",
                    "framing_emotion": "joy",
                    "selected_coping_style": None,
                    "resource_ids": ["game_autodraw"],
                    "used_llm": False,
                    "safety_mode": False,
                },
            },
        )
        patch_response = await client.patch("/entries/1/feedback", json={"feedback": "unsure"})
        entries_response = await client.get("/entries")
        analytics_response = await client.get("/analytics")
        resources_response = await client.get("/resources", params={"emotion": "joy"})
        resource_summary_response = await client.get("/resources/summary")
        resource_recommendations_response = await client.get(
            "/resources/recommendations",
            params={"emotion": "anger", "coping_style": "read", "goal": "plan"},
        )
        interaction_response = await client.post(
            "/resource-interactions",
            json={"resource_id": "game_autodraw", "action": "opened", "emotion": "joy"},
        )
        coach_response = await client.post(
            "/coach/respond",
            json={
                "text": "I feel great",
                "emotion": "joy",
                "confidence_band": "high",
                "user_message": "watch",
                "coach_state": {"step": "opening", "framing_emotion": "joy", "selected_coping_style": None},
                "is_crisis": False,
                "use_llm": False,
            },
        )

    assert predict_response.status_code == 200
    predict_payload = predict_response.json()
    assert predict_payload["emotion"] == "joy"
    assert predict_payload["confidence_band"] == "high"
    assert predict_payload["model_name"] == "distilroberta-base"
    assert predict_payload["explanation_phrases"] == ["feel great", "good day"]
    assert predict_payload["secondary_emotions"] == ["love"]
    assert predict_payload["emotion_tags"] == ["gratitude"]
    assert predict_payload["is_mixed"] is False
    assert predict_payload["top_margin"] == 0.8
    assert predict_payload["calibration_notes"] == ["joy_journal_cues"]
    assert predict_payload["classifier_mode"] == "llm"
    assert predict_payload["classifier_source"] == "llm"
    assert predict_payload["classifier_fallback_reason"] is None
    assert predict_payload["resources"]
    assert "goal_tags" in predict_payload["resources"][0]
    assert "source_tier" in predict_payload["resources"][0]
    assert "rationale" in predict_payload["resources"][0]
    assert predict_payload["coach_opening"]
    assert predict_payload["coach_available"] is True

    assert health_response.status_code == 200
    assert health_response.json()["status"] == "ok"

    assert ready_response.status_code == 200
    ready_payload = ready_response.json()
    assert ready_payload["status"] == "ready"
    assert ready_payload["model_ready"] is True
    assert ready_payload["database_ready"] is True
    assert ready_payload["resources_ready"] is True
    assert ready_payload["resource_count"] >= 1

    assert create_response.status_code == 200
    payload = create_response.json()
    assert payload["id"] == 1
    assert payload["feedback"] == "helpful"
    assert payload["reflection_summary"]
    assert payload["suggested_resource_ids"]
    assert payload["coach_summary"]["turn_count"] == 1
    assert payload["coach_summary"]["framing_emotion"] == "joy"

    assert patch_response.status_code == 200
    assert patch_response.json()["feedback"] == "unsure"

    assert entries_response.status_code == 200
    entry = entries_response.json()["entries"][0]
    assert entry["confidence_band"] == "high"
    assert entry["explanation_phrases"] == ["feel great", "good day"]
    assert entry["classifier_source"] == "llm"
    assert entry["resources"]
    assert entry["coach_summary"]["resource_ids"] == ["game_autodraw"]

    assert analytics_response.status_code == 200
    analytics = analytics_response.json()
    assert analytics["total_entries"] == 1
    assert analytics["confidence_band_counts"]["high"] == 1

    assert resources_response.status_code == 200
    assert resources_response.json()["resources"]

    assert resource_summary_response.status_code == 200
    assert resource_summary_response.json()["coverage_gaps"] == []

    assert resource_recommendations_response.status_code == 200
    assert resource_recommendations_response.json()["resources"][0]["id"] == "site_mind_manage_anger"

    assert interaction_response.status_code == 200
    assert interaction_response.json()["action"] == "opened"

    assert coach_response.status_code == 200
    coach_payload = coach_response.json()
    assert coach_payload["assistant_message"]
    assert coach_payload["suggested_replies"]
    assert "tips" in coach_payload
    assert coach_payload["coach_mode"] == "deterministic"
    assert coach_payload["resource_intent"] == "watch"
    assert coach_payload["resource_rationales"]
    assert "practical_steps" in coach_payload
    assert coach_payload["agent_mode"] == "deterministic"


@pytest.mark.anyio
async def test_api_structured_agent_fields(monkeypatch, tmp_path):
    class GoodAdapter:
        model = "openrouter/test-agent"

        def structured(self, *, deterministic_payload, allowed_resources, context):
            return {
                "assistant_message": "This sounds like a work moment where a clear next step could reduce the replay loop.",
                "tips": ["Keep the next step small."],
                "practical_steps": ["Write the exact concern.", "Choose one sentence to say tomorrow."],
                "suggested_replies": ["Help me plan", "Ground me"],
                "resource_intent": "plan",
                "resource_ids": [allowed_resources[0]["id"]],
                "reflection_question": "What outcome do you want from the conversation?",
                "communication_draft": "I wanted to revisit the meeting because I felt my idea was dismissed.",
                "confidence_note": "This is a planning suggestion based on the anger signal.",
            }

    monkeypatch.setenv("JOURNALPULSE_LLM_MODE", "structured")
    monkeypatch.setattr(OpenAICompatibleCoachAdapter, "from_env", classmethod(lambda cls: GoodAdapter()))

    db_path = tmp_path / "api-agent.db"
    app = create_app(predictor=DummyPredictor(), db_path=db_path)
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        predict_response = await client.post(
            "/predict",
            json={
                "text": "I was talked over in the meeting and I keep replaying it.",
                "emotion": "anger",
                "use_llm": True,
            },
        )
        response = await client.post(
            "/coach/respond",
            json={
                "text": "I was talked over in the meeting and I keep replaying it.",
                "emotion": "anger",
                "confidence_band": "high",
                "user_message": "help me plan",
                "coach_state": {"step": "opening", "framing_emotion": "anger"},
                "is_crisis": False,
                "use_llm": True,
            },
        )

    assert predict_response.status_code == 200
    predict_payload = predict_response.json()
    assert predict_payload["agent_mode"] == "structured"
    assert predict_payload["agent_model"] == "openrouter/test-agent"
    assert predict_payload["practical_steps"]

    assert response.status_code == 200
    payload = response.json()
    assert payload["coach_mode"] == "structured"
    assert payload["agent_mode"] == "structured"
    assert payload["agent_model"] == "openrouter/test-agent"
    assert payload["practical_steps"] == ["Write the exact concern.", "Choose one sentence to say tomorrow."]
    assert payload["reflection_question"].startswith("What outcome")
    assert payload["communication_draft"].startswith("I wanted")
    assert payload["confidence_note"].startswith("This is")


@pytest.mark.anyio
async def test_api_validation_failure(tmp_path):
    db_path = tmp_path / "api.db"
    app = create_app(predictor=DummyPredictor(), db_path=db_path)
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.post("/predict", json={"text": ""})
    assert response.status_code == 422
