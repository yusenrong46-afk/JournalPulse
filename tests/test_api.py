import httpx
import pytest

from emotion_journal.api import create_app


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
            },
        )()


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
    assert predict_payload["resources"]
    assert predict_payload["coach_opening"]
    assert predict_payload["coach_available"] is True

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

    assert interaction_response.status_code == 200
    assert interaction_response.json()["action"] == "opened"

    assert coach_response.status_code == 200
    assert coach_response.json()["assistant_message"]
    assert coach_response.json()["suggested_replies"]


@pytest.mark.anyio
async def test_api_validation_failure(tmp_path):
    db_path = tmp_path / "api.db"
    app = create_app(predictor=DummyPredictor(), db_path=db_path)
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.post("/predict", json={"text": ""})
    assert response.status_code == 422
