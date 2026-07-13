from pathlib import Path

from fastapi.testclient import TestClient

from journalpulse.api import create_app
from journalpulse.config import Settings

USER_A = "00000000-0000-4000-8000-000000000001"
USER_B = "00000000-0000-4000-8000-000000000002"


def settings(tmp_path: Path) -> Settings:
    root = Path(__file__).resolve().parents[1]
    return Settings(
        environment="test",
        database_path=tmp_path / "beta.db",
        resource_catalog_path=root / "assets" / "resources" / "catalog.json",
        openrouter_api_key=None,
        openrouter_model="openai/gpt-5.4-mini",
        openrouter_base_url="https://openrouter.ai/api/v1",
        openrouter_zdr=True,
        openrouter_timeout_seconds=2,
        supabase_url=None,
        supabase_anon_key=None,
        raw_text_retention_default=False,
    )


def reflection_payload(**overrides) -> dict:
    payload = {
        "text": "The meeting is still replaying in my head and I want to feel more settled.",
        "context": {"activity": "after work"},
        "self_report": {
            "valence": -0.45,
            "arousal": 0.72,
            "agency": 0.38,
            "emotion_tags": ["frustration"],
            "confidence": 1.0,
        },
        "target": {"valence": 0.0, "arousal": 0.35, "agency": 0.65, "goal": "settle"},
        "llm_consent": False,
        "retain_text": False,
        "locale": "CA",
    }
    payload.update(overrides)
    return payload


def test_guided_reflection_outcome_insights_and_delete(tmp_path: Path):
    app = create_app(settings=settings(tmp_path))
    with TestClient(app) as client:
        response = client.post(
            "/v1/reflections", json=reflection_payload(), headers={"X-JournalPulse-User": USER_A}
        )
        assert response.status_code == 201
        record = response.json()
        assert record["text"] is None
        assert record["text_retained"] is False
        assert record["state"]["agency"] == 0.38
        assert record["decision"]["policy_name"] == "fixed-baseline"
        assert record["decision"]["propensity"] == 1.0
        assert record["model_run"]["used_fallback"] is True

        outcome = client.post(
            "/v1/outcomes",
            headers={"X-JournalPulse-User": USER_A},
            json={
                "decision_id": record["decision"]["decision_id"],
                "completed": True,
                "post_state": {
                    "valence": -0.1,
                    "arousal": 0.4,
                    "agency": 0.62,
                    "emotion_tags": ["settled"],
                    "confidence": 1.0,
                },
                "helpfulness": 4,
                "effort": 2,
                "elapsed_minutes": 15,
            },
        )
        assert outcome.status_code == 201

        insights = client.get("/v1/insights", headers={"X-JournalPulse-User": USER_A}).json()
        assert insights["reflection_count"] == 1
        assert insights["completed_outcomes"] == 1
        assert insights["average_state_change"]["agency"] == 0.24
        assert insights["completion_rate"] == 1.0
        assert insights["pending_decision_ids"] == []
        assert insights["state_trajectory"][0]["reflection_id"] == record["id"]
        assert len(
            client.get("/v1/outcomes", headers={"X-JournalPulse-User": USER_A}).json()["items"]
        ) == 1

        assert client.get("/v1/reflections", headers={"X-JournalPulse-User": USER_B}).json()["items"] == []
        deleted = client.delete(f"/v1/reflections/{record['id']}", headers={"X-JournalPulse-User": USER_A})
        assert deleted.status_code == 204
        assert client.get("/v1/reflections", headers={"X-JournalPulse-User": USER_A}).json()["items"] == []


def test_can_retain_text_only_when_requested(tmp_path: Path):
    app = create_app(settings=settings(tmp_path))
    with TestClient(app) as client:
        record = client.post(
            "/v1/reflections",
            json=reflection_payload(retain_text=True),
            headers={"X-JournalPulse-User": USER_A},
        ).json()
        assert record["text"] == reflection_payload()["text"]
        assert record["text_retained"] is True


def test_canadian_support_mode_bypasses_normal_policy_and_llm(tmp_path: Path):
    app = create_app(settings=settings(tmp_path))
    payload = reflection_payload(
        text="I have a plan to kill myself tonight and I might act on it.",
        llm_consent=True,
        retain_text=False,
    )
    with TestClient(app) as client:
        response = client.post("/v1/reflections", json=payload, headers={"X-JournalPulse-User": USER_A})
        assert response.status_code == 201
        record = response.json()
        assert record["safety"]["mode"] == "support"
        assert record["safety"]["exploration_allowed"] is False
        assert "9-8-8" in record["safety"]["support_message"]
        assert record["decision"]["policy_name"] == "safety-router"
        assert record["model_run"]["fallback_reason"] == "support_mode_llm_bypassed"


def test_health_is_liveness_only_and_readiness_is_explicit(tmp_path: Path):
    app = create_app(settings=settings(tmp_path))
    with TestClient(app) as client:
        assert client.get("/health").json() == {"status": "ok"}
        readiness = client.get("/ready").json()
        assert readiness["status"] == "not_ready"
        assert readiness["checks"]["llm"] == "not_ready:not_configured"


def test_analysis_is_transient_and_can_be_corrected_before_save(tmp_path: Path):
    app = create_app(settings=settings(tmp_path))
    with TestClient(app) as client:
        analysis = client.post(
            "/v1/reflections/analyze",
            json={
                "text": reflection_payload()["text"],
                "context": {"activity": "after work"},
                "llm_consent": False,
                "locale": "CA",
            },
        )
        assert analysis.status_code == 200
        assert client.get("/v1/reflections", headers={"X-JournalPulse-User": USER_A}).json()[
            "items"
        ] == []

        payload = reflection_payload(prepared_analysis=analysis.json())
        payload["self_report"]["agency"] = 0.72
        saved = client.post(
            "/v1/reflections", json=payload, headers={"X-JournalPulse-User": USER_A}
        ).json()
        assert saved["state"]["agency"] == 0.72


def test_export_and_delete_all_user_data_are_isolated(tmp_path: Path):
    app = create_app(settings=settings(tmp_path))
    with TestClient(app) as client:
        first = client.post(
            "/v1/reflections", json=reflection_payload(), headers={"X-JournalPulse-User": USER_A}
        ).json()
        client.post(
            "/v1/outcomes",
            headers={"X-JournalPulse-User": USER_A},
            json={"decision_id": first["decision"]["decision_id"], "completed": False},
        )
        client.post(
            "/v1/reflections", json=reflection_payload(), headers={"X-JournalPulse-User": USER_B}
        )

        exported = client.get("/v1/export", headers={"X-JournalPulse-User": USER_A}).json()
        assert len(exported["reflections"]) == 1
        assert len(exported["outcomes"]) == 1

        deleted = client.delete("/v1/account/data", headers={"X-JournalPulse-User": USER_A})
        assert deleted.status_code == 200
        assert deleted.json()["deleted_records"] == 2
        assert deleted.json()["auth_identity_deleted"] is False
        assert client.get("/v1/export", headers={"X-JournalPulse-User": USER_A}).json() == {
            "reflections": [],
            "outcomes": [],
        }
        assert len(
            client.get("/v1/reflections", headers={"X-JournalPulse-User": USER_B}).json()["items"]
        ) == 1


def test_outcome_cannot_be_attached_to_another_users_decision(tmp_path: Path):
    app = create_app(settings=settings(tmp_path))
    with TestClient(app) as client:
        other_record = client.post(
            "/v1/reflections", json=reflection_payload(), headers={"X-JournalPulse-User": USER_B}
        ).json()
        response = client.post(
            "/v1/outcomes",
            headers={"X-JournalPulse-User": USER_A},
            json={"decision_id": other_record["decision"]["decision_id"], "completed": True},
        )
        assert response.status_code == 404
        assert response.json()["detail"] == "Policy decision not found"


def test_action_preview_and_user_override_preserve_policy_provenance(tmp_path: Path):
    app = create_app(settings=settings(tmp_path))
    with TestClient(app) as client:
        preview = client.post(
            "/v1/actions/preview",
            json={
                "state": reflection_payload()["self_report"],
                "target": reflection_payload()["target"],
                "context": {"activity": "after work"},
                "resource_intent": "reflect",
            },
        )
        assert preview.status_code == 200
        choices = preview.json()["actions"]
        assert len(choices) == 3
        assert all(item["url"].startswith("https://") for item in choices)

        recommended = preview.json()["decision"]["action_id"]
        alternative = next(item["id"] for item in choices if item["id"] != recommended)
        saved = client.post(
            "/v1/reflections",
            headers={"X-JournalPulse-User": USER_A},
            json=reflection_payload(chosen_action_id=alternative),
        )
        assert saved.status_code == 201
        decision = saved.json()["decision"]
        assert decision["action_id"] == alternative
        assert decision["recommended_action_id"] == recommended
        assert decision["selection_source"] == "user_override"
        assert decision["eligible_for_ope"] is False


def test_unsafe_action_choice_is_rejected(tmp_path: Path):
    app = create_app(settings=settings(tmp_path))
    with TestClient(app) as client:
        response = client.post(
            "/v1/reflections",
            headers={"X-JournalPulse-User": USER_A},
            json=reflection_payload(chosen_action_id="model-generated-url"),
        )
        assert response.status_code == 422
        assert response.json()["detail"] == "Chosen action is not in the safe set"


def test_only_one_check_in_is_accepted_per_decision(tmp_path: Path):
    app = create_app(settings=settings(tmp_path))
    with TestClient(app) as client:
        record = client.post(
            "/v1/reflections",
            headers={"X-JournalPulse-User": USER_A},
            json=reflection_payload(),
        ).json()
        payload = {"decision_id": record["decision"]["decision_id"], "completed": True}
        first = client.post("/v1/outcomes", headers={"X-JournalPulse-User": USER_A}, json=payload)
        duplicate = client.post(
            "/v1/outcomes", headers={"X-JournalPulse-User": USER_A}, json=payload
        )
        assert first.status_code == 201
        assert duplicate.status_code == 409
        assert duplicate.json()["detail"] == "Check-in already recorded"
