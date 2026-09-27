from dataclasses import replace
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
        readiness = client.get("/ready")
        assert readiness.status_code == 503
        assert readiness.json()["status"] == "not_ready"
        assert readiness.json()["checks"]["llm"] == "not_ready:not_configured"


def test_static_export_can_share_the_api_origin(tmp_path: Path, monkeypatch):
    web_dist = tmp_path / "web-out"
    history_dist = web_dist / "history"
    history_dist.mkdir(parents=True)
    (web_dist / "index.html").write_text("<h1>JournalPulse</h1>", encoding="utf-8")
    (web_dist / "app.js").write_text("const journalPulse = true;\n" * 100, encoding="utf-8")
    (history_dist / "index.html").write_text("<h1>History</h1>", encoding="utf-8")
    monkeypatch.setenv("JOURNALPULSE_WEB_DIST", str(web_dist))

    app = create_app(settings=settings(tmp_path))
    with TestClient(app) as client:
        assert client.get("/").text == "<h1>JournalPulse</h1>"
        assert client.get("/history/").text == "<h1>History</h1>"
        assert client.get("/app.js", headers={"Accept-Encoding": "gzip"}).headers[
            "content-encoding"
        ] == "gzip"
        assert client.get("/health").json() == {"status": "ok"}


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


def test_client_request_ids_make_retries_idempotent(tmp_path: Path):
    app = create_app(settings=settings(tmp_path))
    reflection_request_id = "30000000-0000-4000-8000-000000000001"
    outcome_request_id = "40000000-0000-4000-8000-000000000001"
    with TestClient(app) as client:
        payload = reflection_payload(client_request_id=reflection_request_id)
        first = client.post(
            "/v1/reflections", json=payload, headers={"X-JournalPulse-User": USER_A}
        )
        retried = client.post(
            "/v1/reflections", json=payload, headers={"X-JournalPulse-User": USER_A}
        )
        assert first.status_code == retried.status_code == 201
        assert first.json()["id"] == retried.json()["id"] == reflection_request_id
        history = client.get(
            "/v1/reflections", headers={"X-JournalPulse-User": USER_A}
        ).json()
        assert len(history["items"]) == 1

        outcome_payload = {
            "client_request_id": outcome_request_id,
            "decision_id": first.json()["decision"]["decision_id"],
            "completed": True,
        }
        outcome = client.post(
            "/v1/outcomes", json=outcome_payload, headers={"X-JournalPulse-User": USER_A}
        )
        outcome_retry = client.post(
            "/v1/outcomes", json=outcome_payload, headers={"X-JournalPulse-User": USER_A}
        )
        assert outcome.status_code == outcome_retry.status_code == 201
        assert outcome.json()["id"] == outcome_retry.json()["id"] == outcome_request_id


def test_request_guards_add_trace_headers_limit_size_and_rate(tmp_path: Path):
    configured = replace(settings(tmp_path), analysis_rate_limit_per_minute=1)
    app = create_app(settings=configured)
    with TestClient(app) as client:
        health = client.get("/health", headers={"X-Request-ID": "browser-request-123"})
        assert health.headers["x-request-id"] == "browser-request-123"
        assert health.headers["x-content-type-options"] == "nosniff"

        oversized = client.post(
            "/v1/reflections/analyze",
            content=b"x" * (configured.max_request_bytes + 1),
            headers={"Content-Type": "application/json"},
        )
        assert oversized.status_code == 413

        analysis_payload = {
            "text": "A complete thought that is long enough to inspect.",
            "llm_consent": False,
            "locale": "CA",
        }
        assert client.post("/v1/reflections/analyze", json=analysis_payload).status_code == 200
        limited = client.post("/v1/reflections/analyze", json=analysis_payload)
        assert limited.status_code == 429
        assert int(limited.headers["retry-after"]) >= 1


def test_system_status_explains_local_fallback_without_model_language(tmp_path: Path):
    app = create_app(settings=settings(tmp_path))
    with TestClient(app) as client:
        status = client.get("/v1/system/status").json()
        assert status["analysis_mode"] == "local_fallback"
        assert status["persistence_mode"] == "server_sqlite"
        assert "unavailable" in status["message"].lower()


def test_ready_returns_503_when_required_configuration_is_missing(tmp_path: Path):
    app = create_app(settings=settings(tmp_path))
    with TestClient(app) as client:
        response = client.get("/ready")
        assert response.status_code == 503
        body = response.json()
        assert body["status"] == "not_ready"
        assert body["checks"]["llm"] == "not_ready:not_configured"
        assert body["checks"]["persistence"] == "server_sqlite"


def test_ready_does_not_claim_an_unrun_provider_probe(tmp_path: Path):
    configured = replace(settings(tmp_path), openrouter_api_key="test-only-key")
    app = create_app(settings=configured)
    with TestClient(app) as client:
        response = client.get("/ready")
        assert response.status_code == 200
        assert response.json()["status"] == "ready"
        assert response.json()["checks"]["llm"] == "configured:not_probed"


def test_save_without_prepared_analysis_cannot_bypass_generation_limit(tmp_path: Path):
    class CountingClient:
        def __init__(self) -> None:
            self.calls = 0

        def analyze(self, text: str, context: dict[str, str]):
            del text, context
            self.calls += 1
            from journalpulse.intelligence import deterministic_reflection

            return deterministic_reflection("synthetic")

    counter = CountingClient()
    configured = replace(
        settings(tmp_path),
        analysis_rate_limit_per_minute=1,
        openrouter_api_key="test-only-key",
    )
    app = create_app(settings=configured, intelligence_client=counter)
    with TestClient(app) as client:
        analyzed = client.post(
            "/v1/reflections/analyze",
            json={
                "text": "A complete thought that is long enough to inspect.",
                "llm_consent": True,
                "locale": "CA",
            },
        )
        assert analyzed.status_code == 200
        assert counter.calls == 1
        saved = client.post(
            "/v1/reflections",
            json=reflection_payload(llm_consent=True),
            headers={"X-JournalPulse-User": USER_A},
        )
        assert saved.status_code == 429
        assert int(saved.headers["retry-after"]) >= 1
        assert counter.calls == 1


def test_forged_client_model_run_is_not_stored_as_verified_output(tmp_path: Path):
    app = create_app(settings=settings(tmp_path))
    forged = {
        "state": {
            "valence": -0.2,
            "arousal": 0.4,
            "agency": 0.3,
            "emotion_tags": ["avoidance"],
            "confidence": 0.42,
        },
        "reflection": {
            "summary": "Client summary that must stay distinguishable.",
            "interpretation": "Client interpretation.",
            "reflection_question": "What is one next step?",
        },
        "safety": {
            "mode": "support",
            "reasons": ["forged"],
            "locale": "US",
            "exploration_allowed": False,
            "support_message": "forged support message",
            "resource_ids": ["forged-resource"],
        },
        "model_run": {
            "model": "forged-model",
            "provider": "forged-provider",
            "latency_ms": 12,
            "schema_valid": True,
            "used_fallback": False,
        },
        "resource_intent": "reflect",
    }
    with TestClient(app) as client:
        saved = client.post(
            "/v1/reflections",
            json=reflection_payload(prepared_analysis=forged, llm_consent=True),
            headers={"X-JournalPulse-User": USER_A},
        )
        assert saved.status_code == 201
        record = saved.json()
        assert record["model_run"]["model"] == "unverified-client-analysis"
        assert record["model_run"]["provider"] == "client"
        assert record["model_run"]["schema_valid"] is False
        assert record["model_run"]["used_fallback"] is True
        assert record["model_run"]["fallback_reason"] == "unverified_client_analysis"
        assert record["model_run"]["model"] != "forged-model"
        assert record["safety"]["mode"] == "normal"
        assert record["safety"]["support_message"] is None
