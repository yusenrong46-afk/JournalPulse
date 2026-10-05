"""Regressions found by the October foundation audit; inputs are fictional."""

from dataclasses import replace
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from journalpulse.api import create_app
from journalpulse.config import Settings
from journalpulse.domain import SafetyMode
from journalpulse.safety import assess_safety


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        environment="test",
        database_path=tmp_path / "audit.db",
        resource_catalog_path=Path(__file__).resolve().parents[1] / "assets/resources/catalog.json",
        openrouter_api_key=None,
        openrouter_model="test-model",
        openrouter_base_url="https://provider.example",
        openrouter_zdr=True,
        openrouter_timeout_seconds=2,
        supabase_url=None,
        supabase_anon_key=None,
        raw_text_retention_default=False,
    )


def test_completion_rate_counts_completed_actions_not_submitted_checkins(tmp_path: Path) -> None:
    with TestClient(create_app(settings=_settings(tmp_path))) as client:
        for completed in (False, True):
            reflection = client.post(
                "/v1/reflections",
                json={"text": "A fictional afternoon felt busy.", "target": {"goal": "settle"}},
            )
            assert reflection.status_code == 201
            outcome = client.post(
                "/v1/outcomes",
                json={
                    "decision_id": reflection.json()["decision"]["decision_id"],
                    "completed": completed,
                },
            )
            assert outcome.status_code == 201
        insights = client.get("/v1/insights").json()
        assert insights["completed_outcomes"] == 1
        assert insights["completion_rate"] == 0.5
        # A declined action has still been checked in; it should not be asked again.
        assert insights["pending_decision_ids"] == []


def test_support_analysis_remains_available_after_generation_quota(tmp_path: Path) -> None:
    settings = replace(_settings(tmp_path), analysis_rate_limit_per_minute=1)
    with TestClient(create_app(settings=settings)) as client:
        normal = {"text": "A fictional afternoon felt busy."}
        assert client.post("/v1/reflections/analyze", json=normal).status_code == 200
        assert client.post("/v1/reflections/analyze", json=normal).status_code == 429
        supported = client.post(
            "/v1/reflections/analyze", json={"text": "I have a suicide plan", "locale": "CA"}
        )
        assert supported.status_code == 200
        assert supported.json()["safety"]["mode"] == "support"
        assert supported.json()["model_run"]["model"] == "safety-router"


def test_support_analysis_does_not_consume_generation_quota(tmp_path: Path) -> None:
    settings = replace(_settings(tmp_path), analysis_rate_limit_per_minute=1)
    with TestClient(create_app(settings=settings)) as client:
        supported = client.post(
            "/v1/reflections/analyze", json={"text": "I have a suicide plan"}
        )
        assert supported.status_code == 200
        normal = client.post(
            "/v1/reflections/analyze", json={"text": "A fictional afternoon felt busy."}
        )
        assert normal.status_code == 200


@pytest.mark.parametrize(
    "text",
    [
        "I am not suicidal and I plan to kill myself tonight",
        "I don't want to die and I might act on these thoughts",
        "I don't want to harm myself and I want to die",
        "I don’t feel safe tonight",
    ],
)
def test_unrelated_negation_and_typographic_apostrophes_do_not_hide_explicit_risk(text: str) -> None:
    assert assess_safety(text).mode == SafetyMode.SUPPORT


@pytest.mark.parametrize("text", ["I don't want to die", "I am not suicidal and do not want to hurt myself"])
def test_explicit_denials_remain_normal(text: str) -> None:
    assert assess_safety(text).mode == SafetyMode.NORMAL


@pytest.mark.parametrize(
    "text",
    ["Yesterday I said I want to die", 'The quoted message said "I have a suicide plan"'],
)
def test_historical_or_quoted_explicit_risk_remains_conservative(text: str) -> None:
    # The phrase router does not infer whether quoted or historical risk is resolved.
    assert assess_safety(text).mode == SafetyMode.SUPPORT
