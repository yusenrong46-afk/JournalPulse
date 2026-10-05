"""Explicit first-person future intent and phrase-scoped denials, without provider calls."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from journalpulse.api import create_app
from journalpulse.domain import SafetyMode
from journalpulse.safety import assess_safety
from test_conversations_api import ScriptedClient, chat_settings, say, start


@pytest.mark.parametrize("text", [
    "I will kill myself tonight.",
    "I'll kill myself tonight.",
    "I will end my life tonight.",
    "I’ll end my life tonight.",
    "I don't want to die and I'll kill myself tonight.",
    "I will not kill myself, but I'll end my life tonight.",
])
def test_explicit_future_intent_routes_to_support_despite_separate_denials(text: str):
    result = assess_safety(text, "CA")
    assert result.mode == SafetyMode.SUPPORT
    assert result.exploration_allowed is False


@pytest.mark.parametrize("text", [
    "I will not kill myself.",
    "I won't end my life.",
    "I don't want to kill myself.",
    "I do not want to end my life.",
    "I never want to kill myself.",
])
def test_obvious_direct_denials_remain_normal(text: str):
    assert assess_safety(text, "CA").mode == SafetyMode.NORMAL


@pytest.mark.parametrize("text", [
    "I will kill myself tonight.",
    "I'll end my life tonight.",
    "I don’t want to die and I’ll kill myself tonight.",
])
def test_future_intent_api_bypasses_model_and_ordinary_actions(tmp_path: Path, text: str):
    model = ScriptedClient(offers=[True])
    with TestClient(create_app(settings=chat_settings(tmp_path), conversation_client=model)) as client:
        chat = start(client)
        response = say(client, chat["id"], text)
        assert response.status_code == 200
        body = response.json()
        assert body["conversation"]["safety_mode"] == "support"
        assert body["conversation"]["ready_for_action"] is False
        assert body["assistant_message"]["model_run"]["model"] == "safety-router"
        assert model.calls == []


def test_negated_direct_intent_api_retains_normal_chat(tmp_path: Path):
    model = ScriptedClient(offers=[False])
    with TestClient(create_app(settings=chat_settings(tmp_path), conversation_client=model)) as client:
        chat = start(client)
        response = say(client, chat["id"], "I do not want to end my life. I want to talk about a deadline.")
        assert response.status_code == 200
        assert response.json()["conversation"]["safety_mode"] == "normal"
        assert len(model.calls) == 1
