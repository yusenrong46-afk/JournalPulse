"""Indirect first-person risk reaches support; clearly past risk does not lock the chat.

The router is still an English phrase list, not a classifier. These cases pin two
corrections from the 2026-10-05 UI audit and the deliberately cautious limits around them.
"""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from journalpulse.api import create_app
from journalpulse.domain import SafetyMode
from journalpulse.safety import assess_safety
from test_conversations_api import ScriptedClient, chat_settings, say, start

INDIRECT_RISK = [
    "Everyone would be better off without me",
    "my family would be better off without me.",
    "I've been thinking about ending it all",
    "I won't be around much longer",
    "I will not be here much longer",
    "I feel suicidal tonight",
    "I'm suicidal",
    "I am feeling suicidal again",
    "I keep thinking about killing myself",
    "I wish I were dead",
    "I don't want to be alive anymore",
]

PAST_RISK = [
    "I used to want to die but therapy helped",
    "Years ago I wanted to hurt myself; now I am fine",
    "When I was a teenager I would hurt myself. That is behind me.",
    "I used to feel suicidal, and that changed with support",
]

PAST_RISK_THAT_IS_ALSO_PRESENT = [
    "I used to want to die and now I want to die again",
    "Years ago I wanted to hurt myself and I still do",
    "When I was younger I would hurt myself, and tonight I want to hurt myself",
    "I used to want to die. I want to die.",
    "I used to feel suicidal and I feel suicidal again",
]

# Reported risk about someone else keeps the cautious support route: crisis lines also
# help people who are worried about another person.
REPORTED_RISK = [
    'My friend said "I want to die"',
    "In the movie he says I will kill myself",
]


@pytest.mark.parametrize("text", INDIRECT_RISK)
def test_indirect_first_person_risk_routes_to_support(text: str):
    assert assess_safety(text, "CA").mode == SafetyMode.SUPPORT


@pytest.mark.parametrize("text", PAST_RISK)
def test_clearly_past_risk_stays_in_normal_chat(text: str):
    assert assess_safety(text, "CA").mode == SafetyMode.NORMAL


@pytest.mark.parametrize("text", PAST_RISK_THAT_IS_ALSO_PRESENT)
def test_a_past_marker_never_hides_present_risk(text: str):
    assert assess_safety(text, "CA").mode == SafetyMode.SUPPORT


@pytest.mark.parametrize("text", REPORTED_RISK)
def test_reported_risk_still_errs_toward_support(text: str):
    assert assess_safety(text, "CA").mode == SafetyMode.SUPPORT


@pytest.mark.parametrize("text", [
    "This job is killing me.",
    "The commute will be the end of me.",
    "I'll be around later if you want to talk.",
    "We would be better off without this meeting.",
    "My suitcase won't be around much longer, it is falling apart.",
])
def test_everyday_phrases_near_the_new_patterns_stay_normal(text: str):
    assert assess_safety(text, "CA").mode == SafetyMode.NORMAL


def test_indirect_risk_api_routes_to_support_without_a_model_call(tmp_path: Path):
    model = ScriptedClient(offers=[True])
    with TestClient(create_app(settings=chat_settings(tmp_path), conversation_client=model)) as client:
        chat = start(client)
        body = say(client, chat["id"], "Honestly everyone would be better off without me.").json()
        assert body["conversation"]["safety_mode"] == "support"
        assert body["assistant_message"]["model_run"]["model"] == "safety-router"
        assert model.calls == []


def test_past_risk_api_keeps_the_conversation_open_to_luna(tmp_path: Path):
    model = ScriptedClient(offers=[False])
    with TestClient(create_app(settings=chat_settings(tmp_path), conversation_client=model)) as client:
        chat = start(client)
        text = "I used to want to die but therapy helped. Work is stressful now."
        body = say(client, chat["id"], text).json()
        assert body["conversation"]["safety_mode"] == "normal"
        assert len(model.calls) == 1
