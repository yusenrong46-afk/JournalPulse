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


@pytest.mark.parametrize("text", [
    "I used to feel suicidal and I will kill myself.",
    "I used to feel suicidal and I want to kill myself.",
    "When I was a student I liked art and I will kill myself.",
    "Years ago I hurt myself and I might act on these thoughts.",
    "I used to feel suicidal, so I will kill myself.",
    "I used to feel better because I want to die.",
])
def test_separate_current_clause_is_not_made_historical(text: str, tmp_path: Path):
    model = ScriptedClient(offers=[True])
    with TestClient(create_app(settings=chat_settings(tmp_path), conversation_client=model)) as client:
        chat = start(client)
        response = say(client, chat["id"], text)
        assert response.status_code == 200
        body = response.json()
        assert body["conversation"]["safety_mode"] == "support"
        assert body["conversation"]["activity_card"] is None
        assert model.calls == []


@pytest.mark.parametrize("text", [
    "I used to want to die and hurt myself, but support helped.",
    "When I was a student I wanted to hurt myself and felt suicidal.",
])
def test_shared_historical_predicates_remain_historical(text: str):
    assert assess_safety(text).mode == SafetyMode.NORMAL


@pytest.mark.parametrize("current", [
    "I plan to kill myself", "I made a suicide plan", *INDIRECT_RISK,
])
def test_historical_prefix_does_not_suppress_recognized_new_subject(current, tmp_path):
    # Every standalone recognized current statement must stay recognized after
    # an independent historical clause, including indirect non-I subjects.
    assert assess_safety(current).mode == SafetyMode.SUPPORT
    model = ScriptedClient(offers=[True])
    with TestClient(create_app(settings=chat_settings(tmp_path), conversation_client=model)) as client:
        chat = start(client)
        response = say(client, chat["id"], f"I used to feel suicidal and {current}.")
        assert response.status_code == 200
        body = response.json()
        assert body["conversation"]["safety_mode"] == "support"
        assert body["conversation"]["activity_card"] is None
        assert model.calls == []


@pytest.mark.parametrize("current", [
    "I really want to kill myself", "I honestly want to die", "I desperately want to end my life",
    "I think I might act on these thoughts", "sometimes I want to die",
    "after everything I feel suicidal", "honestly everyone would be better off without me",
    "I would be better off if I were dead", "I would like to end it all",
])
def test_current_subject_is_independent_of_modifiers(current, tmp_path):
    assert assess_safety(current).mode == SafetyMode.SUPPORT
    model = ScriptedClient(offers=[True])
    with TestClient(create_app(settings=chat_settings(tmp_path), conversation_client=model)) as client:
        chat = start(client)
        response = say(client, chat["id"], f"I used to feel suicidal and {current}.")
        assert response.status_code == 200
        assert response.json()["conversation"]["safety_mode"] == "support"
        assert response.json()["conversation"]["activity_card"] is None
        assert model.calls == []


@pytest.mark.parametrize("text", [
    "I used to feel suicidal and I used to want to die.",
    "Years ago I wanted to hurt myself and back then I felt suicidal.",
    "When I was a student I wanted to hurt myself and I felt suicidal.",
    "I used to want to die and I was suicidal.",
    "I used to feel suicidal and I really do not want to die.",
    "I used to feel suicidal and I am not suicidal.",
])
def test_independent_subject_handling_preserves_history_and_denial(text):
    assert assess_safety(text).mode == SafetyMode.NORMAL


@pytest.mark.parametrize("text", [
    "Back then I would hurt myself.",
    "Back then I felt suicidal.",
    "Back then I was suicidal.",
    "I used to feel suicidal and back then I would hurt myself.",
])
def test_historical_marker_is_not_a_coordinating_then(text, tmp_path):
    model = ScriptedClient(offers=[False])
    with TestClient(create_app(settings=chat_settings(tmp_path), conversation_client=model)) as client:
        chat = start(client)
        response = say(client, chat["id"], text)
        assert response.status_code == 200
        assert response.json()["conversation"]["safety_mode"] == "normal"
        assert len(model.calls) == 1


def test_coordinating_then_still_separates_current_risk():
    assert assess_safety("I used to feel suicidal then I really want to die.").mode == SafetyMode.SUPPORT


@pytest.mark.parametrize("noun", ["feedback", "pushback", "setback", "flashback"])
def test_historical_back_is_a_complete_word(noun, tmp_path):
    model = ScriptedClient(offers=[True])
    with TestClient(create_app(settings=chat_settings(tmp_path), conversation_client=model)) as client:
        chat = start(client)
        text = f"I used to feel suicidal after negative {noun} then I really want to die."
        response = say(client, chat["id"], text)
        assert response.status_code == 200
        assert response.json()["conversation"]["safety_mode"] == "support"
        assert response.json()["conversation"]["activity_card"] is None
        assert model.calls == []


PUNCTUATED_CLAUSE_BOUNDARIES = [
    ": ", " — ", " – ", " - ", " -- ", "\n", "\r\n", "—", ":\n", " -", "- ", "--", " / ",
]


@pytest.mark.parametrize("boundary", PUNCTUATED_CLAUSE_BOUNDARIES)
@pytest.mark.parametrize("history", [
    "I used to feel suicidal", "When I was a student I liked art",
])
@pytest.mark.parametrize("current", [
    "I will kill myself", "I really want to die", "I think I might act on these thoughts",
    "everyone would be better off without me", "my family would be better off without me",
    "I am not going to be around much longer",
])
def test_punctuation_does_not_extend_history_to_an_independent_current_subject(
    boundary: str, history: str, current: str,
):
    assert assess_safety(current).mode == SafetyMode.SUPPORT
    assert assess_safety(f"{history}{boundary}{current}.").mode == SafetyMode.SUPPORT


@pytest.mark.parametrize("boundary", PUNCTUATED_CLAUSE_BOUNDARIES)
@pytest.mark.parametrize("history, continuation", [
    ("I used to want to die", "and hurt myself"),
    ("When I was a student I wanted to hurt myself", "I felt suicidal"),
    ("When I was a student I wanted to hurt myself", "honestly I was suicidal"),
    ("I used to feel suicidal", "I used to want to die"),
    ("I used to feel suicidal", "I really do not want to die"),
    ("I used to feel suicidal", "I am not suicidal"),
])
def test_punctuation_preserves_shared_past_predicates_and_clear_denials(
    boundary: str, history: str, continuation: str,
):
    assert assess_safety(f"{history}{boundary}{continuation}.").mode == SafetyMode.NORMAL


@pytest.mark.parametrize("boundary", PUNCTUATED_CLAUSE_BOUNDARIES)
@pytest.mark.parametrize("history", ["When I was a teenager", "Years ago"])
def test_a_punctuated_history_heading_still_qualifies_a_past_habit(boundary: str, history: str):
    assert assess_safety(f"{history}{boundary}I would hurt myself.").mode == SafetyMode.NORMAL


@pytest.mark.parametrize("boundary", PUNCTUATED_CLAUSE_BOUNDARIES)
@pytest.mark.parametrize("correction", [
    "actually I want to die", "to be clear I will kill myself", "I still do",
])
def test_punctuated_correction_can_establish_present_risk(boundary: str, correction: str):
    assert assess_safety(f"I used to want to die{boundary}{correction}.").mode == SafetyMode.SUPPORT


@pytest.mark.parametrize("text, mode", [
    ("I want to\ndie.", SafetyMode.SUPPORT),
    ("I do not\nwant to die.", SafetyMode.NORMAL),
    ("I am not\nsuicidal.", SafetyMode.NORMAL),
    ("When I was a\nstudent I wanted to hurt myself.", SafetyMode.NORMAL),
    ("I used to feel suicidal\nI\nwill kill myself.", SafetyMode.SUPPORT),
])
def test_wrapped_lines_keep_risk_phrases_and_their_qualifiers_intact(text: str, mode: SafetyMode):
    assert assess_safety(text).mode == mode


@pytest.mark.parametrize("boundary", [": ", " — ", " – ", "\n"])
@pytest.mark.parametrize("history", [
    "I used to feel suicidal", "When I was a student I liked art",
])
def test_punctuated_current_risk_api_bypasses_model_and_ordinary_actions(
    tmp_path: Path, boundary: str, history: str,
):
    model = ScriptedClient(offers=[True])
    with TestClient(create_app(settings=chat_settings(tmp_path), conversation_client=model)) as client:
        chat = start(client)
        response = say(client, chat["id"], f"{history}{boundary}I will kill myself.")
        assert response.status_code == 200
        body = response.json()
        assert body["conversation"]["safety_mode"] == "support"
        assert body["conversation"]["ready_for_action"] is False
        assert body["conversation"]["activity_card"] is None
        assert body["assistant_message"]["model_run"]["model"] == "safety-router"
        assert model.calls == []


@pytest.mark.parametrize("current", [
    "my\nfamily would be better off without me",
    "the\nworld would be better off without me",
    "although I was happy earlier I want to die",
    "although I was happy earlier I think I might act on these thoughts",
])
@pytest.mark.parametrize("boundary", ["\n", ": ", " -", "- "])
def test_wrapped_or_embedded_current_subject_keeps_its_own_scope(current: str, boundary: str):
    assert assess_safety(current).mode == SafetyMode.SUPPORT
    text = f"I used to feel suicidal{boundary}{current}"
    assert assess_safety(text).mode == SafetyMode.SUPPORT


@pytest.mark.parametrize("text", [
    "When I\nwas a teenager I would hurt myself",
    "I used\nto think my family would be better off without me",
    "When I was a student I thought everyone would be better off without me",
    "I used to feel suicidal: I thought my family would be better off without me",
    "I used to feel suicidal: I felt like everyone would be better off without me",
    "I used to feel suicidal: although I was happy earlier I do not want to die",
])
def test_current_subject_review_preserves_history_and_denial(text: str):
    assert assess_safety(text).mode == SafetyMode.NORMAL


@pytest.mark.parametrize("text", [
    "When I was a student I told my family I want to die",
    "I used to think about how everyone would be better off without me",
    "I used to tell myself I want to die",
    "Years ago I felt that my family would be better off without me",
    "I used to be convinced my family would be better off without me",
    "I used to feel suicidal: I thought about how my family would be better off without me",
])
def test_explicit_history_is_not_split_at_subjects_within_a_reported_thought(text: str):
    assert assess_safety(text).mode == SafetyMode.NORMAL


@pytest.mark.parametrize("belief", ["convinced", "sure"])
@pytest.mark.parametrize("boundary", [": ", " — ", "\n", " and "])
@pytest.mark.parametrize("complement", ["my family", "that my family"])
def test_past_belief_complement_retains_historical_scope(belief: str, boundary: str, complement: str):
    text = f"When I was a teenager{boundary}I was {belief} {complement} would be better off without me"
    assert assess_safety(text).mode == SafetyMode.NORMAL


@pytest.mark.parametrize("belief", ["convinced", "sure"])
@pytest.mark.parametrize("prefix", ["", "I used to feel suicidal: "])
def test_current_belief_complement_still_reaches_support(belief: str, prefix: str):
    text = f"{prefix}I am {belief} my family would be better off without me"
    assert assess_safety(text).mode == SafetyMode.SUPPORT


@pytest.mark.parametrize("belief", ["convinced", "sure"])
@pytest.mark.parametrize("boundary", [" ", " and ", " — ", "\n"])
def test_independent_current_risk_after_past_belief_is_not_hidden(belief: str, boundary: str):
    text = (
        f"When I was a teenager: I was {belief} my family would be better off without me"
        f"{boundary}I want to die"
    )
    assert assess_safety(text).mode == SafetyMode.SUPPORT


@pytest.mark.parametrize("belief", ["convinced", "sure"])
@pytest.mark.parametrize("predicate, mode", [("was", "normal"), ("am", "support")])
def test_belief_complement_api_preserves_past_and_current_routes(
    tmp_path: Path, belief: str, predicate: str, mode: str,
):
    model = ScriptedClient(offers=[False])
    text = f"When I was a teenager: I {predicate} {belief} my family would be better off without me"
    with TestClient(create_app(settings=chat_settings(tmp_path), conversation_client=model)) as client:
        chat = start(client)
        response = say(client, chat["id"], text)
        assert response.status_code == 200
        body = response.json()
        assert body["conversation"]["safety_mode"] == mode
        assert len(model.calls) == (1 if mode == "normal" else 0)
        assert body["conversation"]["ready_for_action"] is False
        assert body["conversation"]["activity_card"] is None


@pytest.mark.parametrize("text", [
    "I used to feel suicidal\nmy\nfamily would be better off without me",
    "I used to feel suicidal\nthe\nworld would be better off without me",
    "I used to feel suicidal -I want to die",
    "I used to feel suicidal- I want to die",
    "I used to feel suicidal: although I was happy earlier I want to die",
])
def test_rechecked_current_risk_api_has_no_model_call_or_activity(tmp_path: Path, text: str):
    model = ScriptedClient(offers=[True])
    with TestClient(create_app(settings=chat_settings(tmp_path), conversation_client=model)) as client:
        chat = start(client)
        response = say(client, chat["id"], text)
        assert response.status_code == 200
        body = response.json()
        assert body["conversation"]["safety_mode"] == "support"
        assert body["conversation"]["ready_for_action"] is False
        assert body["conversation"]["activity_card"] is None
        assert model.calls == []
