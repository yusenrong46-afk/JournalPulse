"""The safety gate is a small set of phrase patterns with phrase-scoped negation. It routes
clear, explicit risk language to human support before any model call. It is not a
classifier and does not understand meaning; the cases below pin its intended behaviour,
including where it deliberately errs toward support and where it is known to miss."""

import pytest

from journalpulse.domain import SafetyMode
from journalpulse.safety import assess_safety


def test_safety_detects_explicit_risk_and_respects_locale():
    result = assess_safety("I have a plan to end my life and I might act on it", "CA")
    assert result.mode == SafetyMode.SUPPORT
    assert result.exploration_allowed is False
    assert "support_988_canada" in result.resource_ids


def test_safety_does_not_trigger_on_clear_negation():
    result = assess_safety("I am not suicidal and do not want to hurt myself", "CA")
    assert result.mode == SafetyMode.NORMAL
    assert result.exploration_allowed is True


def test_safety_matches_lowercased_risk_phrases():
    plan = assess_safety("I have a suicide plan", "US")
    assert plan.mode == SafetyMode.SUPPORT
    thoughts = assess_safety("I might act on these thoughts", "US")
    assert thoughts.mode == SafetyMode.SUPPORT


@pytest.mark.parametrize("text", [
    "I don't feel safe", "I don't feel safe.", "I don’t feel safe.",
    "I do not feel safe.", "I do not feel safe; please help.",
])
def test_unsafe_feeling_without_a_time_suffix_routes_to_support(text: str):
    result = assess_safety(text, "CA")
    assert result.mode == SafetyMode.SUPPORT
    assert result.exploration_allowed is False


def test_safety_negation_does_not_clear_risk_in_another_clause():
    mixed_sentences = assess_safety("I am not suicidal. I plan to kill myself tonight.", "US")
    assert mixed_sentences.mode == SafetyMode.SUPPORT
    mixed_clauses = assess_safety("I don't want to die, but I will overdose tonight.", "US")
    assert mixed_clauses.mode == SafetyMode.SUPPORT


@pytest.mark.parametrize(
    "text",
    [
        "I DON'T FEEL SAFE TONIGHT",
        "never suicidal; i plan to end my life",
        "I don't want to die but I want to kill myself",
        "i'm fine, i want to die",
        "Honestly?   I   want   to   end   my   life.",
        "I'm not okay and I don't feel safe alone",
    ],
)
def test_adversarial_phrasings_still_route_to_support(text: str):
    assert assess_safety(text, "CA").mode == SafetyMode.SUPPORT


@pytest.mark.parametrize(
    "text",
    [
        "I don't want to die. I just want to sleep for a week.",
        "I'm not suicidal, just exhausted, and the deadline is killing me.",
        "This spreadsheet makes me want to scream.",
    ],
)
def test_ordinary_frustration_stays_in_the_normal_chat(text: str):
    assert assess_safety(text, "CA").mode == SafetyMode.NORMAL


def test_unrecognised_negation_errs_toward_support():
    # "not going to" is not a known negation, so the gate stays cautious.
    assert assess_safety("I'm not going to kill myself", "CA").mode == SafetyMode.SUPPORT


@pytest.mark.parametrize(
    "text",
    [
        "I've been thinking about ending it all",
        "Everyone would be better off without me",
    ],
)
def test_known_gaps_are_documented_not_hidden(text: str):
    # Indirect language is outside the pattern list. The gate is not a classifier, and
    # docs/ARCHITECTURE.md says so; these assertions record the current gap.
    assert assess_safety(text, "CA").mode == SafetyMode.NORMAL


def test_unknown_locale_falls_back_to_international_help():
    result = assess_safety("I want to end my life", "ZZ")
    assert result.mode == SafetyMode.SUPPORT
    assert result.resource_ids == ["support_befrienders"]
