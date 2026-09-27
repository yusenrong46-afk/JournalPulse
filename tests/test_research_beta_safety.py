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


def test_safety_negation_does_not_clear_risk_in_another_clause():
    mixed_sentences = assess_safety(
        "I am not suicidal. I plan to kill myself tonight.",
        "US",
    )
    assert mixed_sentences.mode == SafetyMode.SUPPORT
    mixed_clauses = assess_safety(
        "I don't want to die, but I will overdose tonight.",
        "US",
    )
    assert mixed_clauses.mode == SafetyMode.SUPPORT
