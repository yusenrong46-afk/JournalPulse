"""Local guided commands remain conservative rules, not model-quality evidence."""

import pytest

from journalpulse.guided import guided_completion


def test_guided_turn_count_does_not_create_action_readiness():
    texts = ["I'm frustrated.", "I rearranged my day.", "I still want to understand.", "The timing mattered."]
    result = guided_completion(texts)
    assert result.offer_action is False
    assert "small thing" not in result.reply


@pytest.mark.parametrize("text", [
    "Please stop and don't ask another question.", "That's enough for now.",
    "Let's pause here.", "Stop.",
])
def test_clear_guided_stop_is_acknowledged_without_a_question(text: str):
    result = guided_completion(["My day felt frustrating.", text])
    assert result.offer_action is False
    assert "?" not in result.reply


@pytest.mark.parametrize("text", [
    "I'd like to find one small thing.", "I'm ready to try one small step.",
    "Can you suggest an activity?",
])
def test_clear_guided_action_request_can_offer_the_next_control(text: str):
    result = guided_completion([text])
    assert result.offer_action is True
    assert "?" not in result.reply


@pytest.mark.parametrize("text", [
    "No action now, just help me understand.", "Don't suggest an activity.",
    "I am not ready to try one small thing.", "The trains stop near my office.",
])
def test_refusal_and_incidental_words_do_not_offer_actions(text: str):
    assert guided_completion(["Earlier I wanted an activity.", text]).offer_action is False


def test_listening_preference_overrides_a_matching_command():
    assert guided_completion(["Find one small thing."], listening=True).offer_action is False
