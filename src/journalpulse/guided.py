"""Luna without a language model.

Used when the person has not allowed AI processing or no model is configured. The
replies are fixed, the feeling guesses come from keyword matching, and nothing the
person wrote is copied into the summary, so text retention rules stay simple.
"""

from __future__ import annotations

import re

from .domain import FEELINGS, Goal, ModelRun
from .intelligence import ConversationCompletion

GUIDED_PROMPT_VERSION = "guided-2026-09-27.1"

GUIDED_REPLIES: tuple[str, ...] = (
    "Thank you for telling me. What part of that is sitting with you most right now?",
    "That makes sense. When you notice it, where does it show up most: in your body, your "
    "thoughts, or your energy?",
    "Thank you for sharing that with me. I think I have a sense of it. Want to find one small "
    "thing to try together?",
)

GUIDED_SUMMARY = "You checked in with Luna and talked through what was on your mind."

_FEELING_PATTERNS: dict[str, re.Pattern[str]] = {
    "tired": re.compile(r"\b(tired|exhaust\w*|drained|sleepy|worn out|fatigue\w*)\b"),
    "anxious": re.compile(r"\b(anxious|anxiety|nervous|worr\w+|panic\w*|on edge|scared|afraid)\b"),
    "stressed": re.compile(r"\b(stress\w*|pressure\w*|deadline\w*|swamped|busy)\b"),
    "sad": re.compile(r"\b(sad|down|cry\w*|cried|blue|grief|griev\w*|hurt\w*|upset)\b"),
    "frustrated": re.compile(r"\b(frustrat\w*|angry|anger|annoy\w*|mad|irritat\w*|furious)\b"),
    "lonely": re.compile(r"\b(lonely|alone|isolat\w*|left out|no one)\b"),
    "overwhelmed": re.compile(r"\b(overwhelm\w*|too much|can't cope|cannot cope|drowning)\b"),
    "numb": re.compile(r"\b(numb|empty|blank|nothing matters|flat)\b"),
    "calm": re.compile(r"\b(calm|peaceful|relaxed|settled)\b"),
    "hopeful": re.compile(r"\b(hope\w*|optimis\w*|looking forward)\b"),
    "okay": re.compile(r"\b(okay|ok|fine|alright|so-so|meh)\b"),
    "happy": re.compile(r"\b(happy|glad|great|good|joy\w*|excited)\b"),
}

GOAL_PHRASES: dict[Goal, str] = {
    Goal.SETTLE: "calm things down",
    Goal.MOVE: "get a little energy back",
    Goal.UNDERSTAND: "make sense of it",
    Goal.CONNECT: "feel a little less alone",
    Goal.ACT: "take one small step",
}


def guess_feelings(texts: list[str]) -> list[str]:
    combined = " ".join(texts).lower()
    found = [name for name, pattern in _FEELING_PATTERNS.items() if pattern.search(combined)]
    if "happy" in found and any(item in found for item in ("sad", "anxious", "overwhelmed")):
        # "not good" and similar phrases are far more common than mixed joy in a check-in.
        found.remove("happy")
    ordered = [name for name in FEELINGS if name in found]
    return ordered[:3]


def guided_completion(user_texts: list[str]) -> ConversationCompletion:
    """Reply for the next turn. user_texts includes the message being answered."""
    turn = max(len(user_texts) - 1, 0)
    reply = GUIDED_REPLIES[min(turn, len(GUIDED_REPLIES) - 1)]
    ready = turn >= len(GUIDED_REPLIES) - 1
    return ConversationCompletion(
        reply=reply,
        offer_action=ready,
        resource_intent="reflect",
        card_reason="A few gentle options from the reviewed list." if ready else "",
        summary=GUIDED_SUMMARY,
        feelings=tuple(guess_feelings(user_texts)),
        model_run=guided_model_run("guided_mode"),
    )


def goal_reply(goal: Goal) -> str:
    return (
        f"Here are three small ways to {GOAL_PHRASES[goal]}. My pick is on top, but any of them "
        "is a good choice. Tap one when you're ready."
    )


def goal_card_reason(goal: Goal) -> str:
    return f"You said you'd like to {GOAL_PHRASES[goal]}. These come from the reviewed list."


def guided_model_run(reason: str) -> ModelRun:
    return ModelRun(
        model="luna-guided",
        provider="local",
        latency_ms=0,
        schema_valid=True,
        used_fallback=True,
        fallback_reason=reason,
        prompt_version=GUIDED_PROMPT_VERSION,
    )
