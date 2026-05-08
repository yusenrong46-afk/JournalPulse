import hashlib
from typing import Iterable, List, Optional

from .config import (
    CRISIS_DISCLAIMER,
    CRISIS_INTERPRETATION,
    CRISIS_RECOMMENDATION,
    CRISIS_REFLECTION_SUMMARY,
    CRISIS_SUPPORT_MESSAGE,
    DEFAULT_DISCLAIMER,
)
from .preprocessing import contains_crisis_language, normalize_text

RECOMMENDATION_HEADLINES = {
    "sadness": [
        "Stay with the feeling before trying to solve it.",
        "Lower the pressure and name what hurts most clearly.",
        "Give the emotion structure instead of trying to outrun it.",
    ],
    "joy": [
        "Turn the good energy into one deliberate next step.",
        "Capture the lift before the day moves on.",
        "Let the momentum become something repeatable.",
    ],
    "love": [
        "Translate connection into one concrete action.",
        "Notice what made you feel close and protect more of it.",
        "Let the warmth become a clearer choice, not just a passing feeling.",
    ],
    "anger": [
        "Slow the reaction down until the real boundary comes into focus.",
        "Separate the trigger from the next move.",
        "Use the heat to clarify what needs to change.",
    ],
    "fear": [
        "Shrink the uncertainty into one controllable next step.",
        "Ground the feeling before forecasting the future.",
        "Trade spiraling for specifics.",
    ],
    "surprise": [
        "Hold the disruption still long enough to understand it.",
        "Turn the unexpected into signal instead of noise.",
        "Notice what changed before deciding what it means.",
    ],
}

SUMMARY_TEMPLATES = {
    "sadness": {
        "high": [
            "This reads like a heavy, inward moment where the loss or weight feels close to the surface{context}.",
            "There is a clear tone of depletion here, as if the entry is carrying more weight than momentum{context}.",
        ],
        "medium": [
            "There is a pull toward sadness here, though some of the entry still feels mixed or unsettled{context}.",
            "The writing leans low and tired, even if the emotional center is not completely singular{context}.",
        ],
        "low": [
            "The entry suggests sadness, but it also carries overlap with other difficult emotions{context}.",
            "This looks emotionally low, though the signal is more blended than cleanly singular{context}.",
        ],
    },
    "joy": {
        "high": [
            "This reads like genuine lift, where relief or gratitude is turning into momentum{context}.",
            "There is clear upward energy here, as if the entry is anchored in appreciation or confidence{context}.",
        ],
        "medium": [
            "The entry leans positive and energized, even if part of the feeling still has some ambiguity{context}.",
            "There is a noticeable lift in the language, though it is not completely free of tension{context}.",
        ],
        "low": [
            "The writing points toward joy, but the emotional signal is more mixed than settled{context}.",
            "There is positive energy here, though it may be sharing space with another emotion{context}.",
        ],
    },
    "love": {
        "high": [
            "This feels centered on closeness, care, or emotional attachment rather than simple positivity{context}.",
            "The entry reads like connection is the emotional core, not just a passing pleasant mood{context}.",
        ],
        "medium": [
            "There is a relational warmth here, even if the emotional signal is not completely pure{context}.",
            "The writing leans toward affection or closeness, though it still carries some ambiguity{context}.",
        ],
        "low": [
            "The model sees connection in the entry, but the feeling may also overlap with joy or surprise{context}.",
            "This may be about care or attachment, though the signal is softer than the strongest cases{context}.",
        ],
    },
    "anger": {
        "high": [
            "This reads like a sharp activation point, where something feels crossed, blocked, or unfair{context}.",
            "There is direct heat in the language, as if the entry is reacting to pressure, violation, or frustration{context}.",
        ],
        "medium": [
            "The entry leans angry, though some of the language still overlaps with fear or stress{context}.",
            "There is friction in the writing, even if the emotion is not fully singular{context}.",
        ],
        "low": [
            "The writing suggests anger, but the signal may be shared with fear or sadness{context}.",
            "This looks like activation and friction, though not with complete clarity{context}.",
        ],
    },
    "fear": {
        "high": [
            "This reads like apprehension, vigilance, or unease is driving the entry{context}.",
            "There is strong uncertainty in the language, as if the writing is bracing for impact{context}.",
        ],
        "medium": [
            "The entry leans fearful, though some of the signal overlaps with sadness or anger{context}.",
            "There is a strong undertone of worry here, even if the feeling is somewhat mixed{context}.",
        ],
        "low": [
            "The writing suggests fear, but the emotional center may still be shared with another difficult state{context}.",
            "This looks uneasy and guarded, though the signal is not fully locked in{context}.",
        ],
    },
    "surprise": {
        "high": [
            "This reads like the entry is orienting around an unexpected turn rather than a stable mood{context}.",
            "There is a clear sense of interruption or sudden shift in the emotional signal{context}.",
        ],
        "medium": [
            "The writing leans toward surprise, though the reaction to the event is still mixed with another feeling{context}.",
            "Something unexpected seems central here, even if the emotional landing is not fully singular{context}.",
        ],
        "low": [
            "The model detects surprise, but the entry may really be surprise layered with another stronger emotion{context}.",
            "This looks reactive to change, though not with the cleanest possible signal{context}.",
        ],
    },
}

INTERPRETATION_TEMPLATES = {
    "sadness": {
        "high": "The model is picking up language that sounds depleted, resigned, or emotionally weighed down.",
        "medium": "The model is seeing low-energy language, though it also detects some emotional blending.",
        "low": "The model sees low affect here, but not with complete confidence.",
    },
    "joy": {
        "high": "The model is reacting to language that sounds uplifted, appreciative, or energized.",
        "medium": "The model sees a positive upward signal, though some ambiguity remains.",
        "low": "The model detects positive energy, but not with full certainty.",
    },
    "love": {
        "high": "The model is picking up words that suggest closeness, care, and relational warmth.",
        "medium": "The model sees connection-focused language, though it is not the only signal present.",
        "low": "The model detects attachment or warmth, but the signal is fairly soft.",
    },
    "anger": {
        "high": "The model is reacting to language that sounds blocked, provoked, or boundary-focused.",
        "medium": "The model sees activated, friction-heavy language, though the emotion is somewhat mixed.",
        "low": "The model detects agitation, but not with strong certainty.",
    },
    "fear": {
        "high": "The model is picking up caution, threat-sensitivity, and uncertainty in the language.",
        "medium": "The model sees worry-driven language, though some overlap with other difficult emotions remains.",
        "low": "The model detects unease, but the signal is not especially strong.",
    },
    "surprise": {
        "high": "The model is reacting to language that suggests sudden change, interruption, or shock.",
        "medium": "The model sees an unexpected-event signal, though the emotional reaction is still mixed.",
        "low": "The model detects surprise, but the entry may be leaning more strongly toward another emotion.",
    },
}

FOLLOW_UP_PROMPTS = {
    "sadness": [
        [
            "What feels heaviest in this situation, and what part of it is still changeable?",
            "If this feeling had a shape or texture, what would it be?",
            "What is the smallest act of care that would make tonight feel less sharp?",
        ],
        [
            "What loss, disappointment, or exhaustion sits underneath this entry?",
            "What are you needing that this moment is not giving you?",
            "What would be a gentler standard to hold yourself to for the next few hours?",
        ],
    ],
    "joy": [
        [
            "What exactly created this lift, and how much of it was under your control?",
            "What would it look like to carry this energy into one concrete action today?",
            "Who would understand why this moment matters to you?",
        ],
        [
            "What about this moment feels earned rather than accidental?",
            "Which part of this good feeling do you want to remember in detail?",
            "How could you recreate even ten percent of this on a harder day?",
        ],
    ],
    "love": [
        [
            "Who or what are you feeling especially connected to right now?",
            "What made this feel like care instead of simple comfort?",
            "How do you want to respond to this feeling while it is still present?",
        ],
        [
            "What kind of closeness is this entry pointing toward?",
            "What helped you feel safe, seen, or valued here?",
            "How could you protect more of this kind of connection in your life?",
        ],
    ],
    "anger": [
        [
            "What boundary, value, or expectation feels crossed here?",
            "What part of your reaction is about the present moment, and what part is older than this event?",
            "What response would feel strong without being reckless?",
        ],
        [
            "What exactly are you resisting, and why does it matter this much?",
            "Where does the anger want movement, distance, or clarity?",
            "What would a deliberate next step look like once the heat comes down a notch?",
        ],
    ],
    "fear": [
        [
            "What are you predicting right now, and which part is fact versus projection?",
            "What is the next concrete thing you can do to make the situation safer or clearer?",
            "Who or what would make this feel less isolating?",
        ],
        [
            "What uncertainty is your mind trying hardest to solve?",
            "If you narrowed this to the next hour, what would matter most?",
            "What support would reduce the fear instead of just distracting from it?",
        ],
    ],
    "surprise": [
        [
            "What changed faster than you expected, and how did that land in you?",
            "Is this surprise energizing, disorienting, or both?",
            "What do you need to understand before deciding what this means?",
        ],
        [
            "What about this caught you off guard?",
            "What story are you tempted to tell yourself too quickly about it?",
            "What would help you respond with curiosity instead of urgency?",
        ],
    ],
}


def confidence_band_for_score(confidence: float) -> str:
    if confidence >= 0.75:
        return "high"
    if confidence >= 0.55:
        return "medium"
    return "low"


def _pick(options: Iterable, text: str, salt: str):
    options = list(options)
    seed = hashlib.sha256(f"{normalize_text(text)}::{salt}".encode("utf-8")).hexdigest()
    index = int(seed, 16) % len(options)
    return options[index]


def _context_clause(location: Optional[str], activity: Optional[str]) -> str:
    fragments = []
    if activity:
        fragments.append(f"while you were {activity}")
    if location:
        fragments.append(f"in {location}")
    if not fragments:
        return ""
    if len(fragments) == 1:
        return f" {fragments[0]}"
    return f" {fragments[0]} {fragments[1]}"


def _format_phrase_list(explanation_phrases: List[str]) -> str:
    if not explanation_phrases:
        return ""
    if len(explanation_phrases) == 1:
        return explanation_phrases[0]
    if len(explanation_phrases) == 2:
        return f"{explanation_phrases[0]} and {explanation_phrases[1]}"
    return ", ".join(explanation_phrases[:-1]) + f", and {explanation_phrases[-1]}"


def choose_recommendation(emotion: str, text: str) -> str:
    return _pick(
        RECOMMENDATION_HEADLINES.get(
            emotion,
            ["Move one layer deeper instead of stopping at the first explanation."],
        ),
        text,
        f"headline:{emotion}",
    )


def build_support_response(
    emotion: str,
    text: str,
    confidence: float = 0.5,
    *,
    location: Optional[str] = None,
    activity: Optional[str] = None,
    explanation_phrases: Optional[List[str]] = None,
) -> dict:
    band = confidence_band_for_score(confidence)
    phrases = explanation_phrases or []
    if contains_crisis_language(text):
        return {
            "recommendation": CRISIS_RECOMMENDATION,
            "reflection_summary": CRISIS_REFLECTION_SUMMARY,
            "interpretation": CRISIS_INTERPRETATION,
            "follow_up_prompts": [],
            "disclaimer": CRISIS_DISCLAIMER,
            "is_crisis": True,
            "confidence_band": band,
            "model_name": None,
            "explanation_phrases": [],
            "support_message": CRISIS_SUPPORT_MESSAGE,
        }

    context = _context_clause(location, activity)
    summary = _pick(
        SUMMARY_TEMPLATES.get(emotion, SUMMARY_TEMPLATES["surprise"])[band],
        text,
        f"summary:{emotion}:{band}",
    ).format(context=context)
    if phrases:
        interpretation = (
            f"The model is mostly reacting to phrases like {_format_phrase_list(phrases[:3])}, "
            f"which often cluster around {emotion}-leaning language."
        )
    else:
        interpretation = INTERPRETATION_TEMPLATES.get(emotion, INTERPRETATION_TEMPLATES["surprise"])[band]

    prompts = list(
        _pick(
            FOLLOW_UP_PROMPTS.get(emotion, FOLLOW_UP_PROMPTS["surprise"]),
            text,
            f"prompts:{emotion}",
        )
    )
    if band == "low":
        prompts[-1] = "What other emotion might also be present here if the first read is incomplete?"

    return {
        "recommendation": choose_recommendation(emotion, text),
        "reflection_summary": summary,
        "interpretation": interpretation,
        "follow_up_prompts": prompts,
        "disclaimer": DEFAULT_DISCLAIMER,
        "is_crisis": False,
        "confidence_band": band,
        "model_name": None,
        "explanation_phrases": phrases,
        "support_message": None,
    }
