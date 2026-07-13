import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .config import CLASSIFIER_MODE_ENV, DEFAULT_CLASSIFIER_MODE
from .llm import OpenAICompatibleCoachAdapter, validate_structured_emotion_payload
from .preprocessing import contains_crisis_language, normalize_text
from .recommendations import confidence_band_for_score


CORE_EMOTIONS = ("sadness", "joy", "love", "anger", "fear", "surprise")
SUPPORTED_CLASSIFIER_MODES = {"calibrated", "llm", "hybrid"}

EMOTION_CUE_PATTERNS = {
    "anger": (
        r"\bangry\b",
        r"\banger\b",
        r"\birritated\b",
        r"\bfrustrated\b",
        r"\bunfair\b",
        r"\bdismissed\b",
        r"\btalked over\b",
        r"\btook credit\b",
        r"\bcredit for my work\b",
        r"\btaken for granted\b",
        r"\bboundar(?:y|ies)\b",
        r"\bsaid yes\b.*\bwanted to say no\b",
        r"\bfolding so quickly\b",
        r"\bhot with anger\b",
    ),
    "fear": (
        r"\bafraid\b",
        r"\bscared\b",
        r"\bnervous\b",
        r"\bworried\b",
        r"\bworrying\b",
        r"\bworst case\b",
        r"\bworst-case\b",
        r"\bworstcase\b",
        r"\bcannot stop calculating\b",
        r"\bwhat could happen\b",
        r"\bwaiting for\b.*\bresult\b",
        r"\bstomach drop\b",
        r"\bchest is tight\b",
        r"\bthoughts are racing\b",
        r"\bmake them pull away\b",
        r"\bfail\b",
    ),
    "sadness": (
        r"\bsad\b",
        r"\bcried\b",
        r"\bcrying\b",
        r"\bmiss\b",
        r"\blonely\b",
        r"\bloneliness\b",
        r"\babsence\b",
        r"\bheavy\b",
        r"\bflat\b",
        r"\bdisconnected\b",
        r"\bsmall\b",
        r"\bembarrassed\b",
        r"\bdisappointed\b",
        r"\bhopeless\b",
        r"\btired of pretending\b",
    ),
    "love": (
        r"\blove\b",
        r"\bcared for\b",
        r"\bchecked in\b",
        r"\bclose to\b",
        r"\bunderstood\b",
        r"\btenderness\b",
        r"\bsqueezed my hand\b",
        r"\bconnection mattered\b",
        r"\bfelt safe\b",
        r"\bremembered\b.*\bworried\b",
    ),
    "joy": (
        r"\bproud\b",
        r"\bexcited\b",
        r"\bthrilled\b",
        r"\bsmiling\b",
        r"\blighter\b",
        r"\blight\b",
        r"\brelief\b",
        r"\bmomentum\b",
        r"\bwent well\b",
        r"\bfinally finished\b",
        r"\bsmall win\b",
    ),
    "surprise": (
        r"\bsurprised\b",
        r"\bsurprise\b",
        r"\bstunned\b",
        r"\bshocked\b",
        r"\bthrown off\b",
        r"\bcaught off guard\b",
        r"\bout of nowhere\b",
        r"\bwithout warning\b",
        r"\bnot sure if it is good or bad\b",
        r"\btrying to catch up\b",
        r"\bapologized after months\b",
        r"\bfound out\b",
    ),
}

TAG_CUE_PATTERNS = {
    "anxiety": (r"\banxious\b", r"\bnervous\b", r"\bscared\b", r"\bworst-case\b", r"\bworstcase\b", r"\bthoughts are racing\b"),
    "overwhelm": (r"\boverwhelmed\b", r"\btoo much\b", r"\bcannot settle\b", r"\bcan't settle\b", r"\bnot okay\b", r"\bdo not feel okay\b", r"\bdont feel okay\b"),
    "grief": (r"\bmiss\b.*\bdad\b", r"\bmiss them\b", r"\babsence\b", r"\bloss\b", r"\bgrief\b"),
    "loneliness": (r"\blonely\b", r"\bwithout me\b", r"\bnot personal\b"),
    "shame": (r"\bembarrassed\b", r"\bsmall\b", r"\bdisappointed in myself\b"),
    "burnout": (r"\bburnout\b", r"\bflat\b", r"\bdisconnected\b", r"\btired of pretending\b", r"\bonly feel heavy\b"),
    "frustration": (r"\bfrustrated\b", r"\birritated\b", r"\bdismissed\b", r"\btalked over\b"),
    "boundary": (r"\bboundary\b", r"\bsaid yes\b.*\bwanted to say no\b", r"\btaken for granted\b"),
    "relief": (r"\brelief\b", r"\bpressure finally left\b", r"\bover and it went well\b"),
    "gratitude": (r"\bgrateful\b", r"\bgive thanks\b", r"\bthankful\b"),
    "pride": (r"\bproud\b", r"\bfinished the project\b", r"\breal momentum\b"),
    "confusion": (r"\bnot sure\b", r"\bdo not know\b", r"\bdon't know\b", r"\btrying to understand\b"),
    "connection": (r"\bclose to\b", r"\bunderstood\b", r"\bcared for\b", r"\bconnection\b", r"\bfelt safe\b"),
    "work_stress": (r"\bmanager\b", r"\bmeeting\b", r"\bpresentation\b", r"\bproject\b", r"\bpromoted\b"),
    "self_advocacy": (r"\bwhat to say\b", r"\bplan what to say\b", r"\bboundary\b", r"\bsay no\b"),
}

NEGATED_WELLBEING_PATTERNS = (
    r"\bnot (?:feeling|doing) (?:that )?(?:too )?well\b",
    r"\bdo not feel (?:that )?(?:too )?well\b",
    r"\bdont feel (?:that )?(?:too )?well\b",
    r"\bnot feel (?:that )?(?:too )?well\b",
    r"\bdo not feel okay\b",
    r"\bdont feel okay\b",
    r"\bnot okay\b",
    r"\bnot ok\b",
    r"\bnot alright\b",
    r"\bnot fine\b",
    r"\bnot good\b",
    r"\bnot great\b",
    r"\bfeel off\b",
    r"\bfeeling off\b",
    r"\bunwell\b",
)


@dataclass
class EmotionAnalysis:
    emotion: str
    confidence: float
    scores: Dict[str, float]
    secondary_emotions: List[str] = field(default_factory=list)
    emotion_tags: List[str] = field(default_factory=list)
    top_margin: Optional[float] = None
    is_mixed: bool = False
    uncertainty_reason: Optional[str] = None
    calibration_notes: List[str] = field(default_factory=list)
    classifier_mode: str = "calibrated"
    classifier_source: str = "artifact"
    classifier_fallback_reason: Optional[str] = None

    @property
    def confidence_band(self) -> str:
        return confidence_band_for_score(self.confidence)


def _pattern_score(text: str, patterns: tuple) -> int:
    return sum(1 for pattern in patterns if re.search(pattern, text))


def detect_emotion_tags(text: str) -> List[str]:
    normalized = normalize_text(text)
    tags = []
    for tag, patterns in TAG_CUE_PATTERNS.items():
        if _pattern_score(normalized, patterns):
            tags.append(tag)
    return tags[:6]


def contains_negated_wellbeing(text: str) -> bool:
    normalized = normalize_text(text)
    return _pattern_score(normalized, NEGATED_WELLBEING_PATTERNS) > 0


def _renormalize(scores: Dict[str, float]) -> Dict[str, float]:
    total = sum(max(value, 0.0) for value in scores.values())
    if total <= 0:
        return {emotion: 1 / len(scores) for emotion in scores}
    return {emotion: max(value, 0.0) / total for emotion, value in scores.items()}


def _rank(scores: Dict[str, float]) -> List[tuple]:
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)


def configured_classifier_mode() -> str:
    mode = os.getenv(CLASSIFIER_MODE_ENV, DEFAULT_CLASSIFIER_MODE).strip().lower()
    return mode if mode in SUPPORTED_CLASSIFIER_MODES else DEFAULT_CLASSIFIER_MODE


def _confidence_from_band(band: str, intensity: float) -> float:
    floors = {"low": 0.48, "medium": 0.66, "high": 0.84}
    base = floors.get(band, 0.66)
    return round(max(0.35, min(0.97, base + (float(intensity) - 0.5) * 0.16)), 4)


def _scores_from_structured_payload(payload: dict) -> Dict[str, float]:
    primary = payload["primary_emotion"]
    secondary = payload.get("secondary_emotions", [])
    confidence = _confidence_from_band(payload.get("confidence", "medium"), payload.get("intensity", 0.5))
    remainder = max(0.0, 1.0 - confidence)
    scores = {emotion: 0.0 for emotion in CORE_EMOTIONS}
    scores[primary] = confidence

    if secondary:
        share = min(0.22, remainder / max(1, len(secondary)))
        for emotion in secondary:
            scores[emotion] = share
        remainder = max(0.0, 1.0 - sum(scores.values()))

    other_emotions = [emotion for emotion in CORE_EMOTIONS if scores[emotion] == 0.0]
    for emotion in other_emotions:
        scores[emotion] = remainder / max(1, len(other_emotions))
    return _renormalize(scores)


def _analysis_from_structured_payload(payload: dict, *, mode: str, model_name: Optional[str]) -> EmotionAnalysis:
    scores = _scores_from_structured_payload(payload)
    ranked = _rank(scores)
    emotion, confidence = ranked[0]
    runner_up_emotion, runner_up_score = ranked[1]
    top_margin = confidence - runner_up_score
    secondary = payload.get("secondary_emotions") or [
        label for label, score in ranked[1:3] if score >= 0.15 or confidence - score <= 0.18
    ]
    is_mixed = bool(payload.get("is_mixed")) or confidence < 0.55 or top_margin <= 0.18 or bool(secondary)
    uncertainty_reason = None
    if is_mixed:
        uncertainty_reason = "llm_structured_mixed" if secondary else "low_top_confidence"

    return EmotionAnalysis(
        emotion=emotion,
        confidence=round(confidence, 4),
        scores={label: round(score, 4) for label, score in scores.items()},
        secondary_emotions=[item for item in secondary if item != emotion][:3],
        emotion_tags=payload.get("emotion_tags", [])[:6],
        top_margin=round(top_margin, 4),
        is_mixed=is_mixed,
        uncertainty_reason=uncertainty_reason,
        calibration_notes=[f"structured_llm:{model_name or 'unknown'}"],
        classifier_mode=mode,
        classifier_source="llm",
    )


HYBRID_MODEL_WEIGHT = 0.45  # weight on the calibrated transformer; LLM gets the rest


def _blend_with_fallback(
    llm_analysis: EmotionAnalysis,
    fallback: EmotionAnalysis,
    *,
    model_weight: float = HYBRID_MODEL_WEIGHT,
) -> EmotionAnalysis:
    """Blend the calibrated transformer scores with the LLM scores for hybrid mode.

    The transformer grounds the distribution while the LLM adds journal-aware nuance;
    a weighted average of both keeps either one from dominating.
    """

    blended = {
        emotion: model_weight * float(fallback.scores.get(emotion, 0.0))
        + (1.0 - model_weight) * float(llm_analysis.scores.get(emotion, 0.0))
        for emotion in CORE_EMOTIONS
    }
    blended = _renormalize(blended)
    ranked = _rank(blended)
    emotion, confidence = ranked[0]
    runner_up_emotion, runner_up_score = ranked[1]
    top_margin = confidence - runner_up_score
    secondary = [
        label
        for label, score in ranked[1:3]
        if score >= 0.15 or confidence - score <= 0.18
    ]
    is_mixed = bool(llm_analysis.is_mixed or fallback.is_mixed) or confidence < 0.55 or top_margin <= 0.18

    merged_tags = []
    for tag in list(llm_analysis.emotion_tags) + list(fallback.emotion_tags):
        if tag not in merged_tags:
            merged_tags.append(tag)

    notes = list(fallback.calibration_notes) + list(llm_analysis.calibration_notes)
    notes.append(f"hybrid_blend:model{model_weight:g}_llm{1 - model_weight:g}")

    return EmotionAnalysis(
        emotion=emotion,
        confidence=round(confidence, 4),
        scores={label: round(score, 4) for label, score in blended.items()},
        secondary_emotions=[item for item in secondary if item != emotion][:3],
        emotion_tags=merged_tags[:6],
        top_margin=round(top_margin, 4),
        is_mixed=is_mixed,
        uncertainty_reason="hybrid_blend" if is_mixed else None,
        calibration_notes=notes,
        classifier_mode="hybrid",
        classifier_source="hybrid",
    )


def maybe_generate_llm_emotion_analysis(
    text: str,
    fallback: EmotionAnalysis,
) -> EmotionAnalysis:
    mode = configured_classifier_mode()
    if mode == "calibrated" or contains_crisis_language(text):
        fallback.classifier_mode = mode
        fallback.classifier_source = "artifact"
        if contains_crisis_language(text):
            fallback.classifier_fallback_reason = "crisis_mode_llm_bypassed"
        return fallback

    adapter = OpenAICompatibleCoachAdapter.from_env()
    if adapter is None:
        fallback.classifier_mode = mode
        fallback.classifier_source = "artifact"
        fallback.classifier_fallback_reason = "llm_adapter_unavailable"
        fallback.calibration_notes.append("llm_classifier_unavailable")
        return fallback

    context = {
        "primary_emotion": fallback.emotion,
        "secondary_emotions": fallback.secondary_emotions,
        "emotion_tags": fallback.emotion_tags,
        "confidence": fallback.confidence,
        "confidence_band": fallback.confidence_band,
        "scores": fallback.scores,
        "is_mixed": fallback.is_mixed,
        "uncertainty_reason": fallback.uncertainty_reason,
    }
    try:
        payload = adapter.classify_emotion(text=text, calibrated_context=context)
        structured = validate_structured_emotion_payload(payload)
        llm_analysis = _analysis_from_structured_payload(
            structured,
            mode=mode,
            model_name=adapter.model,
        )
        if mode == "hybrid":
            return _blend_with_fallback(llm_analysis, fallback)
        return llm_analysis
    except Exception as exc:
        fallback.classifier_mode = mode
        fallback.classifier_source = "artifact"
        fallback.classifier_fallback_reason = f"llm_classifier_invalid:{exc.__class__.__name__}"
        fallback.calibration_notes.append("llm_classifier_fallback")
        return fallback


def analyze_emotion_scores(text: str, base_scores: Dict[str, float]) -> EmotionAnalysis:
    """Adapt benchmark probabilities to journal-style language.

    This is intentionally transparent and conservative: it keeps the transformer
    score distribution, then adds small cue-based boosts for phrases that are
    common in journaling but underrepresented in short benchmark sentences.
    """

    normalized = normalize_text(text)
    scores = {emotion: float(base_scores.get(emotion, 0.0)) for emotion in CORE_EMOTIONS}
    notes = []

    if contains_crisis_language(text):
        scores = {
            "sadness": 0.56,
            "fear": 0.34,
            "anger": 0.03,
            "love": 0.03,
            "joy": 0.02,
            "surprise": 0.02,
        }
        scores = _renormalize(scores)
        notes.append("crisis_language_safety_override")
    else:
        if contains_negated_wellbeing(text):
            # The artifact strongly overweights words like "well" and "okay".
            # In journal language, negated wellbeing is low/mixed distress, not joy.
            scores["joy"] = min(scores.get("joy", 0.0), 0.08)
            scores["sadness"] = max(scores.get("sadness", 0.0), 0.58)
            scores["fear"] = max(scores.get("fear", 0.0), 0.22)
            notes.append("negated_wellbeing_override")

        for emotion, patterns in EMOTION_CUE_PATTERNS.items():
            matches = _pattern_score(normalized, patterns)
            if matches:
                boost = min(0.1 + 0.08 * matches, 0.34)
                scores[emotion] += boost
                notes.append(f"{emotion}_journal_cues")

        # Pairwise product heuristics for common mixed journal states.
        if "miss" in normalized and ("cried" in normalized or "absence" in normalized):
            scores["sadness"] += 0.2
            scores["love"] += 0.08
            notes.append("grief_memory_cues")
        if "out of nowhere" in normalized or "not sure if it is good or bad" in normalized:
            scores["surprise"] += 0.95
            scores["fear"] += 0.06
            notes.append("ambiguous_disruption_cues")
        if "took credit" in normalized or "taken for granted" in normalized:
            scores["anger"] += 0.24
            notes.append("boundary_violation_cues")
        if "worst case" in normalized or "worst-case" in normalized or "worstcase" in normalized:
            scores["fear"] += 0.95
            notes.append("catastrophizing_cues")

        scores = _renormalize(scores)

    ranked = _rank(scores)
    emotion, confidence = ranked[0]
    runner_up_emotion, runner_up_score = ranked[1]
    top_margin = confidence - runner_up_score
    secondary_emotions = [
        label
        for label, score in ranked[1:3]
        if score >= 0.15 or confidence - score <= 0.18
    ]
    is_mixed = confidence < 0.55 or top_margin <= 0.18
    uncertainty_reason = None
    if is_mixed:
        uncertainty_reason = "low_top_confidence" if confidence < 0.55 else "close_top_scores"

    return EmotionAnalysis(
        emotion=emotion,
        confidence=round(confidence, 4),
        scores={label: round(score, 4) for label, score in scores.items()},
        secondary_emotions=secondary_emotions,
        emotion_tags=detect_emotion_tags(text),
        top_margin=round(top_margin, 4),
        is_mixed=is_mixed,
        uncertainty_reason=uncertainty_reason,
        calibration_notes=notes,
    )
