"""Turn the person's button taps into the numeric state stored on a reflection.

The explicit report (which feelings, which mood face) is kept separately on the record;
this module only produces the derived representation. The values mirror
web/lib/feelings.ts, and tests/test_self_report.py fails if the two drift apart.
"""

from __future__ import annotations

from .domain import AffectiveState, SelfReportInput

DERIVATION = "feeling-buttons-v1"

# (valence, arousal, agency) per feeling.
FEELING_COORDINATES: dict[str, tuple[float, float, float]] = {
    "tired": (-0.3, 0.2, 0.4),
    "anxious": (-0.5, 0.8, 0.35),
    "stressed": (-0.45, 0.75, 0.4),
    "sad": (-0.6, 0.3, 0.35),
    "frustrated": (-0.5, 0.75, 0.45),
    "lonely": (-0.5, 0.3, 0.35),
    "overwhelmed": (-0.65, 0.85, 0.2),
    "numb": (-0.3, 0.15, 0.3),
    "calm": (0.45, 0.25, 0.65),
    "hopeful": (0.5, 0.5, 0.7),
    "okay": (0.1, 0.4, 0.55),
    "happy": (0.7, 0.6, 0.7),
}

MOOD_VALENCE: dict[int, float] = {5: 0.75, 4: 0.4, 3: 0.0, 2: -0.4, 1: -0.75}


def _clamp(value: float, low: float, high: float) -> float:
    return min(high, max(low, value))


def derive_state(report: SelfReportInput) -> AffectiveState:
    mood = MOOD_VALENCE.get(report.mood_score) if report.mood_score is not None else None
    chosen = [FEELING_COORDINATES[item] for item in report.feelings if item in FEELING_COORDINATES]
    if not chosen:
        return AffectiveState(
            valence=round(mood or 0.0, 2),
            arousal=0.5,
            agency=0.5,
            emotion_tags=[],
            confidence=None,
            uncertainty="Only an overall mood was reported."
            if mood is not None
            else "No feelings were reported.",
            derivation=DERIVATION,
        )
    valence = sum(item[0] for item in chosen) / len(chosen)
    if mood is not None:
        # The mood face is the most direct answer to "how are you?", so it weighs double.
        valence = (valence + mood * 2) / 3
    return AffectiveState(
        valence=round(_clamp(valence, -1, 1), 2),
        arousal=round(_clamp(sum(item[1] for item in chosen) / len(chosen), 0, 1), 2),
        agency=round(_clamp(sum(item[2] for item in chosen) / len(chosen), 0, 1), 2),
        emotion_tags=list(report.feelings),
        confidence=None,
        uncertainty="Derived from the feeling buttons the person chose.",
        derivation=DERIVATION,
    )
