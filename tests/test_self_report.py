import re
from pathlib import Path

from journalpulse.domain import FEELINGS, SelfReportInput
from journalpulse.self_report import DERIVATION, FEELING_COORDINATES, MOOD_VALENCE, derive_state

WEB_FEELINGS = Path(__file__).resolve().parents[1] / "web" / "lib" / "feelings.ts"


def test_server_and_browser_use_the_same_button_values():
    source = WEB_FEELINGS.read_text()
    feelings = {
        match["id"]: (float(match["v"]), float(match["a"]), float(match["g"]))
        for match in re.finditer(
            r'id: "(?P<id>\w+)".*?valence: (?P<v>-?[\d.]+), arousal: (?P<a>[\d.]+), agency: (?P<g>[\d.]+)',
            source,
        )
    }
    moods = {
        int(match["score"]): float(match["v"])
        for match in re.finditer(r"score: (?P<score>\d).*?valence: (?P<v>-?[\d.]+)", source)
    }
    assert feelings == FEELING_COORDINATES
    assert moods == MOOD_VALENCE
    assert tuple(FEELING_COORDINATES) == FEELINGS


def test_derived_state_keeps_the_report_and_claims_no_confidence():
    state = derive_state(SelfReportInput(feelings=["tired", "anxious"], mood_score=None))
    assert state.emotion_tags == ["tired", "anxious"]
    assert state.valence == -0.4
    assert state.arousal == 0.5
    assert state.confidence is None
    assert state.derivation == DERIVATION


def test_the_mood_face_weighs_double_for_valence():
    state = derive_state(SelfReportInput(feelings=["calm"], mood_score=1))
    assert state.valence == round((0.45 + -0.75 * 2) / 3, 2)


def test_no_feelings_is_recorded_as_no_feelings():
    state = derive_state(SelfReportInput(feelings=[], mood_score=None))
    assert state.emotion_tags == []
    assert state.confidence is None
    assert state.uncertainty == "No feelings were reported."
    mood_only = derive_state(SelfReportInput(feelings=[], mood_score=4))
    assert mood_only.valence == 0.4
    assert mood_only.uncertainty == "Only an overall mood was reported."


def test_duplicate_taps_count_once():
    assert SelfReportInput(feelings=["sad", "sad", "tired"]).feelings == ["sad", "tired"]
