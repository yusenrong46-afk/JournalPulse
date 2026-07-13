from pathlib import Path
from types import SimpleNamespace

from emotion_journal.classification import analyze_emotion_scores, contains_negated_wellbeing
from emotion_journal.evaluation import evaluate_prediction_cases, load_eval_cases


EVAL_PATH = Path(__file__).resolve().parents[1] / "data" / "evals" / "journalpulse_eval.jsonl"


def test_journalpulse_eval_cases_are_valid():
    cases = load_eval_cases(EVAL_PATH)

    assert len(cases) >= 30
    assert len({case.id for case in cases}) == len(cases)
    assert any(case.should_trigger_crisis for case in cases)
    assert any("mixed" in case.tags for case in cases)
    assert all(case.primary_emotion in case.accepted_emotions for case in cases)


def test_evaluate_prediction_cases_summarizes_quality():
    class StubPredictor:
        def predict(self, text):
            if "unsafe" in text:
                emotion = "sadness"
                crisis = True
            elif "angry" in text:
                emotion = "anger"
                crisis = False
            else:
                emotion = "joy"
                crisis = False
            scores = {label: 0.02 for label in ["sadness", "joy", "love", "anger", "fear", "surprise"]}
            scores[emotion] = 0.9
            return SimpleNamespace(
                emotion=emotion,
                confidence=0.9,
                confidence_band="high",
                scores=scores,
                is_crisis=crisis,
                secondary_emotions=[],
                top_margin=0.8,
                is_mixed=False,
                uncertainty_reason=None,
                emotion_tags=[],
                calibration_notes=[],
            )

    cases = load_eval_cases(EVAL_PATH)[:3]
    report = evaluate_prediction_cases(StubPredictor(), cases)

    assert report["total_cases"] == 3
    assert 0 <= report["accepted_accuracy"] <= 1
    assert "by_primary_emotion" in report
    assert report["rows"][0]["top3"]


def test_journal_calibration_handles_common_misses():
    base_scores = {
        "sadness": 0.07,
        "joy": 0.54,
        "love": 0.01,
        "anger": 0.05,
        "fear": 0.29,
        "surprise": 0.04,
    }
    money_fear = analyze_emotion_scores(
        "The bill is bigger than I expected and I cannot stop calculating worst-case scenarios.",
        base_scores,
    )
    assert money_fear.emotion == "fear"
    assert "anxiety" in money_fear.emotion_tags
    assert "catastrophizing_cues" in money_fear.calibration_notes

    surprise = analyze_emotion_scores(
        "The news came out of nowhere. I am not sure if it is good or bad yet.",
        {**base_scores, "joy": 0.8, "surprise": 0.04},
    )
    assert surprise.emotion == "surprise"
    assert "ambiguous_disruption_cues" in surprise.calibration_notes

    crisis = analyze_emotion_scores(
        "I want to die and I do not feel safe being alone right now.",
        {**base_scores, "joy": 0.99},
    )
    assert crisis.emotion == "sadness"
    assert "crisis_language_safety_override" in crisis.calibration_notes


def test_journal_calibration_handles_negated_wellbeing():
    base_scores = {
        "sadness": 0.004,
        "joy": 0.982,
        "love": 0.007,
        "anger": 0.001,
        "fear": 0.004,
        "surprise": 0.002,
    }

    for text in [
        "I am not feeling that well.",
        "I am not feeling well.",
        "I am not doing well today.",
        "I do not feel okay.",
    ]:
        analysis = analyze_emotion_scores(text, base_scores)
        assert contains_negated_wellbeing(text)
        assert analysis.emotion in {"sadness", "fear"}
        assert analysis.emotion != "joy"
        assert analysis.confidence_band == "medium"
        assert "negated_wellbeing_override" in analysis.calibration_notes

    off_day = analyze_emotion_scores("I feel off today.", base_scores)
    assert off_day.emotion in {"sadness", "fear"}
    assert off_day.emotion != "joy"
    assert "negated_wellbeing_override" in off_day.calibration_notes
