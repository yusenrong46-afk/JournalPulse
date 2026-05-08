from pathlib import Path

from emotion_journal.experience import build_prediction_experience
from emotion_journal.model import ArtifactPredictor


def test_saved_artifact_predictor_smoke():
    model_dir = Path(__file__).resolve().parents[1] / "artifacts" / "models"
    predictor = ArtifactPredictor.from_artifacts(model_dir)

    joyful = predictor.predict("I feel excited, grateful, and energized after today.")
    fearful = predictor.predict("I am nervous and scared about what could happen tomorrow.")
    crisis = predictor.predict("I want to die and I do not feel safe.")

    assert joyful.emotion in {"joy", "love", "surprise"}
    assert fearful.emotion in {"fear", "sadness", "anger"}
    assert 0.0 <= joyful.confidence <= 1.0
    assert 0.0 <= fearful.confidence <= 1.0
    assert joyful.model_name
    assert joyful.confidence_band in {"high", "medium", "low"}
    assert len(joyful.follow_up_prompts) in {0, 3}
    assert joyful.support_message is None
    assert crisis.is_crisis is True
    assert crisis.follow_up_prompts == []
    assert crisis.explanation_phrases == []
    assert crisis.support_message

    experience = build_prediction_experience(
        predictor,
        "I feel excited, grateful, and energized after today.",
        db_path=Path(__file__).resolve().parents[1] / "artifacts" / "journal_smoke.db",
    )
    assert experience["resources"]
    assert experience["coach"]["assistant_message"]
