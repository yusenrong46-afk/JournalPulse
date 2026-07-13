from pathlib import Path
from typing import Optional

from .coach import build_initial_coach_turn
from .config import DEFAULT_DB_PATH
from .resources import build_recommendation_set


def build_prediction_experience(
    predictor,
    text: str,
    *,
    location: Optional[str] = None,
    activity: Optional[str] = None,
    db_path: Path = DEFAULT_DB_PATH,
    use_llm: bool = False,
) -> dict:
    """Build the complete response shown by the API and Streamlit app.

    Think of this as the product-level pipeline:
    model prediction -> resource ranking -> first coach message.
    """

    prediction = predictor.predict(
        text,
        location=location,
        activity=activity,
    )
    analysis_context = {
        "emotion": prediction.emotion,
        "secondary_emotions": prediction.secondary_emotions,
        "emotion_tags": prediction.emotion_tags,
        "confidence_band": prediction.confidence_band,
        "is_mixed": prediction.is_mixed,
    }
    resources, recommendation_meta = build_recommendation_set(
        text,
        emotion=prediction.emotion,
        is_crisis=prediction.is_crisis,
        analysis_context=analysis_context,
        db_path=db_path,
        use_llm=use_llm,
    )
    coach = build_initial_coach_turn(
        entry_text=text,
        emotion=prediction.emotion,
        confidence_band=prediction.confidence_band,
        is_crisis=prediction.is_crisis,
        use_llm=use_llm,
        allowed_resources=resources,
        nlp_context={
            "secondary_emotions": prediction.secondary_emotions,
            "emotion_tags": prediction.emotion_tags,
            "is_mixed": prediction.is_mixed,
            "uncertainty_reason": prediction.uncertainty_reason,
            "classifier_source": prediction.classifier_source,
            "classifier_mode": prediction.classifier_mode,
        },
    )
    return {
        "prediction": prediction,
        "resources": resources,
        "coach": coach,
        "recommendation_meta": recommendation_meta,
    }
