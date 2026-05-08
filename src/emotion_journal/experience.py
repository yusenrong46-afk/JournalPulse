from pathlib import Path
from typing import Optional

from .coach import build_initial_coach_turn
from .config import DEFAULT_DB_PATH
from .resources import recommend_resources


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
    resources = recommend_resources(
        prediction.emotion,
        is_crisis=prediction.is_crisis,
        db_path=db_path,
    )
    coach = build_initial_coach_turn(
        entry_text=text,
        emotion=prediction.emotion,
        confidence_band=prediction.confidence_band,
        is_crisis=prediction.is_crisis,
        use_llm=use_llm,
    )
    return {
        "prediction": prediction,
        "resources": resources,
        "coach": coach,
    }
