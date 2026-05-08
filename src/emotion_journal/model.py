import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np

from .config import (
    BASELINE_MODEL_NAME,
    EMOTION_TO_ID,
    EXPLANATION_PHRASE_LIMIT,
    LABELS,
    MODELS_DIR,
)
from .preprocessing import normalize_text
from .recommendations import build_support_response


@dataclass
class Prediction:
    """Everything the app needs after reading one journal entry."""

    emotion: str
    confidence: float
    recommendation: str
    disclaimer: str
    is_crisis: bool
    scores: Dict[str, float]
    support_message: Optional[str] = None
    model_name: Optional[str] = None
    confidence_band: Optional[str] = None
    reflection_summary: Optional[str] = None
    interpretation: Optional[str] = None
    follow_up_prompts: List[str] = field(default_factory=list)
    explanation_phrases: List[str] = field(default_factory=list)


class BaselineExplainer:
    """Use the saved linear baseline to explain the transformer's prediction.

    The transformer is the stronger production model, but its internals are hard
    to inspect. This small helper keeps the classical TF-IDF model around so the
    UI can say, in plain language, which words or short phrases mattered.
    """

    def __init__(self, pipeline) -> None:
        self.vectorizer = pipeline.named_steps["tfidf"]
        self.classifier = pipeline.named_steps["classifier"]
        self.feature_names = np.asarray(self.vectorizer.get_feature_names_out())

    def explain(
        self,
        text: str,
        predicted_emotion: str,
        *,
        top_k: int = EXPLANATION_PHRASE_LIMIT,
    ) -> List[str]:
        normalized = normalize_text(text)
        if not normalized:
            return []

        vector = self.vectorizer.transform([normalized])
        if vector.nnz == 0:
            return []

        class_value = EMOTION_TO_ID[predicted_emotion]
        class_index = int(np.where(self.classifier.classes_ == class_value)[0][0])
        coefficients = self.classifier.coef_[class_index]

        # In a linear model, each present phrase contributes:
        # TF-IDF value for the phrase * weight for this emotion class.
        contributions = vector.multiply(coefficients).tocoo()
        candidates = [
            (self.feature_names[feature_index], float(score))
            for feature_index, score in zip(contributions.col, contributions.data)
            if score > 0
        ]

        if not candidates:
            candidates = [
                (self.feature_names[feature_index], abs(float(score)))
                for feature_index, score in zip(contributions.col, contributions.data)
            ]

        phrases = []
        seen = set()
        for phrase, _score in sorted(candidates, key=lambda item: item[1], reverse=True):
            if phrase in seen:
                continue
            phrases.append(phrase)
            seen.add(phrase)
            if len(phrases) == top_k:
                break
        return phrases


class ArtifactPredictor:
    """Loads the saved model artifacts and exposes one easy `predict` method."""

    def __init__(
        self,
        *,
        model,
        model_type: str,
        metadata: dict,
        tokenizer: Optional[object] = None,
        baseline_pipeline=None,
    ) -> None:
        self.model = model
        self.model_type = model_type
        self.metadata = metadata
        self.tokenizer = tokenizer
        self.max_length = metadata.get("max_length", 128)
        label_map = metadata.get("label_map", LABELS)
        self.label_map = {int(key): value for key, value in label_map.items()}
        self.model_name = metadata.get("model_name", BASELINE_MODEL_NAME)
        self.baseline_pipeline = baseline_pipeline
        self.explainer = BaselineExplainer(baseline_pipeline) if baseline_pipeline is not None else None
        self.device = None
        if self.model_type == "hf_transformer":
            import torch

            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
            self.model.to(self.device)
            self.model.eval()

    @classmethod
    def from_artifacts(cls, model_dir: Path = MODELS_DIR) -> "ArtifactPredictor":
        """Build the predictor from files created by `scripts/train.py`."""

        production_metadata = json.loads((model_dir / "production.json").read_text())
        model_type = production_metadata["model_type"]

        baseline_pipeline = None
        baseline_artifact_path = production_metadata.get("baseline_artifact_path")
        if baseline_artifact_path:
            baseline_pipeline = joblib.load(model_dir / baseline_artifact_path)

        if model_type == "sklearn_pipeline":
            model = joblib.load(model_dir / production_metadata["artifact_path"])
            baseline_pipeline = baseline_pipeline or model
            return cls(
                model=model,
                model_type=model_type,
                metadata=production_metadata,
                baseline_pipeline=baseline_pipeline,
            )

        if model_type == "hf_transformer":
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            artifact_dir = model_dir / production_metadata["artifact_dir"]
            tokenizer = AutoTokenizer.from_pretrained(artifact_dir)
            model = AutoModelForSequenceClassification.from_pretrained(artifact_dir)
            return cls(
                model=model,
                model_type=model_type,
                metadata=production_metadata,
                tokenizer=tokenizer,
                baseline_pipeline=baseline_pipeline,
            )

        raise ValueError(f"Unsupported model type: {model_type}")

    def _predict_probabilities(self, text: str) -> np.ndarray:
        """Return one probability per emotion label."""

        if self.model_type == "sklearn_pipeline":
            if hasattr(self.model, "predict_proba"):
                return np.asarray(self.model.predict_proba([text])[0], dtype=float)
            if hasattr(self.model, "decision_function"):
                scores = np.asarray(self.model.decision_function([text])[0], dtype=float)
                scores = scores - np.max(scores)
                exps = np.exp(scores)
                return exps / np.sum(exps)
            raise ValueError("Unsupported sklearn pipeline: missing predict_proba and decision_function")

        if self.model_type == "hf_transformer":
            import torch

            # Tokenization turns the user's words into model-readable IDs.
            encoded = self.tokenizer(
                text,
                truncation=True,
                padding=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            with torch.no_grad():
                logits = self.model(**encoded).logits
                # Softmax converts raw model scores into probabilities.
                probabilities = torch.softmax(logits, dim=-1)[0].cpu().numpy()
            return np.asarray(probabilities, dtype=float)

        raise ValueError(f"Unsupported model type: {self.model_type}")

    def predict(
        self,
        text: str,
        *,
        location: Optional[str] = None,
        activity: Optional[str] = None,
    ) -> Prediction:
        raw_text = text.strip()
        if not raw_text:
            raise ValueError("Prediction requires non-empty text")

        probabilities = self._predict_probabilities(raw_text)
        winner = int(np.argmax(probabilities))
        emotion = self.label_map[winner]
        confidence = float(probabilities[winner])
        scores = {
            self.label_map[index]: round(float(score), 4)
            for index, score in enumerate(probabilities)
        }

        explanation_phrases: List[str] = []
        if self.explainer is not None:
            explanation_phrases = self.explainer.explain(raw_text, emotion)

        # The model gives us the emotion. The support layer turns that into
        # user-facing guidance, prompts, confidence bands, and safety handling.
        support = build_support_response(
            emotion,
            raw_text,
            confidence,
            location=location,
            activity=activity,
            explanation_phrases=explanation_phrases,
        )
        if support["is_crisis"]:
            explanation_phrases = []

        return Prediction(
            emotion=emotion,
            confidence=round(confidence, 4),
            recommendation=support["recommendation"],
            disclaimer=support["disclaimer"],
            is_crisis=support["is_crisis"],
            scores=scores,
            support_message=support["support_message"],
            model_name=self.model_name,
            confidence_band=support["confidence_band"],
            reflection_summary=support["reflection_summary"],
            interpretation=support["interpretation"],
            follow_up_prompts=support["follow_up_prompts"],
            explanation_phrases=explanation_phrases,
        )


@lru_cache(maxsize=1)
def get_default_predictor() -> ArtifactPredictor:
    return ArtifactPredictor.from_artifacts()
