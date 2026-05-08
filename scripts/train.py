#!/usr/bin/env python3
import argparse
import copy
import json
import os
import random
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from emotion_journal.config import (
    BASELINE_MODEL_NAME,
    DEFAULT_TRANSFORMER_CANDIDATES,
    LABELS,
    LINEAR_SVC_MODEL_NAME,
    MAX_TRANSFORMER_LENGTH,
    MODELS_DIR,
    RANDOM_SEED,
    REPORTS_DIR,
    SECOND_TRANSFORMER_MODEL_NAME,
    TRANSFORMER_MODEL_NAME,
)
from emotion_journal.preprocessing import normalize_text


def parse_transformer_candidates(value: str = None, *, skip_transformers: bool = False) -> list:
    if skip_transformers:
        return []
    raw_candidates = value.split(",") if value else list(DEFAULT_TRANSFORMER_CANDIDATES)
    candidates = []
    for candidate in raw_candidates:
        name = candidate.strip()
        if name and name not in candidates:
            candidates.append(name)
    return candidates


def build_training_plan(args: argparse.Namespace) -> dict:
    transformer_candidates = parse_transformer_candidates(
        args.transformers,
        skip_transformers=args.skip_transformers,
    )
    return {
        "classical_models": [BASELINE_MODEL_NAME, LINEAR_SVC_MODEL_NAME],
        "transformer_candidates": transformer_candidates,
        "default_transformer": TRANSFORMER_MODEL_NAME,
        "suggested_second_transformer": SECOND_TRANSFORMER_MODEL_NAME,
        "calibration": {
            "logistic_regression": "predict_proba",
            "linear_svc": "softmax(decision_function) plus calibrated sigmoid cv=3 report",
            "transformers": "softmax(logits)",
        },
        "will_train_transformers": bool(transformer_candidates),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate JournalPulse model artifacts.")
    parser.add_argument(
        "--transformers",
        default=None,
        help=(
            "Comma-separated Hugging Face transformer candidates. "
            "Default keeps current behavior: distilroberta-base."
        ),
    )
    parser.add_argument(
        "--skip-transformers",
        action="store_true",
        help="Train only the classical baselines and select the best classical model.",
    )
    parser.add_argument(
        "--dry-run-candidates",
        action="store_true",
        help="Print the training plan and exit without loading data or training.",
    )
    return parser.parse_args()


def transformer_artifact_dir(model_name: str) -> str:
    if model_name == TRANSFORMER_MODEL_NAME:
        return "transformer_model"
    slug = re.sub(r"[^a-z0-9]+", "_", model_name.lower()).strip("_")
    return f"transformer_{slug}"


def load_training_splits() -> dict:
    """Download the emotion dataset and return pandas dataframes."""

    dataset = load_dataset("dair-ai/emotion")
    splits = {}
    for split_name in ("train", "validation", "test"):
        frame = dataset[split_name].to_pandas()
        frame["clean_text"] = frame["text"].map(normalize_text)
        splits[split_name] = frame
    return splits


def set_seeds() -> None:
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)


def softmax_rows(scores) -> np.ndarray:
    values = np.asarray(scores, dtype=float)
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    values = values - np.max(values, axis=1, keepdims=True)
    exps = np.exp(values)
    return exps / np.sum(exps, axis=1, keepdims=True)


def confidence_band_for_value(confidence: float) -> str:
    if confidence >= 0.75:
        return "high"
    if confidence >= 0.55:
        return "medium"
    return "low"


def calibration_summary(y_true, predictions, scores) -> dict:
    score_array = np.asarray(scores, dtype=float)
    y_true = np.asarray(y_true, dtype=int)
    predictions = np.asarray(predictions, dtype=int)
    if score_array.ndim == 1:
        confidences = score_array
        brier_score = None
    else:
        confidences = np.max(score_array, axis=1)
        one_hot = np.zeros_like(score_array, dtype=float)
        for row_index, label in enumerate(y_true):
            if 0 <= label < one_hot.shape[1]:
                one_hot[row_index, label] = 1.0
        brier_score = round(float(np.mean(np.sum((score_array - one_hot) ** 2, axis=1))), 4)

    correct = predictions == y_true
    bins = np.linspace(0.0, 1.0, 11)
    expected_calibration_error = 0.0
    for lower, upper in zip(bins[:-1], bins[1:]):
        if upper == 1.0:
            mask = (confidences >= lower) & (confidences <= upper)
        else:
            mask = (confidences >= lower) & (confidences < upper)
        if not np.any(mask):
            continue
        bin_accuracy = float(np.mean(correct[mask]))
        bin_confidence = float(np.mean(confidences[mask]))
        expected_calibration_error += float(np.mean(mask)) * abs(bin_accuracy - bin_confidence)

    bands = {}
    for band in ("low", "medium", "high"):
        mask = np.asarray([confidence_band_for_value(value) == band for value in confidences])
        if not np.any(mask):
            bands[band] = {"count": 0, "accuracy": None, "average_confidence": None}
            continue
        bands[band] = {
            "count": int(np.sum(mask)),
            "accuracy": round(float(np.mean(correct[mask])), 4),
            "average_confidence": round(float(np.mean(confidences[mask])), 4),
        }

    return {
        "brier_score": brier_score,
        "expected_calibration_error": round(expected_calibration_error, 4),
        "confidence_band_accuracy": bands,
    }


def evaluate_predictions(y_true, predictions, scores, frame, *, probability_source: str) -> dict:
    report = classification_report(
        y_true,
        predictions,
        target_names=[LABELS[idx] for idx in sorted(LABELS)],
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(y_true, predictions, labels=list(sorted(LABELS)))

    score_array = np.asarray(scores, dtype=float)
    if score_array.ndim == 1:
        confidences = score_array
    else:
        confidences = np.max(score_array, axis=1)

    examples = []
    sample = frame.assign(
        predicted=predictions,
        confidence=confidences,
    )
    sample["is_correct"] = sample["label"] == sample["predicted"]
    correct_examples = sample[sample["is_correct"]].head(2)
    incorrect_examples = sample[~sample["is_correct"]].head(3)
    for _, row in pd.concat([correct_examples, incorrect_examples]).iterrows():
        examples.append(
            {
                "text": row["text"],
                "true_label": LABELS[int(row["label"])],
                "predicted_label": LABELS[int(row["predicted"])],
                "confidence": round(float(row["confidence"]), 4),
                "correct": bool(row["is_correct"]),
            }
        )

    return {
        "accuracy": round(float(accuracy_score(y_true, predictions)), 4),
        "macro_f1": round(float(f1_score(y_true, predictions, average="macro")), 4),
        "classification_report": report,
        "confusion_matrix": matrix.tolist(),
        "calibration": {
            "probability_source": probability_source,
            **calibration_summary(y_true, predictions, scores),
        },
        "examples": examples,
    }


def train_logistic_baseline(splits: dict) -> tuple:
    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=30000)),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    random_state=RANDOM_SEED,
                    multi_class="auto",
                ),
            ),
        ]
    )
    pipeline.fit(splits["train"]["clean_text"], splits["train"]["label"])

    probabilities = pipeline.predict_proba(splits["test"]["clean_text"])
    predictions = np.argmax(probabilities, axis=1)
    metrics = evaluate_predictions(
        splits["test"]["label"],
        predictions,
        probabilities,
        splits["test"],
        probability_source="predict_proba",
    )
    return pipeline, metrics


def train_linear_svc_baseline(splits: dict) -> tuple:
    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=30000)),
            ("classifier", LinearSVC(random_state=RANDOM_SEED)),
        ]
    )
    pipeline.fit(splits["train"]["clean_text"], splits["train"]["label"])

    decision_scores = pipeline.decision_function(splits["test"]["clean_text"])
    probabilities = softmax_rows(decision_scores)
    predictions = np.argmax(probabilities, axis=1)
    metrics = evaluate_predictions(
        splits["test"]["label"],
        predictions,
        probabilities,
        splits["test"],
        probability_source="softmax_decision_function",
    )

    calibrated_pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=30000)),
            (
                "classifier",
                CalibratedClassifierCV(
                    estimator=LinearSVC(random_state=RANDOM_SEED),
                    method="sigmoid",
                    cv=3,
                ),
            ),
        ]
    )
    calibrated_pipeline.fit(splits["train"]["clean_text"], splits["train"]["label"])
    calibrated_probabilities = calibrated_pipeline.predict_proba(splits["test"]["clean_text"])
    calibrated_predictions = np.argmax(calibrated_probabilities, axis=1)
    metrics["calibrated_probability_metrics"] = evaluate_predictions(
        splits["test"]["label"],
        calibrated_predictions,
        calibrated_probabilities,
        splits["test"],
        probability_source="calibrated_sigmoid_cv3",
    )
    return pipeline, metrics


def freeze_supported_encoder_layers(model) -> int:
    base_model = (
        getattr(model, "roberta", None)
        or getattr(model, "bert", None)
        or getattr(model, "distilbert", None)
    )
    if base_model is None:
        return 0

    embeddings = getattr(base_model, "embeddings", None)
    if embeddings is not None:
        for parameter in embeddings.parameters():
            parameter.requires_grad = False

    layers = []
    encoder = getattr(base_model, "encoder", None)
    if encoder is not None:
        layers = list(getattr(encoder, "layer", []))
    transformer = getattr(base_model, "transformer", None)
    if not layers and transformer is not None:
        layers = list(getattr(transformer, "layer", []))

    frozen_count = 0
    for layer in layers[:2]:
        for parameter in layer.parameters():
            parameter.requires_grad = False
        frozen_count += 1
    return frozen_count


def train_transformer(splits: dict, model_name: str) -> tuple:
    try:
        import torch
        from torch.optim import AdamW
        from torch.utils.data import DataLoader, Dataset
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            get_linear_schedule_with_warmup,
        )
    except ImportError:
        return None, {"status": "skipped", "reason": "torch/transformers are not installed"}, None

    class EmotionDataset(Dataset):
        def __init__(self, texts, labels, tokenizer, max_length):
            self.texts = list(texts)
            self.labels = list(labels)
            self.tokenizer = tokenizer
            self.max_length = max_length

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, index):
            encoded = self.tokenizer(
                self.texts[index],
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt",
            )
            item = {key: value.squeeze(0) for key, value in encoded.items()}
            item["labels"] = torch.tensor(self.labels[index], dtype=torch.long)
            return item

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(LABELS),
        id2label={idx: label for idx, label in LABELS.items()},
        label2id={label: idx for idx, label in LABELS.items()},
    ).to(device)

    frozen_layers = freeze_supported_encoder_layers(model)

    max_length = MAX_TRANSFORMER_LENGTH
    train_dataset = EmotionDataset(
        splits["train"]["text"].tolist(),
        splits["train"]["label"].tolist(),
        tokenizer,
        max_length,
    )
    validation_dataset = EmotionDataset(
        splits["validation"]["text"].tolist(),
        splits["validation"]["label"].tolist(),
        tokenizer,
        max_length,
    )
    test_dataset = EmotionDataset(
        splits["test"]["text"].tolist(),
        splits["test"]["label"].tolist(),
        tokenizer,
        max_length,
    )

    batch_size = 32
    epochs = 2
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = AdamW(trainable_parameters, lr=2e-5)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, total_steps // 10),
        num_training_steps=total_steps,
    )

    def run_eval(loader):
        model.eval()
        predictions = []
        labels = []
        probabilities = []
        with torch.no_grad():
            for batch in loader:
                batch = {key: value.to(device) for key, value in batch.items()}
                outputs = model(**batch)
                probs = torch.softmax(outputs.logits, dim=-1)
                predictions.extend(torch.argmax(probs, dim=-1).cpu().numpy().tolist())
                labels.extend(batch["labels"].cpu().numpy().tolist())
                probabilities.extend(probs.cpu().numpy().tolist())
        return np.asarray(labels), np.asarray(predictions), np.asarray(probabilities)

    best_state = None
    best_val_f1 = -1.0
    for epoch_index in range(epochs):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)
            outputs.loss.backward()
            optimizer.step()
            scheduler.step()

        val_labels, val_predictions, val_probabilities = run_eval(validation_loader)
        val_metrics = evaluate_predictions(
            val_labels,
            val_predictions,
            val_probabilities,
            splits["validation"],
            probability_source="softmax_logits",
        )
        print(
            f"{model_name} epoch {epoch_index + 1}/{epochs} "
            f"- validation accuracy: {val_metrics['accuracy']:.4f} "
            f"- validation macro F1: {val_metrics['macro_f1']:.4f}"
        )
        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            best_state = copy.deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)

    test_labels, test_predictions, test_probabilities = run_eval(test_loader)
    metrics = evaluate_predictions(
        test_labels,
        test_predictions,
        test_probabilities,
        splits["test"],
        probability_source="softmax_logits",
    )
    metadata = {
        "model_name": model_name,
        "tokenizer_name": model_name,
        "artifact_dir": transformer_artifact_dir(model_name),
        "max_length": max_length,
        "device_used": str(device),
        "epochs": epochs,
        "batch_size": batch_size,
        "frozen_encoder_layers": frozen_layers,
    }
    return model, metrics, {"tokenizer": tokenizer, "metadata": metadata}


def choose_classical_explainer(logreg_model, logreg_metrics, svc_model, svc_metrics) -> tuple:
    margin = svc_metrics["macro_f1"] - logreg_metrics["macro_f1"]
    if margin >= 0.01:
        return LINEAR_SVC_MODEL_NAME, svc_model, svc_metrics
    return BASELINE_MODEL_NAME, logreg_model, logreg_metrics


def valid_metrics(metrics: dict) -> bool:
    return bool(metrics) and metrics.get("status") != "skipped" and "macro_f1" in metrics


def choose_production_candidate(explainer_metrics: dict, transformer_artifacts: dict) -> tuple:
    selected_model = "classical_explainer"
    selected_transformer_name = None
    selected_score = explainer_metrics["macro_f1"]

    for model_name, artifact in transformer_artifacts.items():
        metrics = artifact["metrics"]
        if valid_metrics(metrics) and metrics["macro_f1"] > selected_score:
            selected_model = "transformer"
            selected_transformer_name = model_name
            selected_score = metrics["macro_f1"]

    return selected_model, selected_transformer_name


def write_markdown_report(report_path: Path, payload: dict) -> None:
    logreg = payload["models"]["logistic_regression"]
    linear_svc = payload["models"]["linear_svc"]
    classical = payload["models"]["classical_explainer"]
    lines = [
        "# Emotion Journal Evaluation",
        "",
        f"- Generated: {payload['generated_at']}",
        f"- Selected production model: `{payload['selected_model']}`",
        f"- Selected model name: `{payload['selected_model_name']}`",
        f"- Selected classical explainer: `{classical['model_name']}`",
        f"- Logistic Regression macro F1: `{logreg['macro_f1']}`",
        f"- LinearSVC macro F1: `{linear_svc['macro_f1']}`",
    ]

    calibrated = linear_svc.get("calibrated_probability_metrics")
    if calibrated:
        lines.append(f"- Calibrated LinearSVC macro F1: `{calibrated['macro_f1']}`")

    transformers = payload["models"].get("transformers", {})
    for model_name, metrics in transformers.items():
        if valid_metrics(metrics):
            lines.append(f"- Transformer `{model_name}` macro F1: `{metrics['macro_f1']}`")
        else:
            lines.append(f"- Transformer `{model_name}` status: `{metrics.get('status', 'unknown')}`")

    lines.extend(
        [
            "",
            "## Test Metrics",
            "",
            "| Model | Accuracy | Macro F1 | ECE |",
            "| --- | --- | --- | --- |",
            (
                f"| Logistic Regression ({BASELINE_MODEL_NAME}) | {logreg['accuracy']} | "
                f"{logreg['macro_f1']} | {logreg['calibration']['expected_calibration_error']} |"
            ),
            (
                f"| LinearSVC ({LINEAR_SVC_MODEL_NAME}) | {linear_svc['accuracy']} | "
                f"{linear_svc['macro_f1']} | {linear_svc['calibration']['expected_calibration_error']} |"
            ),
        ]
    )
    if calibrated:
        lines.append(
            f"| Calibrated LinearSVC | {calibrated['accuracy']} | {calibrated['macro_f1']} | "
            f"{calibrated['calibration']['expected_calibration_error']} |"
        )
    for model_name, metrics in transformers.items():
        if valid_metrics(metrics):
            lines.append(
                f"| Transformer ({model_name}) | {metrics['accuracy']} | {metrics['macro_f1']} | "
                f"{metrics['calibration']['expected_calibration_error']} |"
            )

    lines.extend(["", "## Example Predictions", ""])
    if payload["selected_model"] == "transformer":
        selected = transformers[payload["selected_model_name"]]
    else:
        selected = classical
    for example in selected["examples"]:
        verdict = "correct" if example["correct"] else "incorrect"
        lines.append(
            f"- `{verdict}` | true=`{example['true_label']}` | predicted=`{example['predicted_label']}` | "
            f"confidence=`{example['confidence']}` | text=\"{example['text']}\""
        )

    report_path.write_text("\n".join(lines) + "\n")


def save_production_artifacts(
    *,
    selected_model: str,
    selected_transformer_name: str,
    explainer_model_name: str,
    explainer_model,
    logreg_metrics: dict,
    linear_svc_metrics: dict,
    explainer_metrics: dict,
    transformer_artifacts: dict,
    training_config: dict,
) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    generated_at = datetime.now(timezone.utc).isoformat()
    baseline_path = MODELS_DIR / "baseline.joblib"
    joblib.dump(explainer_model, baseline_path)

    transformer_metrics = {
        model_name: artifact["metrics"]
        for model_name, artifact in transformer_artifacts.items()
    }
    for model_name, artifact in transformer_artifacts.items():
        model = artifact["model"]
        bundle = artifact["bundle"]
        if model is None or bundle is None:
            continue
        artifact_dir = MODELS_DIR / bundle["metadata"]["artifact_dir"]
        artifact_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(artifact_dir)
        bundle["tokenizer"].save_pretrained(artifact_dir)

    selected_model_name = explainer_model_name
    if selected_model == "transformer":
        selected_model_name = selected_transformer_name

    payload = {
        "generated_at": generated_at,
        "selected_model": selected_model,
        "selected_model_name": selected_model_name,
        "training_config": training_config,
        "models": {
            "logistic_regression": logreg_metrics,
            "linear_svc": linear_svc_metrics,
            "classical_explainer": {
                **explainer_metrics,
                "model_name": explainer_model_name,
            },
            "transformers": transformer_metrics,
        },
    }
    if TRANSFORMER_MODEL_NAME in transformer_metrics:
        payload["models"]["transformer"] = transformer_metrics[TRANSFORMER_MODEL_NAME]

    if selected_model == "classical_explainer":
        production_metadata = {
            "model_type": "sklearn_pipeline",
            "model_name": explainer_model_name,
            "artifact_path": "baseline.joblib",
            "baseline_artifact_path": "baseline.joblib",
            "label_map": LABELS,
            "selected_metric": "macro_f1",
            "trained_at": generated_at,
            "metrics": explainer_metrics,
        }
    else:
        selected_artifact = transformer_artifacts[selected_transformer_name]
        selected_bundle = selected_artifact["bundle"]
        production_metadata = {
            "model_type": "hf_transformer",
            "model_name": selected_transformer_name,
            "artifact_dir": selected_bundle["metadata"]["artifact_dir"],
            "baseline_artifact_path": "baseline.joblib",
            "max_length": selected_bundle["metadata"]["max_length"],
            "label_map": LABELS,
            "selected_metric": "macro_f1",
            "trained_at": generated_at,
            "metrics": selected_artifact["metrics"],
            "transformer_candidates": list(transformer_metrics),
        }

    (MODELS_DIR / "production.json").write_text(json.dumps(production_metadata, indent=2))
    (REPORTS_DIR / "evaluation.json").write_text(json.dumps(payload, indent=2))
    write_markdown_report(REPORTS_DIR / "evaluation.md", payload)


def main() -> None:
    args = parse_args()
    training_plan = build_training_plan(args)
    if args.dry_run_candidates:
        print(json.dumps(training_plan, indent=2))
        return

    set_seeds()
    splits = load_training_splits()
    logreg_model, logreg_metrics = train_logistic_baseline(splits)
    linear_svc_model, linear_svc_metrics = train_linear_svc_baseline(splits)
    explainer_model_name, explainer_model, explainer_metrics = choose_classical_explainer(
        logreg_model,
        logreg_metrics,
        linear_svc_model,
        linear_svc_metrics,
    )

    transformer_artifacts = {}
    for model_name in training_plan["transformer_candidates"]:
        transformer_model, transformer_metrics, transformer_bundle = train_transformer(splits, model_name)
        transformer_artifacts[model_name] = {
            "model": transformer_model,
            "metrics": transformer_metrics,
            "bundle": transformer_bundle,
        }

    selected_model, selected_transformer_name = choose_production_candidate(
        explainer_metrics,
        transformer_artifacts,
    )

    save_production_artifacts(
        selected_model=selected_model,
        selected_transformer_name=selected_transformer_name,
        explainer_model_name=explainer_model_name,
        explainer_model=explainer_model,
        logreg_metrics=logreg_metrics,
        linear_svc_metrics=linear_svc_metrics,
        explainer_metrics=explainer_metrics,
        transformer_artifacts=transformer_artifacts,
        training_config=training_plan,
    )
    print(
        "Training complete. Selected production model: "
        f"{selected_model} | selected model name: "
        f"{selected_transformer_name or explainer_model_name} | "
        f"selected classical explainer: {explainer_model_name}"
    )


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
