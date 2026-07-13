import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional

from .config import LABELS

VALID_EMOTIONS = set(LABELS.values())


@dataclass
class EvalCase:
    id: str
    text: str
    primary_emotion: str
    accepted_emotions: List[str]
    secondary_emotions: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    should_trigger_crisis: bool = False
    notes: Optional[str] = None


def _validate_emotions(values: Iterable[str], *, field_name: str, case_id: str) -> List[str]:
    cleaned = []
    for value in values:
        emotion = str(value).strip().lower()
        if emotion not in VALID_EMOTIONS:
            raise ValueError(f"{case_id}: {field_name} contains unknown emotion {value!r}")
        if emotion not in cleaned:
            cleaned.append(emotion)
    return cleaned


def load_eval_cases(path: Path) -> List[EvalCase]:
    cases = []
    seen_ids = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            payload = json.loads(line)
            case_id = str(payload.get("id", "")).strip()
            if not case_id:
                raise ValueError(f"line {line_number}: id is required")
            if case_id in seen_ids:
                raise ValueError(f"{case_id}: duplicate id")
            seen_ids.add(case_id)

            text = str(payload.get("text", "")).strip()
            if not text:
                raise ValueError(f"{case_id}: text is required")

            primary = str(payload.get("primary_emotion", "")).strip().lower()
            if primary not in VALID_EMOTIONS:
                raise ValueError(f"{case_id}: primary_emotion must be one of {sorted(VALID_EMOTIONS)}")

            accepted = _validate_emotions(
                payload.get("accepted_emotions") or [primary],
                field_name="accepted_emotions",
                case_id=case_id,
            )
            if primary not in accepted:
                accepted.insert(0, primary)

            secondary = _validate_emotions(
                payload.get("secondary_emotions") or [],
                field_name="secondary_emotions",
                case_id=case_id,
            )
            cases.append(
                EvalCase(
                    id=case_id,
                    text=text,
                    primary_emotion=primary,
                    accepted_emotions=accepted,
                    secondary_emotions=secondary,
                    tags=[str(tag).strip().lower() for tag in payload.get("tags", []) if str(tag).strip()],
                    should_trigger_crisis=bool(payload.get("should_trigger_crisis", False)),
                    notes=payload.get("notes"),
                )
            )
    return cases


def evaluate_prediction_cases(predictor, cases: Iterable[EvalCase]) -> dict:
    rows = []
    by_primary = defaultdict(lambda: {"total": 0, "primary_correct": 0, "accepted_correct": 0})
    by_tag = defaultdict(lambda: {"total": 0, "primary_correct": 0, "accepted_correct": 0})
    confusion = Counter()
    total = 0
    primary_correct = 0
    accepted_correct = 0
    top3_contains_primary = 0
    crisis_correct = 0
    crisis_total = 0
    non_crisis_total = 0
    non_crisis_primary_correct = 0
    non_crisis_accepted_correct = 0
    mixed_count = 0
    low_confidence_count = 0

    for case in cases:
        prediction = predictor.predict(case.text)
        score_items = sorted(prediction.scores.items(), key=lambda item: item[1], reverse=True)
        top3 = [label for label, _score in score_items[:3]]
        is_primary_correct = prediction.emotion == case.primary_emotion
        is_accepted_correct = prediction.emotion in case.accepted_emotions
        contains_primary = case.primary_emotion in top3
        crisis_matches = prediction.is_crisis == case.should_trigger_crisis

        total += 1
        primary_correct += int(is_primary_correct)
        accepted_correct += int(is_accepted_correct)
        top3_contains_primary += int(contains_primary)
        mixed_count += int(bool(getattr(prediction, "is_mixed", False)))
        low_confidence_count += int((prediction.confidence_band or "") == "low")
        confusion[(case.primary_emotion, prediction.emotion)] += 1

        if case.should_trigger_crisis or prediction.is_crisis:
            crisis_total += 1
            crisis_correct += int(crisis_matches)
        if not case.should_trigger_crisis:
            non_crisis_total += 1
            non_crisis_primary_correct += int(is_primary_correct)
            non_crisis_accepted_correct += int(is_accepted_correct)

        by_primary[case.primary_emotion]["total"] += 1
        by_primary[case.primary_emotion]["primary_correct"] += int(is_primary_correct)
        by_primary[case.primary_emotion]["accepted_correct"] += int(is_accepted_correct)
        for tag in case.tags:
            by_tag[tag]["total"] += 1
            by_tag[tag]["primary_correct"] += int(is_primary_correct)
            by_tag[tag]["accepted_correct"] += int(is_accepted_correct)

        rows.append(
            {
                "id": case.id,
                "text": case.text,
                "expected_primary": case.primary_emotion,
                "accepted_emotions": case.accepted_emotions,
                "predicted": prediction.emotion,
                "confidence": prediction.confidence,
                "confidence_band": prediction.confidence_band,
                "secondary_emotions": getattr(prediction, "secondary_emotions", []),
                "top_margin": getattr(prediction, "top_margin", None),
                "is_mixed": getattr(prediction, "is_mixed", False),
                "uncertainty_reason": getattr(prediction, "uncertainty_reason", None),
                "top3": top3,
                "primary_correct": is_primary_correct,
                "accepted_correct": is_accepted_correct,
                "top3_contains_primary": contains_primary,
                "expected_crisis": case.should_trigger_crisis,
                "predicted_crisis": prediction.is_crisis,
                "crisis_correct": crisis_matches,
                "tags": case.tags,
                "notes": case.notes,
            }
        )

    def summarize(bucket: dict) -> dict:
        count = bucket["total"]
        return {
            "total": count,
            "primary_accuracy": round(bucket["primary_correct"] / count, 4) if count else None,
            "accepted_accuracy": round(bucket["accepted_correct"] / count, 4) if count else None,
        }

    misses = [
        row
        for row in rows
        if not row["accepted_correct"] or row["expected_crisis"] != row["predicted_crisis"]
    ]
    misses = sorted(
        misses,
        key=lambda row: (row["accepted_correct"], row["confidence"]),
        reverse=True,
    )

    return {
        "total_cases": total,
        "primary_accuracy": round(primary_correct / total, 4) if total else None,
        "accepted_accuracy": round(accepted_correct / total, 4) if total else None,
        "top3_primary_recall": round(top3_contains_primary / total, 4) if total else None,
        "non_crisis_primary_accuracy": (
            round(non_crisis_primary_correct / non_crisis_total, 4) if non_crisis_total else None
        ),
        "non_crisis_accepted_accuracy": (
            round(non_crisis_accepted_correct / non_crisis_total, 4) if non_crisis_total else None
        ),
        "non_crisis_cases": non_crisis_total,
        "crisis_routing_accuracy": round(crisis_correct / crisis_total, 4) if crisis_total else None,
        "crisis_cases": crisis_total,
        "mixed_signal_rate": round(mixed_count / total, 4) if total else None,
        "low_confidence_rate": round(low_confidence_count / total, 4) if total else None,
        "by_primary_emotion": {emotion: summarize(bucket) for emotion, bucket in sorted(by_primary.items())},
        "by_tag": {tag: summarize(bucket) for tag, bucket in sorted(by_tag.items())},
        "confusion": [
            {"expected": expected, "predicted": predicted, "count": count}
            for (expected, predicted), count in sorted(confusion.items())
        ],
        "misses": misses,
        "rows": rows,
    }
