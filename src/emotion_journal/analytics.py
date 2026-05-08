from collections import Counter, defaultdict
from typing import Iterable, List, Mapping, Optional


def build_analytics(
    entries: Iterable[Mapping],
    *,
    resource_interactions: Optional[Iterable[Mapping]] = None,
    resource_catalog: Optional[Iterable[Mapping]] = None,
) -> dict:
    entries_list = list(entries)
    interaction_list = list(resource_interactions or [])
    resource_lookup = {resource["id"]: dict(resource) for resource in (resource_catalog or [])}

    counts = Counter(entry["emotion"] for entry in entries_list)
    trend_lookup = defaultdict(int)
    feedback_counts = Counter()
    confidence_band_counts = Counter()
    explanation_phrase_counts = defaultdict(Counter)
    resource_action_counts = Counter()
    helpful_resource_counts = Counter()
    preferred_coping_styles = Counter()

    helpful = 0
    rated = 0
    for entry in entries_list:
        day_bucket = str(entry["created_at"])[:10]
        trend_lookup[(day_bucket, entry["emotion"])] += 1
        feedback = entry.get("feedback")
        if feedback:
            feedback_counts[feedback] += 1
            if feedback in {"helpful", "not_helpful"}:
                rated += 1
            if feedback == "helpful":
                helpful += 1

        if entry.get("confidence_band"):
            confidence_band_counts[entry["confidence_band"]] += 1
        for phrase in entry.get("explanation_phrases", []):
            explanation_phrase_counts[entry["emotion"]][phrase] += 1

    for interaction in interaction_list:
        action = interaction["action"]
        resource_id = interaction["resource_id"]
        resource_action_counts[action] += 1
        resource = resource_lookup.get(resource_id)
        if resource is None:
            continue

        style = resource.get("coping_style")
        if action == "helpful":
            helpful_resource_counts[resource_id] += 1
            preferred_coping_styles[style] += 2
        elif action == "opened":
            preferred_coping_styles[style] += 1
        elif action == "dismissed":
            preferred_coping_styles[style] -= 1

    trend_buckets: List[dict] = []
    for (bucket_date, emotion), count in sorted(trend_lookup.items()):
        trend_buckets.append({"date": bucket_date, "emotion": emotion, "count": count})

    top_explanation_phrases_by_emotion = {}
    for emotion, counter in explanation_phrase_counts.items():
        top_explanation_phrases_by_emotion[emotion] = [
            {"phrase": phrase, "count": count}
            for phrase, count in counter.most_common(5)
        ]

    top_helpful_resources = []
    for resource_id, count in helpful_resource_counts.most_common(5):
        resource = resource_lookup.get(resource_id)
        if resource is None:
            continue
        top_helpful_resources.append(
            {
                "resource_id": resource_id,
                "title": resource["title"],
                "provider": resource["provider"],
                "count": count,
            }
        )

    helpfulness_rate = round(helpful / rated, 4) if rated else None
    filtered_style_preferences = {
        style: score
        for style, score in sorted(preferred_coping_styles.items())
        if score > 0
    }

    return {
        "total_entries": len(entries_list),
        "counts_by_emotion": dict(sorted(counts.items())),
        "trend_buckets": trend_buckets,
        "feedback_counts": dict(sorted(feedback_counts.items())),
        "feedback_usefulness_rate": helpfulness_rate,
        "confidence_band_counts": dict(sorted(confidence_band_counts.items())),
        "top_explanation_phrases_by_emotion": top_explanation_phrases_by_emotion,
        "resource_action_counts": dict(sorted(resource_action_counts.items())),
        "top_helpful_resources": top_helpful_resources,
        "preferred_coping_styles": filtered_style_preferences,
    }
