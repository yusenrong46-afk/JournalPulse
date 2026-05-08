import json
import re
from functools import lru_cache
from typing import Dict, Iterable, List, Optional

from .config import (
    COPING_STYLES,
    DEFAULT_DB_PATH,
    DEFAULT_RESOURCE_TYPES,
    LABELS,
    RESOURCE_CATALOG_PATH,
    RESOURCE_LIMIT_PER_STYLE,
)
from .db import list_resource_interactions

REQUIRED_RESOURCE_FIELDS = {
    "id",
    "title",
    "url",
    "resource_type",
    "coping_style",
    "provider",
    "embed_kind",
    "duration_minutes",
    "summary",
    "emotion_tags",
    "tone_tags",
    "is_browser_safe",
    "is_crisis_safe",
}


@lru_cache(maxsize=1)
def load_resource_catalog() -> List[dict]:
    catalog = json.loads(RESOURCE_CATALOG_PATH.read_text())
    normalized = []
    for resource in catalog:
        item = dict(resource)
        item["emotion_tags"] = list(item.get("emotion_tags", []))
        item["tone_tags"] = list(item.get("tone_tags", []))
        normalized.append(item)
    return normalized


def validate_resource_catalog(resources: Optional[Iterable[dict]] = None) -> List[str]:
    catalog = list(resources or load_resource_catalog())
    errors = []
    seen_ids = set()
    valid_emotions = set(LABELS.values())

    for index, resource in enumerate(catalog):
        label = resource.get("id", f"index:{index}")
        missing = sorted(REQUIRED_RESOURCE_FIELDS - set(resource))
        if missing:
            errors.append(f"{label} is missing required fields: {', '.join(missing)}")

        resource_id = resource.get("id")
        if resource_id in seen_ids:
            errors.append(f"{resource_id} is duplicated")
        if resource_id:
            seen_ids.add(resource_id)

        if resource.get("resource_type") not in DEFAULT_RESOURCE_TYPES:
            errors.append(f"{label} has unsupported resource_type: {resource.get('resource_type')}")
        if resource.get("coping_style") not in COPING_STYLES:
            errors.append(f"{label} has unsupported coping_style: {resource.get('coping_style')}")

        emotion_tags = set(resource.get("emotion_tags", []))
        unknown_emotions = sorted(emotion_tags - valid_emotions)
        if unknown_emotions:
            errors.append(f"{label} has unknown emotion_tags: {', '.join(unknown_emotions)}")

        if resource.get("resource_type") != "support" and not emotion_tags:
            errors.append(f"{label} needs at least one emotion tag")
        if not resource.get("url", "").startswith(("http://", "https://")):
            errors.append(f"{label} needs an http(s) URL")

    return errors


def resource_catalog_summary(resources: Optional[Iterable[dict]] = None) -> dict:
    catalog = list(resources or load_resource_catalog())
    counts_by_emotion = {emotion: 0 for emotion in LABELS.values()}
    counts_by_coping_style = {style: 0 for style in COPING_STYLES}
    counts_by_type = {resource_type: 0 for resource_type in DEFAULT_RESOURCE_TYPES}
    coverage = {
        emotion: {style: 0 for style in COPING_STYLES}
        for emotion in LABELS.values()
    }

    for resource in catalog:
        style = resource.get("coping_style")
        resource_type = resource.get("resource_type")
        if style in counts_by_coping_style:
            counts_by_coping_style[style] += 1
        if resource_type in counts_by_type:
            counts_by_type[resource_type] += 1

        if resource_type == "support":
            continue
        for emotion in resource.get("emotion_tags", []):
            if emotion in counts_by_emotion:
                counts_by_emotion[emotion] += 1
            if emotion in coverage and style in coverage[emotion]:
                coverage[emotion][style] += 1

    coverage_gaps = []
    for emotion, styles in coverage.items():
        for style, count in styles.items():
            if count == 0:
                coverage_gaps.append({"emotion": emotion, "coping_style": style})

    return {
        "total_resources": len(catalog),
        "counts_by_emotion": counts_by_emotion,
        "counts_by_coping_style": counts_by_coping_style,
        "counts_by_type": counts_by_type,
        "crisis_safe_count": sum(1 for resource in catalog if resource.get("is_crisis_safe")),
        "coverage_gaps": coverage_gaps,
        "validation_errors": validate_resource_catalog(catalog),
    }


def _slugify(value: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return cleaned or "resource"


def build_resource_draft(
    *,
    title: str,
    url: str,
    resource_type: str,
    coping_style: str,
    provider: str,
    embed_kind: str,
    summary: str,
    emotion_tags: Optional[Iterable[str]] = None,
    tone_tags: Optional[Iterable[str]] = None,
    duration_minutes: Optional[int] = None,
    is_browser_safe: bool = True,
    is_crisis_safe: bool = False,
    resource_id: Optional[str] = None,
) -> dict:
    """Build a normalized resource card for admin preview/download workflows."""

    provider_slug = _slugify(provider)
    title_slug = _slugify(title)
    return {
        "id": resource_id or f"{resource_type}_{provider_slug}_{title_slug}"[:80].rstrip("_"),
        "title": title.strip(),
        "url": url.strip(),
        "resource_type": resource_type,
        "coping_style": coping_style,
        "provider": provider.strip(),
        "embed_kind": embed_kind.strip() or "link",
        "duration_minutes": duration_minutes if duration_minutes and duration_minutes > 0 else None,
        "summary": summary.strip(),
        "emotion_tags": sorted({tag.strip().lower() for tag in (emotion_tags or []) if tag.strip()}),
        "tone_tags": sorted({tag.strip().lower() for tag in (tone_tags or []) if tag.strip()}),
        "is_browser_safe": bool(is_browser_safe),
        "is_crisis_safe": bool(is_crisis_safe),
    }


def resource_admin_snapshot(resources: Optional[Iterable[dict]] = None) -> dict:
    catalog = list(resources or load_resource_catalog())
    summary = resource_catalog_summary(catalog)
    return {
        "summary": summary,
        "resources": sorted(catalog, key=lambda item: item["id"]),
        "coverage_gaps": summary["coverage_gaps"],
        "validation_errors": summary["validation_errors"],
    }


def get_resource_lookup() -> Dict[str, dict]:
    return {resource["id"]: resource for resource in load_resource_catalog()}


def _resource_matches(resource: dict, emotion: Optional[str], coping_style: Optional[str], is_crisis: bool) -> bool:
    if is_crisis and not resource.get("is_crisis_safe"):
        return False
    if coping_style and resource.get("coping_style") != coping_style:
        return False
    if is_crisis:
        return resource.get("resource_type") == "support" or resource.get("is_crisis_safe")
    if emotion is None:
        return True
    return emotion in resource.get("emotion_tags", [])


def _interaction_scores(interactions: Iterable[dict], resource_lookup: Dict[str, dict]) -> tuple:
    resource_scores: Dict[str, float] = {}
    style_scores: Dict[str, float] = {style: 0.0 for style in COPING_STYLES}
    for interaction in interactions:
        action = interaction["action"]
        resource_id = interaction["resource_id"]
        resource = resource_lookup.get(resource_id)
        if resource is None:
            continue

        delta = 0.0
        if action == "helpful":
            delta = 2.0
        elif action == "opened":
            delta = 0.75
        elif action == "dismissed":
            delta = -1.5

        resource_scores[resource_id] = resource_scores.get(resource_id, 0.0) + delta
        style = resource.get("coping_style")
        if style in style_scores:
            style_scores[style] += delta
    return resource_scores, style_scores


def _score_resource(
    resource: dict,
    *,
    emotion: Optional[str],
    resource_scores: Dict[str, float],
    style_scores: Dict[str, float],
    coping_style: Optional[str],
    is_crisis: bool,
) -> float:
    score = 0.0
    if emotion and emotion in resource.get("emotion_tags", []):
        score += 5.0
    if coping_style and resource.get("coping_style") == coping_style:
        score += 3.0
    if is_crisis and resource.get("resource_type") == "support":
        score += 10.0
    if resource.get("embed_kind") == "youtube":
        score += 0.25

    score += resource_scores.get(resource["id"], 0.0)
    score += style_scores.get(resource.get("coping_style"), 0.0) * 0.35
    return score


def recommend_resources(
    emotion: Optional[str],
    *,
    coping_style: Optional[str] = None,
    is_crisis: bool = False,
    db_path=DEFAULT_DB_PATH,
    limit_per_style: int = RESOURCE_LIMIT_PER_STYLE,
) -> List[dict]:
    catalog = load_resource_catalog()
    resource_lookup = get_resource_lookup()
    interactions = list_resource_interactions(db_path=db_path)
    resource_scores, style_scores = _interaction_scores(interactions, resource_lookup)

    candidates = [
        resource
        for resource in catalog
        if _resource_matches(resource, emotion, coping_style, is_crisis)
    ]

    ranked = sorted(
        candidates,
        key=lambda resource: (
            _score_resource(
                resource,
                emotion=emotion,
                resource_scores=resource_scores,
                style_scores=style_scores,
                coping_style=coping_style,
                is_crisis=is_crisis,
            ),
            -resource.get("duration_minutes", 0),
            resource["title"],
        ),
        reverse=True,
    )

    if coping_style or is_crisis:
        return ranked[: max(1, limit_per_style * 2)]

    selected = []
    seen_ids = set()
    for style in COPING_STYLES:
        style_items = [resource for resource in ranked if resource.get("coping_style") == style]
        for resource in style_items[:limit_per_style]:
            if resource["id"] in seen_ids:
                continue
            selected.append(resource)
            seen_ids.add(resource["id"])

    if len(selected) < limit_per_style * len(COPING_STYLES):
        for resource in ranked:
            if resource["id"] in seen_ids:
                continue
            selected.append(resource)
            seen_ids.add(resource["id"])
            if len(selected) >= limit_per_style * len(COPING_STYLES):
                break
    return selected


def filter_resources(
    *,
    emotion: Optional[str] = None,
    resource_type: Optional[str] = None,
    coping_style: Optional[str] = None,
    limit: Optional[int] = None,
) -> List[dict]:
    resources = load_resource_catalog()
    filtered = []
    for resource in resources:
        if emotion and emotion not in resource.get("emotion_tags", []):
            continue
        if resource_type and resource.get("resource_type") != resource_type:
            continue
        if coping_style and resource.get("coping_style") != coping_style:
            continue
        filtered.append(resource)
    return filtered[:limit] if limit else filtered


def resources_by_style(resources: Iterable[dict]) -> Dict[str, List[dict]]:
    grouped = {style: [] for style in COPING_STYLES}
    for resource in resources:
        style = resource.get("coping_style")
        if style in grouped:
            grouped[style].append(resource)
    return grouped


def resource_titles(resource_ids: Iterable[str]) -> List[str]:
    lookup = get_resource_lookup()
    titles = []
    for resource_id in resource_ids:
        resource = lookup.get(resource_id)
        if resource is not None:
            titles.append(resource["title"])
    return titles
