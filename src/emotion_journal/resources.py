import json
import re
from datetime import date
from functools import lru_cache
from typing import Dict, Iterable, List, Optional

from .config import (
    COPING_STYLES,
    DEFAULT_DB_PATH,
    DEFAULT_RESOURCE_TYPES,
    LABELS,
    RESOURCE_CATALOG_PATH,
    RESOURCE_GOAL_TAGS,
    RESOURCE_LIMIT_PER_STYLE,
    SOURCE_TIERS,
)
from .db import list_resource_interactions
from .llm import maybe_generate_resource_suggestions

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
    "goal_tags",
    "source_tier",
    "reviewed_at",
}

GOAL_ALIASES = {
    "ground": "ground",
    "grounding": "ground",
    "plan": "planning",
    "planning": "planning",
    "reframe": "reframing",
    "reframing": "reframing",
    "connect": "connection",
    "connection": "connection",
    "move": "movement",
    "movement": "movement",
    "read": "reading",
    "reading": "reading",
    "watch": "watching",
    "watching": "watching",
    "play": "play",
    "tips": "ground",
    "resource": None,
    "none": None,
}

GOAL_LABELS = {
    "ground": "grounding",
    "planning": "planning",
    "reframing": "reframing",
    "connection": "connection",
    "movement": "movement",
    "reading": "reading",
    "watching": "watching",
    "play": "play",
}


@lru_cache(maxsize=1)
def load_resource_catalog() -> List[dict]:
    catalog = json.loads(RESOURCE_CATALOG_PATH.read_text())
    normalized = []
    for resource in catalog:
        item = dict(resource)
        item["emotion_tags"] = list(item.get("emotion_tags", []))
        item["tone_tags"] = list(item.get("tone_tags", []))
        item["goal_tags"] = list(item.get("goal_tags", []))
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

        goal_tags = set(resource.get("goal_tags", []))
        unknown_goals = sorted(goal_tags - set(RESOURCE_GOAL_TAGS))
        if unknown_goals:
            errors.append(f"{label} has unknown goal_tags: {', '.join(unknown_goals)}")
        if resource.get("resource_type") != "support" and not goal_tags:
            errors.append(f"{label} needs at least one goal tag")

        if resource.get("source_tier") not in SOURCE_TIERS:
            errors.append(f"{label} has unsupported source_tier: {resource.get('source_tier')}")
        reviewed_at = resource.get("reviewed_at")
        if not isinstance(reviewed_at, str) or not re.match(r"^\d{4}-\d{2}-\d{2}$", reviewed_at):
            errors.append(f"{label} needs reviewed_at as YYYY-MM-DD")

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
    counts_by_source_tier = {source_tier: 0 for source_tier in SOURCE_TIERS}
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
        source_tier = resource.get("source_tier")
        if source_tier in counts_by_source_tier:
            counts_by_source_tier[source_tier] += 1

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
        "counts_by_source_tier": counts_by_source_tier,
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
    goal_tags: Optional[Iterable[str]] = None,
    source_tier: str = "educational",
    reviewed_at: str = "2026-05-09",
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
        "goal_tags": sorted({tag.strip().lower() for tag in (goal_tags or []) if tag.strip()}),
        "source_tier": source_tier.strip().lower(),
        "reviewed_at": reviewed_at.strip(),
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


def normalize_resource_goal(goal: Optional[str]) -> Optional[str]:
    if not goal:
        return None
    return GOAL_ALIASES.get(str(goal).strip().lower(), str(goal).strip().lower())


_STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "have", "has", "had", "are", "was",
    "were", "but", "not", "you", "your", "she", "her", "him", "his", "they", "them",
    "from", "what", "when", "feel", "feeling", "felt", "just", "really", "like",
    "about", "would", "could", "should", "their", "there", "been", "than", "then",
    "into", "some", "more", "very", "much", "over", "after", "before", "because",
}


def _content_terms(text: Optional[str], extra: Optional[Iterable[str]] = None) -> set:
    """Return meaningful lowercase word stems from free text plus extra tags."""

    terms = set()
    if text:
        for token in re.findall(r"[a-z]{4,}", text.lower()):
            if token not in _STOPWORDS:
                terms.add(token)
    for tag in extra or []:
        for token in re.findall(r"[a-z]{3,}", str(tag).lower()):
            terms.add(token)
    return terms


def _resource_terms(resource: dict) -> set:
    blob = " ".join(
        [
            str(resource.get("title", "")),
            str(resource.get("summary", "")),
            " ".join(resource.get("tone_tags", [])),
            " ".join(resource.get("goal_tags", [])),
            " ".join(resource.get("emotion_tags", [])),
            str(resource.get("provider", "")),
        ]
    )
    return _content_terms(blob)


def _content_overlap_score(resource: dict, query_terms: set) -> float:
    if not query_terms:
        return 0.0
    overlap = query_terms & _resource_terms(resource)
    if not overlap:
        return 0.0
    # Reward overlap but keep it bounded so it tunes ranking without dominating
    # the strong emotion/goal/style signals.
    return min(3.0, 0.9 * len(overlap))


def _resource_matches(resource: dict, emotion: Optional[str], coping_style: Optional[str], is_crisis: bool) -> bool:
    if is_crisis and not resource.get("is_crisis_safe"):
        return False
    if not is_crisis and resource.get("resource_type") == "support":
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
    goal: Optional[str],
    resource_scores: Dict[str, float],
    style_scores: Dict[str, float],
    coping_style: Optional[str],
    is_crisis: bool,
    query_terms: Optional[set] = None,
) -> float:
    score = 0.0
    if query_terms:
        score += _content_overlap_score(resource, query_terms)
    if emotion and emotion in resource.get("emotion_tags", []):
        score += 5.0
        resource_text = f"{resource.get('title', '')} {' '.join(resource.get('tone_tags', []))}".lower()
        if emotion in resource_text:
            score += 0.75
    if goal and goal in resource.get("goal_tags", []):
        score += 4.0
    if coping_style and resource.get("coping_style") == coping_style:
        score += 3.0
    if goal == "watching" and resource.get("coping_style") == "watch":
        score += 2.0
    if goal == "reading" and resource.get("coping_style") == "read":
        score += 2.0
    if goal == "movement" and resource.get("coping_style") == "move":
        score += 2.0
    if goal == "play" and resource.get("coping_style") == "play":
        score += 2.0
    if is_crisis and resource.get("resource_type") == "support":
        score += 10.0
    if resource.get("source_tier") in {"official", "crisis_support"}:
        score += 0.35
    if resource.get("embed_kind") == "youtube":
        score += 0.25

    score += resource_scores.get(resource["id"], 0.0)
    score += style_scores.get(resource.get("coping_style"), 0.0) * 0.35
    return score


def resource_rationale(resource: dict, *, emotion: Optional[str], goal: Optional[str] = None, is_crisis: bool = False) -> str:
    if is_crisis and resource.get("is_crisis_safe"):
        return "Prioritized because safety mode only shows crisis-safe human support resources."

    reasons = []
    if emotion and emotion in resource.get("emotion_tags", []):
        reasons.append(f"matches the {emotion} signal")
    normalized_goal = normalize_resource_goal(goal)
    if normalized_goal and normalized_goal in resource.get("goal_tags", []):
        reasons.append(f"supports {GOAL_LABELS.get(normalized_goal, normalized_goal)}")
    if resource.get("source_tier") == "official":
        reasons.append("comes from an official health source")
    elif resource.get("source_tier") == "nonprofit":
        reasons.append("comes from a nonprofit mental-health source")
    elif resource.get("source_tier") == "crisis_support":
        reasons.append("connects to human support")

    if not reasons:
        style = resource.get("coping_style", "support")
        reasons.append(f"adds a {style}-style option to the resource mix")
    return "Chosen because it " + ", ".join(reasons) + "."


def with_resource_rationales(
    resources: Iterable[dict],
    *,
    emotion: Optional[str],
    goal: Optional[str] = None,
    is_crisis: bool = False,
) -> List[dict]:
    enriched = []
    for resource in resources:
        item = dict(resource)
        item["rationale"] = resource_rationale(item, emotion=emotion, goal=goal, is_crisis=is_crisis)
        enriched.append(item)
    return enriched


def recommend_resources(
    emotion: Optional[str],
    *,
    coping_style: Optional[str] = None,
    goal: Optional[str] = None,
    is_crisis: bool = False,
    db_path=DEFAULT_DB_PATH,
    limit_per_style: int = RESOURCE_LIMIT_PER_STYLE,
    query: Optional[str] = None,
    query_extra_terms: Optional[Iterable[str]] = None,
) -> List[dict]:
    catalog = load_resource_catalog()
    resource_lookup = get_resource_lookup()
    interactions = list_resource_interactions(db_path=db_path)
    resource_scores, style_scores = _interaction_scores(interactions, resource_lookup)
    normalized_goal = normalize_resource_goal(goal)
    query_terms = _content_terms(query, query_extra_terms) if (query or query_extra_terms) else set()

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
                goal=normalized_goal,
                resource_scores=resource_scores,
                style_scores=style_scores,
                coping_style=coping_style,
                is_crisis=is_crisis,
                query_terms=query_terms,
            ),
            -resource.get("duration_minutes", 0),
            resource["title"],
        ),
        reverse=True,
    )

    if coping_style or is_crisis:
        return with_resource_rationales(
            ranked[: max(1, limit_per_style * 2)],
            emotion=emotion,
            goal=normalized_goal,
            is_crisis=is_crisis,
        )

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
    return with_resource_rationales(
        selected,
        emotion=emotion,
        goal=normalized_goal,
        is_crisis=is_crisis,
    )


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


def _llm_catalog_options(emotion: Optional[str]) -> List[dict]:
    """Compact, non-crisis catalog view handed to the LLM re-ranker."""

    options = []
    for resource in load_resource_catalog():
        if resource.get("resource_type") == "support":
            continue
        options.append(
            {
                "id": resource["id"],
                "title": resource.get("title"),
                "coping_style": resource.get("coping_style"),
                "resource_type": resource.get("resource_type"),
                "goal_tags": resource.get("goal_tags", []),
                "emotion_tags": resource.get("emotion_tags", []),
                "summary": resource.get("summary"),
            }
        )
    return options


def make_generated_resource_card(item: dict, *, emotion: Optional[str]) -> dict:
    """Turn a validated LLM suggestion into a catalog-shaped, badged resource card."""

    url = item["url"].strip()
    is_youtube = "youtube.com" in url or "youtu.be" in url
    goal_tags = [normalize_resource_goal(tag) or tag for tag in item.get("goal_tags", [])]
    goal_tags = [tag for tag in goal_tags if tag]
    return {
        "id": f"ai_{_slugify(item['title'])}"[:80].rstrip("_") or "ai_resource",
        "title": item["title"].strip(),
        "url": url,
        "resource_type": item.get("resource_type", "website"),
        "coping_style": item.get("coping_style", "read"),
        "provider": item.get("provider") or "AI-suggested",
        "embed_kind": "youtube" if is_youtube else "link",
        "duration_minutes": None,
        "summary": item.get("summary", "").strip(),
        "emotion_tags": [emotion] if emotion else [],
        "tone_tags": [],
        "goal_tags": goal_tags,
        "is_browser_safe": True,
        "is_crisis_safe": False,
        "source_tier": "educational",
        "reviewed_at": date.today().isoformat(),
        "rationale": item.get("why", "").strip() or "Suggested to fit what you wrote about.",
        "source": "ai_suggested",
    }


def build_recommendation_set(
    text: str,
    *,
    emotion: Optional[str],
    is_crisis: bool = False,
    analysis_context: Optional[dict] = None,
    db_path=DEFAULT_DB_PATH,
    use_llm: bool = False,
    limit_per_style: int = RESOURCE_LIMIT_PER_STYLE,
    max_generated: int = 3,
) -> tuple:
    """Return (resources, meta): content-aware catalog picks plus optional AI suggestions.

    Crisis entries always return the deterministic crisis-safe set untouched. Otherwise,
    when an LLM is configured and ``use_llm`` is set, the catalog is re-ranked by the model
    and a few safelisted, personalized suggestions are merged in (badged ``ai_suggested``).
    Any LLM failure silently falls back to the catalog-only result.
    """

    extra_terms = []
    if analysis_context:
        extra_terms = list(analysis_context.get("emotion_tags", [])) + list(
            analysis_context.get("themes", [])
        )
    # Crisis entries must keep their deterministic safety ordering (e.g. 988 first),
    # so content-aware re-ranking is only applied to ordinary journaling.
    base = recommend_resources(
        emotion,
        is_crisis=is_crisis,
        db_path=db_path,
        limit_per_style=limit_per_style,
        query=None if is_crisis else text,
        query_extra_terms=None if is_crisis else extra_terms,
    )
    for resource in base:
        resource.setdefault("source", "catalog")

    meta = {
        "used_llm_recommender": False,
        "generated_count": 0,
        "recommender_fallback_reason": None,
        "recommender_model": None,
    }
    if is_crisis:
        return base, meta

    payload, used, reason = maybe_generate_resource_suggestions(
        text,
        analysis_context=analysis_context or {"emotion": emotion},
        catalog_options=_llm_catalog_options(emotion),
        use_llm=use_llm,
        max_generated=max_generated,
    )
    meta["recommender_fallback_reason"] = reason
    if not used or payload is None:
        return base, meta

    catalog_lookup = get_resource_lookup()
    ranked_ids = payload.get("ranked_catalog_ids", [])

    # Re-rank the catalog picks by the model's ordering, keeping any it did not mention.
    base_by_id = {resource["id"]: resource for resource in base}
    reordered = [base_by_id[rid] for rid in ranked_ids if rid in base_by_id]
    # Pull in highly-ranked catalog items the deterministic pass missed.
    for rid in ranked_ids:
        if rid not in base_by_id and rid in catalog_lookup and len(reordered) < limit_per_style * len(COPING_STYLES):
            extra = with_resource_rationales([catalog_lookup[rid]], emotion=emotion)[0]
            extra.setdefault("source", "catalog")
            reordered.append(extra)
    for resource in base:
        if resource not in reordered:
            reordered.append(resource)

    generated_cards = [
        make_generated_resource_card(item, emotion=emotion)
        for item in payload.get("generated", [])
    ]
    seen_urls = {resource.get("url", "").rstrip("/") for resource in reordered}
    deduped_generated = []
    for card in generated_cards:
        key = card["url"].rstrip("/")
        if key in seen_urls:
            continue
        seen_urls.add(key)
        deduped_generated.append(card)

    meta["used_llm_recommender"] = True
    meta["generated_count"] = len(deduped_generated)
    meta["recommender_model"] = payload.get("agent_model")

    # Surface personalized suggestions near the top while keeping catalog grounding.
    return deduped_generated + reordered, meta
