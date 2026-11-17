from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlparse

REQUIRED_FIELDS = {
    "id",
    "title",
    "url",
    "summary",
    "provider",
    "resource_type",
    "coping_style",
}


@lru_cache(maxsize=4)
def load_catalog(path: Path) -> list[dict]:
    resources = json.loads(path.read_text(encoding="utf-8"))
    errors = validate_catalog(resources)
    if errors:
        raise ValueError("Invalid resource catalog: " + "; ".join(errors))
    return [resource for resource in resources if resource.get("is_browser_safe", True)]


def validate_catalog(resources: object) -> list[str]:
    if not isinstance(resources, list):
        return ["catalog must be a JSON array"]
    errors: list[str] = []
    seen: set[str] = set()
    for index, resource in enumerate(resources):
        if not isinstance(resource, dict):
            errors.append(f"resource[{index}] must be an object")
            continue
        missing = sorted(REQUIRED_FIELDS - resource.keys())
        if missing:
            errors.append(f"resource[{index}] missing: {', '.join(missing)}")
        resource_id = resource.get("id")
        if not isinstance(resource_id, str) or not resource_id.strip():
            errors.append(f"resource[{index}] has an invalid id")
        elif resource_id in seen:
            errors.append(f"duplicate resource id: {resource_id}")
        else:
            seen.add(resource_id)
        url = resource.get("url")
        if not isinstance(url, str) or urlparse(url).scheme != "https":
            errors.append(f"{resource_id or index} must use an https URL")
    return errors


def catalog_summary(resources: list[dict]) -> dict[str, int]:
    return {
        "total_resources": len(resources),
        "support_resources": sum(item.get("resource_type") == "support" for item in resources),
        "browser_safe_resources": sum(item.get("is_browser_safe", True) for item in resources),
    }


def approved_actions(path: Path, *, intent: str, support_ids: list[str] | None = None) -> list[dict]:
    resources = load_catalog(path)
    if support_ids:
        lookup = {resource["id"]: resource for resource in resources}
        return [lookup[item] for item in support_ids if item in lookup]

    intent_to_style = {
        "move": "move",
        "watch": "watch",
        "read": "read",
        "play": "play",
        "reflect": "read",
    }
    intent_to_goal = {
        "ground": "ground",
        "connect": "connection",
    }
    desired_style = intent_to_style.get(intent)
    desired_goal = intent_to_goal.get(intent)
    matched = [
        resource
        for resource in resources
        if resource.get("resource_type") != "support"
        and (desired_style is None or resource.get("coping_style") == desired_style)
        and (desired_goal is None or desired_goal in resource.get("goal_tags", []))
    ]
    return matched[:8]


def goal_for_intent(resource_intent: str) -> str:
    intents = {
        "ground": "settle",
        "move": "move",
        "connect": "connect",
        "reflect": "understand",
        "read": "understand",
        "play": "act",
        "watch": "settle",
        "pause": "settle",
    }
    return intents.get(resource_intent, "settle")


def action_intent(resource_intent: str, goal: str) -> str:
    goal_intents = {
        "settle": "ground",
        "move": "move",
        "understand": "read",
        "connect": "connect",
        "act": "move",
    }
    return goal_intents.get(goal, resource_intent)
