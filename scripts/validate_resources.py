from __future__ import annotations

import argparse
import json
from pathlib import Path

import httpx

from journalpulse.config import load_settings
from journalpulse.resources import catalog_summary, load_catalog, validate_catalog


def check_links(resources: list[dict], *, timeout_seconds: float = 8.0) -> list[str]:
    errors: list[str] = []
    headers = {"User-Agent": "JournalPulseResourceValidator/2.0"}
    with httpx.Client(follow_redirects=True, timeout=timeout_seconds, headers=headers) as client:
        for resource in resources:
            try:
                response = client.head(resource["url"])
                if response.status_code in {403, 405}:
                    response = client.get(resource["url"])
                if response.status_code >= 400:
                    errors.append(f"{resource['id']} returned HTTP {response.status_code}")
            except httpx.HTTPError as exc:
                errors.append(f"{resource['id']} failed: {exc.__class__.__name__}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the approved JournalPulse catalog.")
    parser.add_argument("--check-links", action="store_true")
    args = parser.parse_args()
    path: Path = load_settings().resource_catalog_path
    raw = json.loads(path.read_text(encoding="utf-8"))
    errors = validate_catalog(raw)
    resources = load_catalog(path) if not errors else []
    if args.check_links and resources:
        errors.extend(check_links(resources))
    summary = catalog_summary(resources)
    print(f"resources: {summary['total_resources']}")
    print(f"support_resources: {summary['support_resources']}")
    print(f"validation_errors: {len(errors)}")
    for error in errors:
        print(f"- {error}")
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())
