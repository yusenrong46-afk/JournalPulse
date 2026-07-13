import argparse
import sys
from pathlib import Path

import httpx

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from emotion_journal.resources import load_resource_catalog, resource_catalog_summary, validate_resource_catalog


def check_links(resources: list, *, timeout_seconds: float = 8.0) -> list:
    errors = []
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; JournalPulseResourceValidator/1.0; +https://example.com)"
    }
    with httpx.Client(follow_redirects=True, timeout=timeout_seconds, headers=headers) as client:
        for resource in resources:
            url = resource["url"]
            try:
                response = client.head(url)
                if response.status_code in {403, 405}:
                    response = client.get(url)
                if response.status_code >= 400:
                    errors.append(f"{resource['id']} returned HTTP {response.status_code}: {url}")
            except Exception as exc:
                errors.append(f"{resource['id']} link check failed ({exc.__class__.__name__}): {url}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the JournalPulse resource catalog.")
    parser.add_argument("--check-links", action="store_true", help="Also make online HEAD/GET requests.")
    args = parser.parse_args()

    resources = load_resource_catalog()
    errors = validate_resource_catalog(resources)
    summary = resource_catalog_summary(resources)

    print(f"resources: {summary['total_resources']}")
    print(f"crisis_safe: {summary['crisis_safe_count']}")
    print(f"coverage_gaps: {len(summary['coverage_gaps'])}")
    print(f"validation_errors: {len(errors)}")

    for error in errors:
        print(f"- {error}")

    link_errors = []
    if args.check_links:
        link_errors = check_links(resources)
        print(f"link_errors: {len(link_errors)}")
        for error in link_errors:
            print(f"- {error}")

    return 1 if errors or link_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
