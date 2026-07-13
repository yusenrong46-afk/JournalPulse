from __future__ import annotations

import argparse
import json

from journalpulse.api import app
from journalpulse.config import PROJECT_ROOT

OUTPUT = PROJECT_ROOT / "web" / "openapi.json"


def rendered_schema() -> str:
    return json.dumps(app.openapi(), indent=2, sort_keys=True) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Export the checked frontend API contract.")
    parser.add_argument("--check", action="store_true", help="Fail when the checked contract is stale.")
    args = parser.parse_args()
    rendered = rendered_schema()
    if args.check:
        if not OUTPUT.exists() or OUTPUT.read_text(encoding="utf-8") != rendered:
            raise SystemExit("web/openapi.json is stale; run scripts/export_openapi.py")
        print("OpenAPI contract is current")
        return
    OUTPUT.write_text(rendered, encoding="utf-8")
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
