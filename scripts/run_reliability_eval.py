"""Start the frozen, capped evaluation; the child prompts privately for its API key."""
import json
import os
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
bundle = root / "artifacts/r1-follow-up-2026-10-25/quality-run-v5"
runner = bundle / "frozen/scripts/quality_eval.py"
if not runner.is_file():
    raise SystemExit("The reviewed frozen evaluation bundle is missing.")
manifest = json.loads((bundle / "manifest.json").read_text())
original = json.loads((root / "artifacts/reliability-2026-10-14/quality-run-v2/manifest.json").read_text())
if manifest["ledger_path"] != original["ledger_path"]:
    raise SystemExit("The authoritative shared ledger binding changed; stop for reconciliation.")
ledger = Path(original["ledger_path"])
if not ledger.is_file():
    raise SystemExit("The authoritative shared USD10 ledger is missing; no paid call is permitted.")
os.execv(sys.executable, [sys.executable, str(runner), "run", "--bundle", str(bundle),
                         "--ledger", str(ledger)])
