"""Start the frozen, capped evaluation; the child prompts privately for its API key."""
import os
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
bundle = root / "artifacts/reliability-2026-10-14/quality-run-v2"
runner = bundle / "frozen/scripts/quality_eval.py"
if not runner.is_file():
    raise SystemExit("The reviewed frozen evaluation bundle is missing.")
os.execv(sys.executable, [sys.executable, str(runner), "run", "--bundle", str(bundle),
                         "--ledger", str(root / "artifacts/reliability-2026-10-14/quality-budget.json")])
