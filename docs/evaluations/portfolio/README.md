# Portfolio evaluation (L3)

`make eval` runs `scripts/portfolio_eval.py`. It makes **no model, search or network
calls** and writes:

* `results.json`: deterministic results (committed). Two runs on one commit are
  byte-identical; `make eval-repeatable` checks this with `cmp`.
* `timings.json`: wall-clock seconds per part (a sidecar, not committed).

| Part | What it computes | Source |
|---|---|---|
| (a) Safety routing | Pass, missed support, false alarm and review-required counts on the 34-case development set. The set keeps its `NOT_QUALIFIED` label: it was used during development, so it is not held-out evidence. | `scripts/evaluate_safety.py`, `assets/evaluation/safety/development.json` |
| (b) Release benchmark | Held-out preferences, mean primary gain and its approximate 95% interval, recomputed from the stored teacher judgments with the same eligibility and interval rules as `journalpulse.guided_evaluation`. Compared with the documented figures and the stored summary; any difference is listed under `discrepancies`. Artifacts are read, never edited. | `artifacts/luna-release-benchmark/comparison.json` |
| (c) Tests | Backend tests passed and branch-aware coverage (pytest), frontend unit tests passed (vitest). | `tests/`, `web/tests/unit/` |

`make eval-guard` (CI job `portfolio-eval`) fails when a guarded number is worse than
[`baseline.json`](baseline.json). `make check-docs` fails when the README's generated
results block or first-screen headline differs from `results.json`; `make eval-readme`
rewrites both.

## Real-model evaluations: documented, NOT RUN

These make paid provider calls. None was run for the Portfolio Standard upgrade
(authorized model spend for this work: US$0). Run them only with a separately
authorized, finite budget.

| Evaluation | Command | Budget | Status |
|---|---|---|---|
| Live structured reflection smoke test (one call) | `uv run python scripts/verify_openrouter.py` | One request to `openai/gpt-6-luna`; the script prints model and latency, not cost | NOT RUN |
| Two-turn Luna conversation smoke test | `uv run python scripts/verify_conversation.py` | Two turns; the script's price table is US$0.10 / US$0.50 per million input / output tokens (OpenRouter list price recorded 2026-09-24) | NOT RUN |
| Guided-action release benchmark (60 scenarios, teacher grading, one live search) | Preparation and rendering are offline: `uv run python scripts/evaluate_guided_action.py --help`. The live dispatch of 2026-10-05 is recorded in `artifacts/luna-release-benchmark/run-budget.json` (215 attempts, about US$2.22 accounted, within a US$7 authorization). | A new run needs a new authorization; the 2026-10-05 plan capped Luna, teacher and search attempts in `budget-authorization.json` | NOT RUN (2026-10-05 results are recomputed offline above) |
| Luna conversation cases | `uv run python scripts/evaluate_luna.py --help` prepares and reviews cases offline; collecting live replies needs the provider and a budget | Depends on case count | NOT RUN |
