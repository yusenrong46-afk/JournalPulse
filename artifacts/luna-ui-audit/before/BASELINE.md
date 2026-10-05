# Baseline before the UI audit changes

- Commit: `e0bceab646aa3a8dd6fada161c34e7c965dfa470` (`main`, clean tree), 2026-10-05.
- Skill: `guided-action-2026-10-05.3` (unchanged).
- Toolchain: Python 3.12.13 with uv 0.12.23, Node 22.23.3 / npm 11 (Homebrew `node@22`),
  PostgreSQL 16 from `pgvector/pgvector:pg16`
  (`sha256:7b822b0aac60967beb1ea5e576b8602c94c300a157d187f385ae3e0da199b90a`) in a disposable
  local Docker container on `127.0.0.1:55432`, and the `psql` 18.6 client.
- Credentials: none were used. Every check uses local stand-ins.

| Check | Result | Log |
|---|---|---|
| `uv sync --frozen --extra dev` | pass | `uv-sync.log` |
| `ruff check src tests scripts` | pass | `ruff.log` |
| `mypy src` | pass | `mypy.log` |
| `export_openapi.py --check` | pass | `openapi.log` |
| `pytest --cov=journalpulse` | 703 passed, 91% coverage | `pytest.log`, `pytest.xml` |
| `validate_resources.py` | pass | `resources.log` |
| Four PostgreSQL verifiers (local) | all pass (activity: 72 assertions) | `pg-*.log` |
| `npm ci`, lint, typecheck | pass | `npm-ci.log`, `lint.log`, `typecheck.log` |
| `npm run test:unit` | 108 passed | `unit.log` |
| `generate:api` with no diff, `build` | pass | `generate-api.log`, `build.log` |
| `npm run test:e2e` | 44/46 on the first run; the 2 failures passed 3/3 on rerun | `e2e.log` |
| `npm run test:integration` | 18/18 (macOS run, after the PostgREST pin below) | `integration.log` |

## Flaky browser checks (timing, not product behaviour)

1. `accessibility.spec.ts` on mobile `/`: axe measured `.eyebrow` at #8778a0 on #f1ecfa (3.46:1)
   while the card was still fading in. The settled colour passes. The test needs to wait for
   animations, or emulate reduced motion.
2. `core-flow.spec.ts` "mood tap to one saved small step": `turns.at(-1)` is read right after a
   click, before the stubbed request is recorded. It needs `expect.poll`.

## Integration stack on macOS

The original `scripts/integration_stack.py` always downloaded the Linux x86-64 PostgREST build
without verifying it, so the integration suite could not run on this Mac. The F5 fix, which pins
SHA-256 hashes, checks versions and adds a macOS arm64 asset, was applied before this run. The
UI under test is unchanged.

## Screens (`screens/`, captured with `web/tests/integration/ui-tour.spec.ts`)

Luna, Brave and auth are deterministic stand-ins. Observations from the images:

- Running (mobile): the activity card's instructions and controls fill the viewport, and the
  conversation is scrolled out of view. The header takes about 17% of the screen height.
- The offer (desktop) shows four separate entry points to alternatives or search ("Just talk",
  "Find resources", "Search for other resources", "Find another resource") around one card.
- The check-in shows four participation radios, three optional selects and a free-text box
  before Save. "Find another resource" stays visible under the form.
- There are separate "Stop activity" and "Finish early" buttons, plus a "Stopped" check-in choice.
