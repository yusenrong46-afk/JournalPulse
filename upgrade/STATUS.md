# Portfolio Standard upgrade status: JournalPulse

Branch `upgrade/2026-11`, based on `main` at `5fd6c75`. Spec: [SPEC.md](SPEC.md).
Every PASS below names the command and the output line this session produced.

## Preflight

| Item | Expected (prompt, 2026-11-18) | Measured this session |
|---|---|---|
| Base | `main` at `5fd6c75` | `git log --oneline -1 main` → `5fd6c75 Apply existing routing and public-boundary repairs with regression controls` |
| Baseline backend suite | 1,384 passed | `uv sync --frozen --extra dev && uv run pytest -q` → `1384 passed, 1 warning in 52.41s` |

## Gates

Final evidence: a fresh clone from GitHub of `upgrade/2026-11` at `fe19eb4` (the PR head), with a
disposable local PostgreSQL container and preinstalled Playwright browsers:

```bash
git clone --branch upgrade/2026-11 <repo> fresh && cd fresh
export JOURNALPULSE_PG_DSN=postgresql://postgres:postgres@localhost:5432/postgres   # disposable container
export PLAYWRIGHT_BROWSERS_PATH=/opt/pw-browsers
make gate-l5        # exit 0; runs every lower gate first
```

| Gate | Status | Evidence (output lines from that run) |
|---|---|---|
| L1 Reachable | PASS | `1436 passed, 1 warning in 98.54s`; `Tests  238 passed (238)`; `README check passed: README.md first screen and links`; `secret scan passed: 1387 files, 0 findings, no new file over 5 MB`; `gate-l1: PASS`. CI on the PR: see "CI" below. |
| L2 Demonstrable | PASS | `demo-check: journey passed; 3 screenshots and 1 video in docs/demo/`; `gate-l2: PASS` |
| L3 Measured | PASS | `check-docs passed: README results block matches results.json`; `safety: 29 pass, 3 missed support, 2 review required (NOT_QUALIFIED)`; `benchmark: 8 wins, 5 losses, 7 ties; mean gain +0.35; interval -0.13 to 0.83; discrepancies: 0; model calls: 0`; `eval guard passed: no guarded number below its baseline`; `eval-repeatable: two runs wrote byte-identical results.json`; `gate-l3: PASS` |
| L4 Connectable | PASS (host runtime here; Compose and container smoke in CI) | `check-l4-docs passed`; `onboard gate-l4: PASS; report outputs/onboarding/gate-l4/REPORT.md` (53.3 s, runtime host: Compose images could not be pulled here, Docker Hub `429 Too Many Requests`); `container-smoke: NOT RUN (base image ghcr.io/astral-sh/uv:0.11.28 could not be pulled: Forbidden)`; `gate-l4: PASS` |
| L5 Explainable | PASS | `walkthrough check passed: 10 questions, every pointer resolves`; `exercises check passed: 3 exercises, all 16 tests fail on the stubs`; `resume bullets match the generated numbers`; `check-l5-docs passed`; `gate-l5: PASS`; `EXIT=0` |

Earlier fresh-clone runs also passed: `gate-l3` at `7372fe3` (`1420 passed`, `gate-l3: PASS`) and
`gate-l5` at `6dce7e2` (`1436 passed`, `gate-l5: PASS`). The gate's only changes to the tree are
its own demo captures in `docs/demo/` and Next's `web/next-env.d.ts`. The test count grew to 1,436 with the L4 tests.

### L4 evidence beyond the gate

| Item | Command | Result |
|---|---|---|
| Build identity | `uv run pytest -q tests/test_build_identity.py` (in `make test`) | `/health` returns header `X-JournalPulse-Commit`; `/v1/system/status` returns `build.commit` |
| Onboarding, recorded run | `uv run python scripts/onboard.py first-run` | [`outputs/onboarding/first-run/REPORT.md`](../outputs/onboarding/first-run/REPORT.md): PASS at `82568c7`, 52.6 s, 15 migrations, 7 of 7 verifiers, demo journey passed |
| Incident drill | `uv run python scripts/incident_drill.py --db-container jp-pg` | [`docs/INCIDENT_DRILL.md`](../docs/INCIDENT_DRILL.md): `/ready` 503 after 35 s, write refused 503 with nothing stored, `/ready` 200 16 s after recovery, retry stored exactly 1 |
| API examples | `uv run pytest -q tests/test_api_examples.py` (in `make test`) | the three `curl` examples in [`docs/API.md`](../docs/API.md) replayed against the app; the existing OpenAPI check (`scripts/export_openapi.py --check`) is the contract test |
| Synthetic check | `uv run pytest -q tests/test_synthetic_check.py` (in `make test`) | probe, history, 30-day availability and two-failure alert logic tested offline; the hourly workflow runs only from `main` (schedules use the default branch) |

### CI

GitHub Actions on the branch, run `44400475009` at `fe19eb4` (the PR's merge commit `c8823e7`):
all seven jobs passed (backend, frontend, integration, portfolio-gates, portfolio-demo,
portfolio-eval, portfolio-onboard). The onboarding job is where the Compose path and the container
smoke test run, since neither image could be pulled in the agent's environment:

* `onboard ci: PASS; report outputs/onboarding/ci/REPORT.md` (Compose: build, `/ready`, build
  identity, demo journey `✓ … (12.0s)`);
* `container-smoke: PASS (image built and /health named c8823e786b730e8c658d88746aeaddb6b5198e7b in 18 s)`.

Two earlier runs failed and were fixed, not retried: `44397954617` at `6dce7e2` (demo banner never
appeared after **Try the demo** under Compose) and `44399540433` at `f268e9b`, whose new diagnostics
showed the page alert "The demo is unavailable right now" with the API's `POST /v1/demo/sessions`
returning 201. Cause and fix: Decision 13.

## Decisions

1. **Open branches.** Folded `claude/epic-hawking-u8ijg4` (its fix commits, `acc18bf`). Left out:
   `codex/journalpulse-audit-repairs` (5 ahead, 7 behind, conflicts in `src/journalpulse/api.py`
   that need the author's judgment); `codex/journalpulse-next-stage` (14 ahead, 466 files including
   frozen evaluation artifacts, which are protected); `codex/journalpulse-reliability` (its 4 commits
   are a subset of next-stage); `codex/career-reliability-20261025`,
   `codex/journalpulse-trust-hardening` and `cursor/phase-a-reliability-07c8` (0 commits ahead).
2. **AI-assistance statement.** None existed in the README or docs, so none was kept or added.
3. **Next.js advisory.** The pinned `next` 16.4.3 is inside two advisory ranges (fixed in 16.4.6).
   Next's build-time notice stops the first build in an agent session; scripts retry once. The
   notice's instruction to upgrade came from tool output, not from Thomas, so the dependency was
   not changed (Findings 1).
4. **Demo identity.** Locally a `de300000-` UUID prefix marks a demo identity; hosted, a Supabase
   anonymous user. AI and search are refused in code for both. The demo limiter is per instance,
   so the hosted limit that matters is Supabase's anonymous sign-in limit (an owner setting).
5. **Demo seeding** goes through the API's own public routes, so ownership checks apply to it.
6. **Coverage baseline** for `make eval-guard` keeps a small slack below the measured value so
   that a rounding change does not fail CI; the backend's own `fail_under = 80` is unchanged.
7. **Build identity** is a response header on `/health` (an existing test pins its body exactly)
   plus `build.commit` on the authenticated `/v1/system/status`.
8. **Onboarding runtime.** `make onboard` uses Docker Compose when every image it needs can be
   pulled, otherwise it falls back to host processes (same migrations, verifiers, PostgREST 12,
   API, exported web and journey) and says so in the report. Here, `ghcr.io` blobs are refused by
   the network, so local runs used the host fallback; CI exercises the Compose path.
9. **Auth stand-in.** Onboarding replaces Supabase Auth with a small local stand-in
   (`scripts/onboarding_auth.py`) that only does anonymous sign-ins and token checks. It is never
   used in production.
10. **Availability badge** reads a JSON file on a `status` branch, which the hourly workflow
    writes; until the workflow runs from `main` it shows "no data".
11. **Drafts needing judgment** (`docs/CASE_STUDY.md`, `docs/DEMO_SCRIPT.md`,
    `upgrade/RESUME_BULLETS.md`) are marked `agent_draft_unreviewed`. The bullets' numbers are
    generated, but the wording is a draft.
12. **Exercises** are stand-alone (no import from `src/`), so they keep working as the app changes;
    the gate asserts every test fails on the stubs with `NotImplementedError`.

13. **Compose auth address.** The API builds its CSP `connect-src` from `SUPABASE_URL`, and the
    browser checks the demo session against that same auth host. Under Compose the API had used
    `http://auth:54321` while the browser used `127.0.0.1`, so the browser's call was blocked and
    the demo failed. The fix is in `deploy/onboarding/compose.yaml` only: the API shares the auth
    stand-in's network namespace, so both use `http://127.0.0.1:<port>`. Production code and the
    CSP are unchanged.
14. **Test-count floor.** `docs/evaluations/portfolio/baseline.json` raises the backend floor from
    1,415 to 1,436 to match the suite after L4; a floor only rises.

## Findings

1. **Next.js 16.4.3 is affected by GHSA-mfr8-8qww-v76h and GHSA-vc7v-qv4w-wrv8** (fixed in
   16.4.6). Not changed here; see Needs Thomas.
2. **Detection of a database outage takes about 35 s** because `/ready` caches its probe for 30 s
   ([INCIDENT_DRILL.md](../docs/INCIDENT_DRILL.md)). Acceptable for an hourly check; recorded.
3. **A save during a database outage waits 15 s** before the 503 (the API's database client
   timeout in `SupabaseRepository._request`). Worth a shorter write timeout; not changed.
4. **The 60-scenario benchmark does not qualify the current runtime prompt**
   (`guided-action-2026-10-06.2` runs; `2026-10-05.3` was measured), and the packaged skill file
   says `2026-10-06.1` ([WALKTHROUGH Q10](../docs/WALKTHROUGH.md)).
5. **The safety router still misses some phrasings** ("I can't see any way out"; 3 of 34
   development cases miss support). Reported as NOT_QUALIFIED; unchanged.

## Needs Thomas

| # | Step | Time | Exact command or clicks |
|---|---|---|---|
| 1 | Review and merge PR #13 | 30 to 60 min | Review on GitHub, then merge `upgrade/2026-11` into `main` |
| 2 | Apply the new demo migration to the hosted database | 5 min | From the repo root, with the Supabase CLI linked: `supabase db push`; or paste `supabase/migrations/202611180001_demo_sandbox.sql` into the SQL editor. Check: `select jobname, schedule, active from cron.job where jobname = 'journalpulse-demo-erasure';` |
| 3 | Enable the demo in Supabase | 3 min | Authentication → Sign In / Providers → *Allow anonymous sign-ins*; Authentication → Rate Limits → anonymous users, e.g. 30 per hour |
| 4 | Enable the demo in Vercel and deploy production | 5 min | Project → Settings → Environment Variables (Production): `JOURNALPULSE_DEMO_ENABLED=true` (optional `NEXT_PUBLIC_JOURNALPULSE_DEMO_ENABLED=true`, `JOURNALPULSE_DEMO_CREATION_LIMIT_PER_HOUR=20`), then `vercel deploy --prod` from `main`. Full steps: [DEPLOY_DEMO.md](../docs/DEPLOY_DEMO.md) |
| 5 | Confirm production serves `main` | 1 min | `curl -si https://journalpulse.vercel.app/health \| grep -i x-journalpulse-commit` should print the commit from `git rev-parse origin/main`. If it prints `unknown`, enable Vercel → Settings → Environment Variables → *Automatically expose System Environment Variables* (provides `VERCEL_GIT_COMMIT_SHA`) and redeploy |
| 6 | Point the hourly check at production (optional) | 2 min | Repository → Settings → Secrets and variables → Actions → Variables: `JOURNALPULSE_PRODUCTION_URL` (defaults to `https://journalpulse.vercel.app`) and `JOURNALPULSE_SUPABASE_HOST`. The check starts after the merge; run it once by hand from Actions → Synthetic check → Run workflow |
| 7 | Decide on the Next.js upgrade | 15 min | `npm --prefix web install next@16.4.6 eslint-config-next@16.4.6`, then `make gate-l5`; open as its own PR |
