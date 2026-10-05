# Claude handoff: audit, repair, then improve the Luna UI

Use the prompt below from the root of `yusenrong46-afk/JournalPulse`. Repository
paths are relative so the handoff also works in a fresh checkout. The latest
preview release is described in `docs/LUNA_PREVIEW_RELEASE_2026-10-05.md`.

```text
Audit, debug, and improve JournalPulse, then redesign the UI around its working
Luna chat + optional activity + honest feedback loop. Implement the work; do not
stop after proposing a plan. This is a serious AI/NLP engineering project. Keep
the code understandable, preserve evidence, and work in small vertical slices.

Read AGENTS.md files and web/CLAUDE.md first, then README.md, docs/ARCHITECTURE.md,
docs/LUNA_GUIDED_ACTION_IMPLEMENTATION.md, docs/LUNA_PREVIEW_RELEASE_2026-10-05.md,
docs/LUNA_FULL_RELEASE_BENCHMARK_2026-10-05.md, and
docs/PRE_EVALUATION_AUDIT_2026-10-05.md. Historical documents describe earlier
versions: verify their dates and actual code instead of assuming every old blocker
is still open. docs/NEXT_UPGRADE.md is a backlog, not authorization to implement
every future capability in this task.

Current state:
- Python 3.12 / FastAPI API, Next.js 16 / React frontend with Node 22, Supabase
  authentication and PostgreSQL, GPT-6 Luna via OpenRouter, and Brave Search.
- The runtime skill is src/journalpulse/skills/guided_action/SKILL.md, version
  guided-action-2026-10-05.3. It is packaged and loaded by application code.
- Luna understands the user's concern, may propose one voluntary fitting activity,
  accepts corrections/negotiation, and stays available through the activity.
- The app controls start/pause/resume/finish/stop and server-owned timer deadlines.
  Expiry presents a participation check-in. Saving the report precedes one bounded
  Luna follow-up. Timer expiry never proves participation or improvement.
- Journal chat uses only an explicitly linked owner entry. Automatic journal lookup
  by date and custom NLP training are future work; do not silently add them.
- Inline Brave search is consented and uses server-constructed public-topic queries.
  Chat/journal text must not be forwarded. Results are snippet evidence, not fully
  reviewed pages. Saving and starting a resource are separate actions.
- The existing preview is
  https://journalpulse-preview-yusenrong46-9212s-projects.vercel.app
  and deployment dpl_9xFz5u928LMBgsiJuFQYecw7gV35 contains the evaluated .3 runtime.
  Production still uses dpl_57Rut4UoR24LsTRjsA7nEpiBQycU.
- The existing shared Supabase project already has all ten current migrations,
  including 202610050001_guided_activity_sessions.sql. It uses manually applied
  history; do not replay migrations, invent historical entries, reset it, or use
  local scratch helpers against it. Add a new migration only when necessary.
- The user tried a real two-minute meditation and is satisfied with the improvement.
  Detailed report comments were not saved; do not invent human annotations.

Evaluation status:
- The completed benchmark contains 60 fictional cases, eight bounded longer pairs,
  and six paired repeat diagnostics. Both versions use Luna; GPT-6.1 Sol judged them.
  This was prompt/application engineering, not weight training or distillation.
- Final-test quality gains passed the declared engineering thresholds: +0.35/5,
  eight updated wins, five earlier wins and seven ties. The uncertainty interval
  includes zero. This is not independent human or clinical validation.
- Formal release remains incomplete because s05_physical_warning has an uncertain
  safety grade. For new chest pain/faintness, both responses reject exercise and
  request immediate medical help, but the candidate is less explicit about emergency
  help and not driving. Review the actual wording and rubric separately; do not
  alter historical grades or claim all safety gates passed. Preview testing was
  explicitly authorized with that limitation recorded.
- Raw paired evidence, evaluator repairs and immutable original grades are under
  artifacts/luna-release-benchmark/. The source and manifests distinguish .2/.3.
  Cases already inspected or used for tuning cannot become fresh final tests.
- The earlier broad audit passed 700 backend tests, 108 frontend tests, 46 browser
  checks, 18 full-stack checks and 208 PostgreSQL assertions. These are historical
  evidence, not a substitute for verifying your own changed version.

Work in this order:

1. Establish and audit the real starting point.
   Record the commit, dirty files, versions and locally verified checks. Map the
   chat, journals, search, timers, reports, persistence, auth and evaluation paths.
   Inspect actual desktop/mobile screens and capture screenshots before redesigning.
   Reproduce suspected bugs and record severity, trigger, cause and evidence.
   Investigate correctness, owner boundaries, privacy/retention/deletion, prompt
   injection, model failure atomicity, concurrency, stale replies/offers, duplicate
   submissions, report recovery, accessible controls, dependencies and deployment.
   If parallel agents are available, they may perform independent scoped reviews;
   integrate and independently verify their findings before making changes.

2. Repair confirmed bugs before redesigning.
   Keep fixes small, add regression tests that reproduce meaningful failures, and
   show the relevant code diff and before/after behavior for each completed slice.
   Verify already-fixed complaints rather than reimplementing them: user messages
   must appear before Luna's new reply; stale action buttons must disappear after
   listening/stopping; new offers must replace obsolete unstarted offers.
   Known candidates for investigation include:
   - indirect-risk misses and quoted/historical-risk false positives in the limited
     English safety phrase router;
   - request retries/timeouts that can exceed Vercel's 120-second function limit;
   - inline outcome history not appearing in the legacy Home/Journey garden;
   - PostgreSQL downloader integrity/version verification and packaging reproducibility;
   - the development-tool braces advisory: distinguish it from production exposure,
     and recheck current advisories instead of blindly upgrading the entire lockfile.
   These are candidates, not permission to present unverified claims as confirmed bugs.

3. Redesign the experience in small, complete UI slices.
   First define a coherent interaction hierarchy from screenshots and actual use:
   chat is central, one optional proposal is easy to understand, an accepted activity
   has compact controls, and its report/follow-up reconnect naturally to the chat.
   Prioritize:
   - readable message order, useful scrolling, typing/loading/error states and an
     always-reachable composer, including mobile keyboards and long conversations;
   - a compact activity/timer presentation that does not bury conversation, clear
     active/paused/expired/reported states, and one check-in without duplicate controls;
   - clear optional alternatives and search refinement within the activity flow,
     with explicit consent, saving and starting;
   - a simple journal-to-chat handoff with visible source/date context and clear
     distinction between an old entry and the user's present state;
   - fewer redundant emotion/goal prompts, clearer labels, accessible focus, contrast,
     touch targets, and responsive empty/loading/error/success states.
   Keep Luna warm and concise. Preserve the currently successful behavior. Do not
   force an activity, positive rating, diagnosis or gamified wellbeing outcome.
   Build and verify each end-to-end slice before moving to the next. Avoid a large
   rewrite, unnecessary dependencies, or abstractions without a concrete benefit.
   Comment the non-obvious reasons and boundaries; do not narrate every code line.

4. Verify and deliver a reviewable result.
   Use the repository's supported setup and test scripts with Python 3.12 and Node 22.
   Existing core commands include:
     uv sync --frozen --extra dev
     uv run ruff check src tests scripts
     uv run mypy src
     uv run python scripts/export_openapi.py --check
     uv run pytest --cov=journalpulse
     uv run python scripts/validate_resources.py
     cd web && npm ci
     npm run lint
     npm run typecheck
     npm run test:unit
     npm run generate:api
     npm run build
     npm run test:e2e
     npm run test:integration
   Consult .github/workflows/ci.yml for the local PostgreSQL prerequisites and all
   four migration verifier scripts. Scratch verification is destructive and must
   use a dedicated LOCAL database, never the shared Supabase project. Mark model,
   auth or search stand-ins explicitly; they do not prove live model quality.
   Generate a concise before/after report with findings, code diffs, desktop/mobile
   screenshots, tests executed, unrun checks and remaining risks. Keep a small
   manual evaluation checklist. Do not inflate coverage into a safety/quality claim.

Boundaries:
- Preserve existing work and frozen reports; never overwrite old failed/uncertain
  evidence or mislabel a new response as an old observation.
- Preserve the .3 skill initially. If a confirmed defect requires prompt changes,
  version them and prepare bounded before/after evaluation with fresh final cases.
  Do not start paid model/search evaluation without a new explicit budget. The
  previous $7 authorization was for the completed benchmark/deployment checks.
- Use available credentials securely; inspect names/presence, never print values,
  publish keys, send emails, or read real private journals for testing. Prefer
  local services and disposable fictional accounts, then clean them up.
- Do not deploy to production, change shared project secrets, or destructively
  migrate hosted data. Prepare changes for review; update the existing preview
  only when the user asks for deployment and the necessary checks pass.
- Keep Git commits focused, explain what changed/why/how verified, and preserve
  API types, database compatibility and retention guarantees through UI changes.

Finish with a simple-language explanation of what was fixed, what the UI improved,
what I should test, and what remains open. Complete useful independent work before
asking for any genuinely missing requirement; do not stop at a plan.
```
