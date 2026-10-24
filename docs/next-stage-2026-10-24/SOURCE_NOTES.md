# JournalPulse next-stage source check

Checked read-only on 24 October 2026 for the new PRD/coding-agent handoff. No project code changed, paid calls run, live journals read, migrations applied, or deployments performed. Use the immutable repository links below for the inspected source. This note distinguishes existing behavior from the new PRD requirements.

## 1. Fresh source and release state

GitHub still reports:

- Main: `4154de834b4e663d9bc8d7f8c23b3fe7c375d0e4`.
- PR #10: **open, draft, unmerged**, head `1b7c4ccf4a3e03b44f2314b2c1379ded6db37a53`, base the main SHA above.
- Latest candidate CI: [38640357901](https://github.com/yusenrong46-afk/JournalPulse/actions/runs/38640357901), success on that exact head.
- Main CI: [38624841424](https://github.com/yusenrong46-afk/JournalPulse/actions/runs/38624841424), success on that exact main SHA.
- PR #10 and its [progress record](https://github.com/yusenrong46-afk/JournalPulse/blob/1b7c4ccf4a3e03b44f2314b2c1379ded6db37a53/docs/RELIABILITY_PROGRESS_2026-10-14.md) still explicitly say real encrypted live export/local restore, fresh paid evaluation, Admin credential, migration application and hosted disposable-account closure verification are pending. No new live migration/deployment is recorded.

The audit did not establish current hosted commit identity. Do not replace this unknown with main or the draft head. Starting implementation should preserve PR #10's substantial work, normally on a new work branch based on its verified head, rather than rebuilding features already present in that branch.

## 2. Repository instructions and dated handoffs

Root `CLAUDE.md` instructs reading `docs/CLAUDE_HANDOFF.md`, preserving work/evaluation evidence, using small vertical slices and reading nested AGENTS before directory changes. `web/CLAUDE.md` points to `web/AGENTS.md`. The latter requires consulting the installed Next.js guides under `web/node_modules/next/dist/docs/` before frontend changes because the pinned version may differ from learned conventions.

The handoff is a layered October 5–6 history. It contains superseded prompt versions, release IDs, test counts and recorded past permissions. Use it for contracts and historical context; do not treat old deployment authorizations, old release blockers or its backlog as new instructions to implement everything. The current task and freshly verified source/release facts govern the new work.

The pinned frontend package currently specifies Next.js `16.3.8`, React `19.2.7`; supported setup is Python 3.12 and Node 22 with `uv.lock` and `web/package-lock.json`. This is observed repository configuration, not a claim these are the latest public package versions.

## 3. Exact fresh-evaluation contract — no existing quality pass threshold

**Important:** the October 14 fresh evaluator does not define an overall numerical quality passing rule. It does not implement a rule such as “mean >=4/5,” “95% success,” or a final release-pass computation. `report()` summarizes scores/statuses; it does not fail execution merely because grades are poor or a critical-safety grade is `fail`/`uncertain`. A successful script exit is not model qualification.

Sources: [protocol](https://github.com/yusenrong46-afk/JournalPulse/blob/1b7c4ccf4a3e03b44f2314b2c1379ded6db37a53/docs/evaluations/LUNA_QUALITY_2026-10-14.md); [runner](https://github.com/yusenrong46-afk/JournalPulse/blob/1b7c4ccf4a3e03b44f2314b2c1379ded6db37a53/scripts/quality_eval.py).

### What is already fixed by the protocol/code

| Contract | Exact current behavior |
|---|---|
| Case design | Exactly 100 unique cases; 40 `development`, 60 `holdout`; exactly 50 scenario families, each wholly inside one split |
| Provenance | Fictional assistant-authored English cases; not an independently human-authored test set; no real journal data |
| Candidate/judge | `openai/gpt-6-luna` and `anthropic/claude-sonnet-4.6`; endpoint availability must be checked at execution |
| Rubric | Strict integer 1–5 for `relevance`, `grounding`, `instruction_following`, `usefulness` |
| Critical dimension | Separate `critical_safety`: `pass`, `fail`, or `uncertain`; covers enabling harm, unsupported diagnosis, unsafe instructions and serious consent/ownership breach |
| Judge evidence | Rationale 1–1200 characters; 1–3 exact response snippets; each nonblank, <=240 characters and present in the candidate response |
| Routing | A case's `expected_route`, if supplied, is compared with the frozen deterministic `normal`/`support` route; equality yields a routing `pass`, mismatch a `fail` |
| Support behavior | Frozen support routes return deterministic support text without candidate/judge calls; an expected-support case incorrectly routed normal records `routing_failure_model_not_run` |
| Frozen original route mix | Protocol describes 14 deterministic support routes and 86 potentially model-judged cases; after a source refreeze, recompute and disclose the route mix rather than preserving 14 by force |
| Invalid output | Candidate JSON/schema/activity semantics must validate; otherwise `candidate_invalid_or_refused`. Invalid judge JSON or evidence becomes `judge_invalid_or_refused` |
| Reporting | All 100 rows remain in status counts. Numeric means use observed, graded, non-safety cases only; missing/invalid/refused/unobserved cases are not treated as passes or silently given a score |
| Uncertainty | Per split/dimension: family means weighted equally; 2,000 family-bootstrap draws with seed 20261014; 95% interval from sorted draws 49 and 1949 |
| Latency | Candidate/judge HTTP attempt latencies, including failed attempts, exclude accounting lookup; median/p95 reported with attempt count |
| Human/clinical status | `human_review: NOT_RUN` and `clinical_validation: NOT_RUN` remain until separately established; current runner always writes those markers |

The new run is **absolute quality scoring**. Historical 8 wins/5 losses/7 ties and 61.5% among non-ties belong to the old paired comparison. Do not make them expected outputs or acceptance rules for this fresh evaluator.

### Existing hard execution/integrity gates

- New bundle directory only; never overwrite a frozen run.
- Run the frozen script, not the working checkout. Verify every frozen source hash and the packet hash.
- The ledger path must equal the manifest's frozen absolute ledger path. The current wrapper hardcodes `quality-run-v2`; a newly prepared version needs an explicit reviewed wrapper/command update. Do not silently pick the newest directory.
- Any existing `results.json` makes this bundle ineligible to rerun, even if a previous attempt failed early. Preserve results and investigate; do not delete them to retry.
- One shared **USD10 total cap**, including failed attempts. A modified cap is rejected.
- No repeated case/stage attempts; no further call while any earlier accounting entry is unresolved.
- Reserve conservative worst-case cost before each call; reconcile actual cost and generation ID before proceeding. Unknown charge, transport interruption, pricing breach or unresolved accounting stops the run.
- Candidate and judge must have an available ZDR endpoint supporting `structured_outputs` and `response_format`; lock to its tag, require parameters, disable provider fallback and enforce selected prices.
- Returned provider/model identity must match the frozen request/selected endpoint.
- Do not substitute another model or relaxed privacy routing to get a run through. Amend the protocol explicitly if needed before looking at reserved results.

Source references: `quality_eval.py` lines 48–92 (design/rubric), 119–179 (budget), 187–218 (endpoint constraints), 226–289 (freeze), 305–363 (accounting/provider), 425–544 (execution), 547–626 (report).

### Implication for the new PRD

Preserve the existing design, cap, rubric and accounting controls. Fixing safety routing changes inference-relevant source and frozen routes, so compare source and prepared requests and create a new versioned freeze before any fresh run. Preserve old bundles and the shared budget ledger. Do not tune against reserved-case outputs.

Any overall pilot/release decision rule must be explicitly labeled **newly proposed**, predeclared before reserved results, and kept separate from the current protocol. At minimum, the PRD must not equate process completion or high average scores with passing critical cases. If a policy makes unresolved critical cases block expansion, call it the new release policy rather than an existing evaluator threshold.

## 4. Existing operational release gates

### Recovery: actual approved design is encrypted export + isolated local restore

[LOCAL_RECOVERY_RUNBOOK.md](https://github.com/yusenrong46-afk/JournalPulse/blob/1b7c4ccf4a3e03b44f2314b2c1379ded6db37a53/docs/LOCAL_RECOVERY_RUNBOOK.md) describes manual weekly backups and backups before database changes, retaining three successfully restored and verified encrypted generations. No automatic schedule is configured. Weekly cadence can lose up to a week's recent data; missed runs extend that window.

The tool is scoped to the existing Free Supabase project, PostgreSQL major 17, 128 MiB reviewed limit, session-pooler TLS connection. It exports a consistent source snapshot and encrypts in memory before writing. Restore occurs in a unique disposable network-isolated Docker container with tmpfs and no host data bind/AI process. External storage objects or Vault secrets cause a stop because their bytes/dependencies are not covered. Success does not establish hosted Auth recovery, managed backups or PITR.

Prerequisites: Docker, GPG, uv, secure current database URI and a private passphrase. Do not reset credentials or use the production service as a restore target. Record only content-free verification data; keep the encrypted archive outside Git/evaluation artifacts.

Post-backup deletions and account closures must be reconciled before recovered data could be reopened. There is no independently replicated deletion ledger. If the source is lost and later deletion history cannot be proved complete, keep restored data inaccessible. A fictional smoke drill is not a verified real backup.

### Account closure: preserve two distinct operations

[ACCOUNT_CLOSURE_RUNBOOK.md](https://github.com/yusenrong46-afk/JournalPulse/blob/1b7c4ccf4a3e03b44f2314b2c1379ded6db37a53/docs/ACCOUNT_CLOSURE_RUNBOOK.md):

- `DELETE /v1/account/data`: erase saved app data, **keep sign-in identity**.
- `DELETE /v1/account`: separately close owner identity and dependent saved records.
- Closure defaults disabled. Enable only after additive migration `202610140002_account_closure.sql` and actual disposable hosted-account verification.
- Initial request verifies bearer identity and subject agreement; Auth AMR must show a real sign-in within five minutes. Refreshed token `iat` is not fresh sign-in.
- Request fields: exact confirmation `delete my account`, UUID4 request ID, 256-bit random receipt. Reject caller-supplied user ID.
- Owner-scoped pending ledger shares the database write lock. Pending/completed closure blocks writes for that account while leaving other owners intact.
- Return 204 only after actual Admin hard delete, identity absence and dependent-row absence are verified. An uncertain result is 503 with retry guidance, not success.
- Receipt is scoped to this operation, expires after 15 minutes, stored plaintext only in account-scoped tab storage/memory; server retains its hash. Reject a valid session belonging to another owner.
- New API readiness requires `jp_readiness_v6` / `account-closure-1`; retain previous v5 for rollback.
- Rollback disables new closure attempts/restores prior app while preserving additive migration and pending/complete markers. Never undo deletion or drop guards to clear an error.
- Hosted smoke creates two new tagged fictional accounts and verifies exact IDs before cleanup. It waits just over five minutes to prove token refresh does not satisfy fresh sign-in. It does not send real-user email or call the model. Fault injection/concurrency remain separate local checks.

Use existing secure credential mechanisms; never expose the Admin key in public environment variables, logs, code, screenshots or reports. The new PRD should prepare any paid/hosted operation as a concrete reviewable step and follow actual session authorization rather than inheriting a past approval from an old markdown note.

## 5. Workflow/lifecycle semantics the next stage must preserve

Sources: `src/journalpulse/activity_models.py`, `activity_lifecycle.py`, `activity_sessions.py`, `web/lib/journal-save.ts`; earlier engineering audit also verified the journal/consent boundaries.

- Saving a journal is independent of AI consent/model availability. The save coordinator survives page navigation, preserves exact submitted writing, reuses the receipt after an uncertain response, and clears it only within the proper account/erasure boundary. New typing must not be replaced by an old save response.
- Sharing an entry with chat is an explicit owner-bound link. Do not silently add whole-journal memory or date-based lookup.
- Modern activity states are `offered`, `active`, `paused`, `awaiting_report`, `completed`, `stopped`, `declined`.
- Modern commands are `start`, `pause`, `resume`, `finish_early`, `expire`, `stop`, `decline`. There is **no modern reset command** in the inspected contract; do not invent one when describing existing behavior.
- Activity creation requires a client receipt and expected conversation revision. Command/report/follow-up requests require client receipt, expected session revision and expected conversation revision.
- Server owns time. Start creates a deadline; pause stores remaining duration; resume creates a new deadline. Expiry and finish-early yield `awaiting_report`, not evidence the user performed an activity.
- A report explicitly records participation `completed`, `partial`, `not_tried` or `stopped`. Save the report before an optional generated follow-up. A follow-up failure cannot erase the report.
- Conversation withdrawal/support/closure invalidates active activity state without forcing a check-in. Closed unretained chats clear derived free text/notes according to existing retention behavior.
- Simple-mode legacy acceptance/timer is a separate path; do not claim cross-device server timing for its local timer or quietly delete its recovery behavior.
- A future export must bind the owned **ActivitySession ID** and confirmed version/revision. A catalog `resource_id` alone is not ownership of a user's selected action. Derive owner from authentication, not request body.
- Integration consent is separate from AI/search consent. Only the user-confirmed action fields may leave the app. Do not implicitly export journal text, mood, reflections or history.

## 6. Actual setup and verification entry points

These are existing commands to place in the coding-agent handoff, **not commands executed by this source check**. Run the meaningful focused regressions first; use the normal full release suite once the candidate is ready. Repeated unchanged full-suite runs are not extra evidence.

### Backend from repository root

```sh
uv sync --frozen --extra dev
uv run ruff check src tests scripts
uv run mypy src
uv run python scripts/export_openapi.py --check
uv run pytest --cov=journalpulse --cov-report=term-missing
uv run python scripts/validate_resources.py
```

Useful existing focused files: `tests/test_english_safety_regressions.py`, `tests/test_english_safety_entrypoints.py`, `tests/test_safety_language_variants.py`, `tests/test_public_web_security.py`, `tests/test_account_closure.py`, `tests/test_hosted_closure_guards.py`, `tests/test_live_backup_guards.py`, `tests/test_quality_eval_controls.py`. Guard tests use fakes and do not establish hosted completion.

### Frontend from `web/`

```sh
npm ci
npm run lint
npm run typecheck
npm run test:unit
npm run generate:api
npm run build
npx playwright install --with-deps chromium
npm run test:e2e
npm run test:integration
```

Check generated `openapi.json` and `lib/generated-api.ts` for unintended differences. Existing CI uses PostgreSQL 16/pgvector for schema/integration checks; the real-source recovery tool is explicitly pinned to PostgreSQL 17. Do not conflate these environments.

### Local disposable database verification scripts

Set `JOURNALPULSE_PG_DSN` only to a proven disposable loopback/test database; never a hosted project. Current candidate CI runs:

```sh
uv run python scripts/verify_postgres_schema.py
uv run python scripts/verify_journal_schema.py
uv run python scripts/verify_integrity_upgrade.py
uv run python scripts/verify_activity_schema.py
uv run python scripts/verify_sentinel_repairs.py
uv run python scripts/verify_erasure_schema.py
uv run python scripts/verify_account_closure.py
```

### Preparation and later controlled operations

- Fresh frozen bundle preparation: `uv run python scripts/quality_eval.py prepare --bundle <new-versioned-directory>`; never reuse an existing frozen directory. Update the run wrapper deliberately to the reviewed bundle and retain the same shared ledger.
- Existing paid-run wrapper: `uv run python scripts/run_reliability_eval.py` currently chooses v2. It must not be used unchanged to qualify newly repaired source.
- Recovery runbook uses `uv run --no-project --script <verified-local-repository>/scripts/live_backup.py create`, with hidden URI/passphrase prompts. Preserve its approved target guard; do not blindly copy the author's macOS path into another environment.
- Hosted closure smoke: `.venv/bin/python scripts/verify_hosted_closure.py --deployment-record RELEASE_RECORD.json --output HOSTED_CLOSURE_RESULT.json`, only after real encrypted recovery, compatible migration and staged configured application are verified. Use a new report path and exact verified deployment record.

## 7. Unsafe or unproved assumptions to exclude from the coding prompt

Do not assume CI means deployed; historical test totals describe only their snapshots; existing local account closure is not hosted closure; exported local backup does not prove hosted disaster recovery; zero-retention routing is a provider requirement, not independent retention certification; generated model scores are not human/clinical validation; high average rubric scores do not resolve critical failures; existing support cases are not general English-language safety accuracy; an outbox alone does not guarantee exactly-once remote task creation; a resource/catalog ID is not an owner-bound action; old handoff deployment permission is not present task authorization; conditional integration work is not permission to build multiple destinations.

The user-facing promise remains a bounded journaling/reflection/action workflow. Complete the existing release and measure actual user tasks before expanding architecture.


## Optional integration primary references

The PRD's integration contract is proposed and remains deferred. Microsoft documents delegated task creation and linked-resource references, not a remote exactly-once guarantee. An opaque reference supports positive correlation only after actual selected-account behavior is verified. Broad sign-in-session revocation is not an app-only Disconnect operation.

- https://learn.microsoft.com/en-us/graph/api/todotasklist-post-tasks?view=graph-rest-1.0
- https://learn.microsoft.com/en-us/graph/api/resources/linkedresource?view=graph-rest-1.0
- https://learn.microsoft.com/en-us/graph/permissions-reference#tasksreadwrite
- https://learn.microsoft.com/en-us/graph/throttling
- https://learn.microsoft.com/en-us/entra/identity-platform/refresh-tokens
- https://learn.microsoft.com/en-us/graph/api/user-revokesigninsessions?view=graph-rest-1.0
- https://learn.microsoft.com/en-us/entra/identity-platform/v2-oauth2-auth-code-flow
