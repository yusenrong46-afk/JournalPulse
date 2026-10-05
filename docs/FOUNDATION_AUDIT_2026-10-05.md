# JournalPulse foundation audit — 2026-10-05

This audit repairs confirmed defects in the current product before the next feature
upgrade. It started from the existing working tree on
`codex/persistent-conversation-preference`, Git HEAD
`8f49fa63b06b79c2a4af82d2dd2f56c46ca3a02d`. Inherited edits were preserved.
The starting snapshot contains 179 source/configuration/documentation files and a
SHA-256 manifest. Review diffs compare against that snapshot, so earlier work is
not attributed to this audit.

Backend, frontend, provider contracts, persistence, database invariants, dependencies,
and verification helpers were reviewed in parallel, followed by independent review.
Confirmed defects have failing reproductions and passing regression evidence.
Focused comments explain ownership, cancellation, safety, and validation decisions.

## Before and after

| Area | Before | After |
| --- | --- | --- |
| Account isolation | Switching accounts on the same route left private component state mounted; consent, resume IDs, and reminders shared device keys. | The workspace remounts for its owner; browser values are account scoped. Old authentication snapshots cannot undo newer sign-out events. |
| Request ownership | Delayed authentication could send a draft with another user's token. Returning A → B → A could revive an obsolete response or retry. | Requests validate token ownership and capture an account revision. Any identity change permanently cancels that request. |
| Browser robustness | Malformed preference strings could enable consent; blocked storage crashed hooks. Failed sign-out looked successful. | Only literal booleans enable consent; guarded storage has a session fallback. Sign-out failures remain visible. Successful deletion clears local references. |
| Check-in state | Changing the selected decision reused answers, request receipts, and stale loads. | Each decision/prefill combination owns a fresh form; obsolete loads and submissions are canceled. |
| Keyboard access | Privacy dialogs and chat menus lacked focus entry, Escape handling, and correct focus restoration. | Dialogs contain keyboard focus; menus support keyboard navigation and dismissal; focus returns to the opener. |
| Metrics | One completed and one declined action reported completion rate 1.0. Journey deletion left tried/helped counts stale. | Completion rate is 0.5 for that example; deleting a reflection removes its local outcome and updates totals immediately. |
| Safety precedence | Exhausted generation quota blocked support information; an unrelated denial suppressed a separate explicit-risk phrase. | Support is checked before generation quota. Negation applies to its own phrase; typographic apostrophes are normalized. |
| Service boundaries | Auth outages appeared as invalid sessions. Malformed provider/storage data could crash, look empty, or make `"false"` a truthy quota grant. | Outages and invalid storage contracts return controlled failures; quota/deletion receipts require actual booleans and valid counts. |
| Provider declines | Discovery and legacy reflection could accept valid-looking JSON despite native refusal/filter signals. Nested JSON escaped decoder guards. | Native declines take precedence; malformed nesting fails safely. Legacy fallback remains explicitly identified; discovery does not retry a refusal. |
| Provider configuration | Nonfinite/nonpositive deadlines passed readiness. Discovery routing did not require schema-parameter support. | Readiness rejects unusable deadlines; discovery and its offline evaluation request the same privacy and parameter-support constraints. |
| Resource ownership | HTTP and SQLite connections relied on garbage collection; finished or nonexistent conversation IDs accumulated locks. | Owned transports and SQLite connections close explicitly. Only active turns occupy the conversation guard; expired rate-limit identities are reclaimed. |
| SQLite journal paging | Chronological ordering required a temporary sort. | An additive local expression index serves the existing ordering directly. |
| Verification safety | Scratch helpers accepted remote PostgreSQL routing. | A shared local-only guard rejects remote hosts and libpq routing overrides before destructive test setup. |
| Dependencies | The registry reported 39 affected JavaScript packages; pytest had a temporary-directory advisory. | Patched compatible JavaScript dependencies and pytest 9.1.1 remove the known production/runtime findings in the scanned graphs. One unpatched development advisory remains. |

## Measured optimization

| Controlled probe | Before | After |
| --- | --- | --- |
| 250 mocked HTTP operations, owned clients left open | 250 | 0 |
| 250 SQLite reads with garbage collection disabled, file descriptors | 4 → 254 | 4 → 4 |
| 250 completed requests for nonexistent conversations, resident guards | 250 | 0 |
| SQLite chronological page query, median | 4.009 ms | 0.039 ms |

The query benchmark uses 20,000 fictional rows, a 50-row page at offset 200, and
30 measurements per stage. Its query plan loses the temporary B-tree sort.
These are local controlled measurements, not deployed Supabase or end-to-end
latency claims. HTTP clients close per owned operation; this pass does not introduce
a shared connection pool or claim a provider latency improvement.

## Verification

| Check | Starting baseline | Audited source |
| --- | --- | --- |
| Backend tests | 315 passed | 481 passed |
| Backend branch-inclusive coverage | 91.09% | 92.63% |
| Frontend unit tests | 65 passed | 89 passed |
| Mobile/desktop browser and accessibility checks | Previous suite: 42 cases | 44 passed |
| UI → FastAPI → PostgREST → PostgreSQL integration | Previous suite: 13 cases | 14 passed |
| Real PostgreSQL invariant assertions | Existing suites | 100 + 17 + 19 passed |

Ruff, mypy, ESLint, the OpenAPI consistency check, resource catalog validation,
static export, and TypeScript verification passed. Next.js type generation restored
production type paths and retained its new root-parameter declaration and generated
agent guidance after the dependency upgrade. Discovery's final routing change also
passed its 69-case suite after the broader backend run.

The integration stack uses real migrations, the database, REST layer, API, exported
UI, and Supabase client. Auth issuance and model/search providers are deterministic
test doubles. The new account-switch case checks private replies, unsent drafts,
and consent in the real browser. Its first combined run exposed shared test data;
`finally` cleanup restores those accounts for the later RLS test.

## Review artifacts

The review directory is
`/workspace/journalpulse-planning/foundation-audit-2026-10-05/`.
It contains the immutable baseline, hashes, failing/passing logs, detailed findings,
probe scripts, six numbered review patches, one combined patch, and a source-only
archive. The patches are grouped review views of one compatible change set; they
are not separate releases or independently deployable steps.

1. `01-browser-privacy-and-correctness.patch`
2. `02-service-boundaries-and-resource-ownership.patch`
3. `03-request-safety-and-metrics.patch`
4. `04-discovery-and-evaluation.patch`
5. `05-dependencies-and-local-verification.patch`
6. `06-documentation.patch`

## Compatibility and remaining work

Existing authenticated users confirm onboarding/privacy choices again. The old
device-level key has no reliable owner, so its consent is not copied into an account.
Browser account scoping is not encryption; backend ownership remains authoritative.
Canceling a client request cannot undo a write already committed by the server.
Existing idempotency and recovery behavior remains necessary.

The remaining dependency advisory is
[GHSA-vfj7-8cjw-p6xm](https://github.com/advisories/GHSA-vfj7-8cjw-p6xm): deeply
nested glob patterns can exhaust the stack in `braces` 3.0.3. Five development
packages belong to this one affected lint dependency chain. No compatible patched
release is available; user writing is not supplied to the lint glob parser.
The production npm graph has zero reported advisories. The PyPI version feed scan
covers 36 locked runtime/development packages with zero failed queries and zero
remaining findings; optional research packages and OS packages are outside that scan.
A narrow literal-secret check covers nonignored source without logging values;
it is not an exhaustive secret scanner.

The safety component remains an explicit phrase router, not a validated clinical
classifier. Paginated exports are not a transactionally frozen cross-table snapshot.
Scheduler checks use cron metadata stand-ins and do not prove a hosted worker ran.
Model usefulness, indirect-language coverage, search relevance, and the planned
five upgrades still need their own evaluations. Passing these tests does not establish
clinical efficacy or prove that every hidden defect has been eliminated.

This audit changes local source only. The existing protected preview and production
deployment remain unchanged; no paid provider calls or hosted database writes were
made, and applied migrations remain immutable.
