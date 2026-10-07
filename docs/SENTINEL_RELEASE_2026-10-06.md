# Sentinel repair release — 6 October 2026

The Paper & Lamp UI and the verified safety, privacy and reliability repairs are
ready for release. Apply the new erasure migration before deploying this API.
Publication status must be confirmed from the deployment record, not this document.

## Audit and repair results

Project Sentinel ran first in audit-only mode against the actual dirty working
tree at HEAD fb69f9ee1a757574425f7ac07e333a0222bd495b. Its frozen first-party source
fingerprint was b71ce87341a7b2c0888c9d4fc1884145da3fe76379fc8a6e1c738b11f202eb5c.
The host reported GPT-6 Astra / Ultra, and independent investigators were selected
with the same model and mode. Repairs were a subsequent implementation phase.

| Finding | Repair and acceptance behavior |
|---|---|
| Current risk hidden by historical wording and punctuation | Subject-aware boundaries preserve recognized current risk across punctuation and wrapped lines, while tested historical reports and denials retain their meaning. |
| Delayed acceptance, offers and controls crossing recreated identities | Deleted journal, conversation, activity and reflection creation IDs are retired. Old requests cannot recreate those objects; fresh UUIDs remain usable. Existing incarnation checks remain. |
| Erasure undone by pending saves and automatic retries | Browser erasure invalidation cancels older work, including other tabs. Server request revisions are checked atomically with writes. |
| Server authentication delay crossing account erasure | The browser first reads its authenticated data revision, then sends the same immutable revision with a mutation and its retries. The server never substitutes a fresh revision after waiting for authentication. |
| Failed evaluation evidence promoted to pass | Explicit failure wins over success status labels. Duplicate replay case IDs are rejected before merge rather than allowing order-dependent overwrites. |
| Journal reflection deadline shorter than provider budget | Journal, chat and activity generation share the 125-second client attempt allowance. Revision preflight is separate; stalled SDK/auth waits now settle on timeout or cancellation. |

Independent rechecks found and drove additional repairs for wrapped safety
subjects, historical belief preservation, duplicate evaluation evidence,
obsolete cross-tab notices with blocked storage, server authentication delays,
and stalled client authentication. Final scoped rechecks reported no remaining
actionable blocker in these repaired contracts.

## Validation

- Final backend suite: **1,259 passed**, **91.69% coverage** (80% required).
- Final frontend suite: **199 passed**; ESLint and TypeScript checks passed.
- Ruff and mypy passed; generated OpenAPI is current.
- All 35 catalog resources and 3 support resources passed structural validation.
- The Next static export built successfully.
- Real PostgreSQL checks ran in a task-owned container with no network, ports or
  host mounts. Existing schema checks and the new erasure suite passed, including
  two-session write/erase ordering, private marker access, auth-user cascade,
  old-readiness compatibility and fresh-request recovery.
- Independent final logic checks included 567 tests, belief/current-risk API
  controls, 193 classifier cases, 288 replay combinations and duplicate evidence.
  Backend recheck covered the authentication-delay schedules and 70 targeted
  cases. Frontend recheck covered 47 protocol, cancellation and recovery cases.
- A browser walkthrough used a fresh fictional SQLite database and named local
  provider stand-ins. Chat, activity start/pause/reload/resume, early finish,
  not-tried check-in, follow-up, journal save, account erasure, empty history and
  fresh writing after erasure worked. Mobile rendering was inspected; no console
  errors were observed in that walkthrough.

These are software-contract checks, not live AI quality or clinical validation.
The prior partially completed paid evaluation and unknown charge remain
unresolved; its historical scores are not transferred to this release.

## Deployment and compatibility

Apply only:

    supabase/migrations/202610060002_erasure_boundaries.sql

The earlier 202610050002 and 202610060001 migrations were already applied to the
shared hosted project. Do not replay them, reset the database or invent migration
history. The new API requires readiness v5; v1–v4 retain their response contracts
so the migration can precede the application switch.

Updated clients read GET /v1/account/data-revision and send
X-JournalPulse-Data-Revision with mutations. Headerless legacy POSTs use revision
zero and fail closed after the first account erasure; reload an old tab to load
the updated client. Explicit deletion remains available.

Deletion retains only private technical owner/object IDs and an account erasure
counter, with no writing, summary, report, model output or content hash. These
markers prevent old creation requests from reviving erased content. Deleting the
auth identity removes them. Previously erased IDs cannot be reconstructed into
markers retroactively.

Requests are ordered from their observed server revision; this is not a promise
of global wall-clock ordering between clicks on disconnected devices. Legacy or
equal-time cross-tab notices cancel conservatively. An already-rendered entry
in another tab can remain visible until refresh/navigation. The safety router
remains a bounded English phrase heuristic; real-device accessibility and fresh
model-quality evaluation have not been established by these tests.
