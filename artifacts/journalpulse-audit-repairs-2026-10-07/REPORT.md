# JournalPulse audit repairs — 7 October 2026

All six audit findings and four related UX concerns are repaired and production verified.

- Live app: https://journalpulse.vercel.app/
- Runtime source commit: `779ac88a5cd809f51e9fed579fe5ef960f00800c`.
- Deployment: `dpl_BQCGgvyeErkmRktk2Uoii7nKTHrr`.
- Immutable build: https://journalpulse-r2k9nn2nq-yusenrong46-9212s-projects.vercel.app/
- Branch: `codex/journalpulse-audit-repairs`. Base: `fdb5fd449cbf9597b3e0fc5d2c8982434994e7ee`.
- Supplied audit: `/Users/thomas/Downloads/JournalPulse-Audit-Evidence.zip`.

## Findings and acceptance evidence

| Finding | Repair | Verification |
|---|---|---|
| JP-01 — writing disappears | Account/tab-scoped journal and chat drafts, explicit discard, storage-failure warning; first draft follows confirmed chat identity before send | Baseline journal loss independently reproduced. Exact synthetic text survives production navigation and refresh; account, incarnation, erasure, source and failure regressions pass. First-chat failure has preserved failing-before/passing-after evidence. |
| JP-02 — both search paths unavailable | Upfront capability check, useful reviewed resources, production Brave configuration, plain unavailable messages | Real Explore initial and refinement returned different results; real inline search returned three signed selectable offers. Frozen final deployment inline smoke passed with an actual generation/cost receipt. |
| JP-03 — repeated ideas | Accurate “Choose another goal” control, finite-collection notice, no duplicate turn, reviewed resource browsing | Production user-message count stayed at two after repeating the same goal. |
| JP-04 — Home returns to welcome | Optional setup with private defaults and safe internal return destination | Production incomplete-onboarding Home remains usable; consent preferences were not changed. Return-path and hostile destination tests pass. |
| JP-05 — Save becomes invisible | Page-independent receipt/result coordinator; list reconciliation; preserve newer editor text | Save followed immediately by navigation and return showed its own entry and confirmation without reload. Exact saved text was reopened. Delayed-save/list-race tests pass. This was not backend write loss. |
| JP-06 — activity controls lost on refresh | Owned accepted reflection restoration; legacy tab timer state; existing canonical server sessions retained | Saved audit meditation restores links and controls. Pause stays at 6:56 across refresh, running timer survives refresh, Reset stays at 7:00. A new manual server-backed choice also survives refresh. |

Related improvements are verified: initial retention copy explicitly mentions summaries, reported feelings and activity choices; reviewed resources allow manual browsing and explicit in-chat choice without another model call; same-tab Explore topic/manual exclusions restore with sharing consent off; check-in has a three-second progress notice and an overall 15-second explicit-save deadline with immutable retry answers.

The corrected manual choice resolves only trusted catalog/built-in IDs. Ownership, source validity, revision, support/listen and hard preferences remain enforced. User choices have user provenance, no model credit and OPE=false. Browsing resources in Explore/Simple mode opens links without claiming a saved choice or completion; the in-chat picker persists an activity choice.

## Validation

- **1,270 backend tests pass**, with **91.88% coverage**. Ruff and mypy pass.
- **217 frontend unit tests pass**. Lint, typecheck, OpenAPI and production build pass.
- **46 desktop/mobile browser checks pass**, including automated accessibility checks.
- **22 real local integration tests pass**; four existing opt-in UI tours remain skipped. The API, exported UI, PostgreSQL and PostgREST are real; local auth and model/search providers are explicitly simulated.
- All existing local scratch schema, ownership, integrity, lifecycle, Sentinel and erasure gates pass. No hosted migration was needed or applied.
- A real production viewport measured **390×844**, document width 390, with no horizontal overflow; override reset afterward.
- Final deployment reports ready, reachable storage, current schema, valid signing and healthy retention. A deployment-scoped error/fatal scan returned zero entries. This does not claim complete accessibility, clinical or security qualification.

## Check-in timing

The latest local PostgreSQL observation was 99ms from click to visible confirmation: approximately 50.7ms before the write (revision/auth and scheduling), 44.7ms to response headers and 44.2ms server processing. The production synthetic “not tried” save showed confirmation in 1176ms including browser automation overhead. Native read-only inspection did not expose detailed production client timing APIs. The original long Saving delay was not reproduced or attributed to a measured performance defect; no production percentile claim is made.

## Release and interface changes

New read-only capability and reviewed-resource endpoints; optional owned `accepted_reflection` in conversation detail; optional strict `user_selected` in activity creation, default false. Search returns bounded generation/cost metadata. No database schema changes.

The first candidate briefly produced a browser ChunkLoadError and was rolled back. Its requested old chunk returned 200 on the previous build and 404 on the candidate. The service worker had cached runtime chunks indefinitely. The repair caches only the standalone offline page, clears earlier worker caches, requests fresh worker updates and serves exported assets with no-store, avoiding metadata-only ETag collisions from normalized archive mtimes. Regression tests cover stale worker chunks and equal-size/equal-mtime assets. The final asset graph and fresh authenticated page checks pass. The previous compatible deployment remains available for rollback.

Source manifests and staged/public checks identify the exact runtime build. Later evidence-only commits do not change runtime source.

## Provider accounting

Existing accounts/models were used; no purchases, upgrades or model substitutions. All five successful model generations have known IDs and actual reported costs. Four Brave requests are counted. One disallowed inline word was rejected by the existing vocabulary guard before provider work. No unknown charge is left unresolved.

- OpenRouter actual measured charge: **US$0.0009384**.
- Brave gross upper bound: **US$0.020**, before credits; exact credit discount is not claimed.
- Combined gross ceiling: **US$0.0209384**, within the agreed **US$5** ceiling.

See `cost-ledger.json`. Pricing: https://openrouter.ai/api/v1/models and https://brave.com/search/api/ .

## Evidence and limits

`production-evidence.json`, `release-state.json`, source manifests, test logs and `screens/` contain the acceptance trail. All authored content and outcomes were synthetic. No real personal entry was opened, edited or deleted. The created journal entry, guided QA chat, synthetic skipped outcome and unstarted manual choice are listed in the evidence file and left in place.

Draft recovery is tab-local and depends on browser session behavior; it is not a guarantee of physical secure erasure. Account change, sign-out and erasure clear application tab records. Browser checks wait for the loaded state; intermediate hydration blanks are not classified as defects.

The repair branch is pushed. An optional draft PR was not created because automatic approval review required explicit publication authorization. The prepared PR body is `/private/tmp/jp-audit-pr-body.md`; this optional action is separate from the completed repair goal.
