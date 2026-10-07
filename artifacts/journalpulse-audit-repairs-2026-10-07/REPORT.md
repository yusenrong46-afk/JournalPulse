# JournalPulse audit repairs — 7 October 2026

Goal remains active until the production acceptance checks pass.
Base checkout: `fdb5fd449cbf9597b3e0fc5d2c8982434994e7ee`.
Branch: `codex/journalpulse-audit-repairs`.
Audit source: `/Users/thomas/Downloads/JournalPulse-Audit-Evidence.zip`.

## Current evidence

| Finding | Reproduced | Repaired | Tested | Production verified |
|---|---|---|---|---|
| JP-01 draft loss | supplied audit; current live journal navigation reproduced | tab-scoped drafts, discard and fallback warning | unit and real local browser/API/Postgres | pending |
| JP-02 search unavailable | supplied audit; production key/flag scope verified | capability gate, reviewed collection, production scopes | unit and local search fixtures | pending real provider |
| JP-03 repeated ideas | supplied audit; same-goal handler confirmed | accurate goal choice, no duplicate turn, manual collection | frontend regression | pending |
| JP-04 onboarding interruption | supplied audit; Home redirect confirmed | optional setup and safe return destination | unit, desktop/mobile browser, local integration | pending |
| JP-05 stale save list | supplied audit; completion discarded after unmount confirmed | page-independent receipt/result and list reconciliation | unit and delayed-save local integration | pending |
| JP-06 missing activity after refresh | supplied audit; closed-chat early return confirmed | owned accepted reflection and tab timer recovery | timer unit and real local integration | pending |

Related concerns: initial retention copy aligned with Settings; reviewed app activities allow manual choice without AI; Explore topic/exclusions restore without restoring consent; check-in progress after 3s and overall 15s explicit-save deadline.

## Validation

- Backend: 1,270 tests pass; 91.88% coverage. Ruff and mypy pass.
- Frontend: 217 unit tests pass. Lint/typecheck pass.
- Desktop/mobile browser suite: 46 pass, including automated accessibility checks; final cache-transition run in progress.
- Integration: 22 pass; four opt-in UI tour scenarios are explicitly skipped by the existing suite. PostgreSQL, API, exported UI and PostgREST are real; auth and model/search providers are deterministic stand-ins.
- All existing scratch PostgreSQL schema, ownership, integrity, activity, Sentinel and erasure gates pass. No hosted migration is needed or applied.

## Check-in timing

Synthetic local check-in: 82ms from click to visible confirmation; approximately 48.7ms before the write request (revision/auth and browser scheduling), 30.6ms request-to-headers, 30.3ms server processing. Approximately 0.3ms residual transport/scheduling. These are one local observation, not a production latency estimate. The original audit did not establish a measured performance defect.

## Release and cost boundaries

Existing Brave key and search flag have been extended securely from Preview to Production. The current public build has not been replaced yet. Staging must use production configuration with `--skip-domain`; promotion follows staged checks. The prior compatible production deployment is `dpl_7hxiFv7SSDU9ccvdDybpf3EY7CXt` and must be rechecked immediately before release.

Live verification ceiling: US$5, using existing accounts/models; no purchases or upgrades. No paid provider calls yet. OpenRouter generation/cost receipts are returned as bounded metadata and logged in the browser console without topics, chat/journal text or credentials. Missing generation or cost stops further paid verification. Brave list-price upper bound is $0.005/request before monthly credits; gross bound and actual OpenRouter charge are kept distinct.
Pricing sources: https://openrouter.ai/api/v1/models and https://brave.com/search/api/ .

Session-storage recovery is tab-local and depends on browser session behavior; it is not a guarantee of secure physical erasure. Sign-out and account erasure explicitly clear the application’s tab records. Real personal entries were neither opened nor modified.

## Release transition defect and repair

The first staged candidate `dpl_98VuPAtmkiGGuoNzgCHgavJaUrPN` was briefly promoted, then rolled back immediately when the existing production browser reported `ChunkLoadError`. The requested chunk returned HTTP 200 on the prior deployment and 404 on the candidate; candidate HTML referenced a different current chunk. The service worker cached runtime chunks indefinitely. Export archives also normalize asset mtimes, making metadata-only ETags unsafe across equal-sized builds.

The repair caches only the standalone offline page, advances the worker cache to v6 (clearing earlier asset caches), requests worker updates without HTTP cache reuse, and serves exported assets with no-store and without metadata-only 304 responses. Chunk error recovery performs a fresh document reload on explicit Try again. A real exported-UI integration test injects a stale chunk into the old worker cache and confirms it cannot replace the current runtime. An API regression test proves equal-size/equal-mtime assets return the new bytes instead of a stale 304. No user writing was entered during the failed production transition, and no paid provider call occurred.

## Manual selection correction

Production inspection found that the reviewed-activity picker reached the existing API with a catalog ID but without a current Luna offer. The API correctly rejected this as a changed recommendation. The repair adds optional, strict boolean `user_selected` to activity creation (default false). Explicit choices resolve only trusted catalog/built-in IDs and still obey ownership, source validity, conversation revision, support/listen state and hard activity constraints. They carry user provenance, no model credit and no OPE eligibility. No database change is needed. SQLite boundary tests and real PostgreSQL/browser integration cover creation, exact retry and refresh recovery. The corrected selection is awaiting final hosted verification.

Production verification already confirmed exact journal/chat draft recovery, interrupted Save visibility, optional Home, retention copy, same-goal message count remaining at two, manual collection visibility, Explore topic/exclusion recovery with consent off, legacy activity links and timer states through refresh, and a real 390x844 viewport with no horizontal overflow. Check-in saved a synthetic not-tried outcome in 1176ms including automation overhead. Detailed client timing APIs were unavailable in the native read-only browser scope; local integration provides phase timings.

Real production Explore initial/refinement and inline initial search succeeded, with generation/cost receipts. One inline topic containing `short` was rejected by the existing vocabulary gate before provider work; the valid public phrase `quiet meditation` succeeded. Current OpenRouter measured spend is US$0.0007225. Three Brave requests have a gross ceiling of US$0.015 before credits. These billing layers remain separate.

## First-chat draft handoff

A new failing regression reproduced loss of the first draft after chat creation changed the URL but the message response remained unconfirmed. The draft now transfers to the confirmed chat/incarnation before sending. This also preserves any unsent typed text when a separate mood tap creates the chat. The regression is preserved in `first-chat-draft-red.log`; the repaired frontend suite passes 217 tests. Existing server contracts are unchanged by this client correction.
