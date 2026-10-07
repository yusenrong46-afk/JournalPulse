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

- Backend: 1,263 tests pass; 91.72% coverage. Ruff and mypy pass.
- Frontend: 216 unit tests pass. Lint/typecheck pass.
- Desktop/mobile browser suite: 46 pass, including automated accessibility checks; final run in progress.
- Integration: 20 pass; four opt-in UI tour scenarios are explicitly skipped by the existing suite. PostgreSQL, API, exported UI and PostgREST are real; auth and model/search providers are deterministic stand-ins.
- All existing scratch PostgreSQL schema, ownership, integrity, activity, Sentinel and erasure gates pass. No hosted migration is needed or applied.

## Check-in timing

Synthetic local check-in: 82ms from click to visible confirmation; approximately 48.7ms before the write request (revision/auth and browser scheduling), 30.6ms request-to-headers, 30.3ms server processing. Approximately 0.3ms residual transport/scheduling. These are one local observation, not a production latency estimate. The original audit did not establish a measured performance defect.

## Release and cost boundaries

Existing Brave key and search flag have been extended securely from Preview to Production. The current public build has not been replaced yet. Staging must use production configuration with `--skip-domain`; promotion follows staged checks. The prior compatible production deployment is `dpl_7hxiFv7SSDU9ccvdDybpf3EY7CXt` and must be rechecked immediately before release.

Live verification ceiling: US$5, using existing accounts/models; no purchases or upgrades. No paid provider calls yet. OpenRouter generation/cost receipts are returned as bounded metadata and logged in the browser console without topics, chat/journal text or credentials. Missing generation or cost stops further paid verification. Brave list-price upper bound is $0.005/request before monthly credits; gross bound and actual OpenRouter charge are kept distinct.
Pricing sources: https://openrouter.ai/api/v1/models and https://brave.com/search/api/ .

Session-storage recovery is tab-local and depends on browser session behavior; it is not a guarantee of secure physical erasure. Sign-out and account erasure explicitly clear the application’s tab records. Real personal entries were neither opened nor modified.
