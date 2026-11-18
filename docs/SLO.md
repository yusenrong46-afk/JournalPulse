# Service level objectives

JournalPulse is a research beta run by one person, so these objectives are deliberately few and
each one is measured by something in this repository. Production is
https://journalpulse.vercel.app (override with the repository variable
`JOURNALPULSE_PRODUCTION_URL`).

## Objectives

| Objective | Target (30 days) | Measured by |
|---|---|---|
| **Availability:** an hourly check finds `/health` 200 and `/ready` 200 (and, once the demo is live, the browser demo journey passes) | 99.5% of checks (about 3 failed checks in 720) | `.github/workflows/synthetic.yml` → `status/history.jsonl` on the `status` branch; README badge from `status/availability.json` |
| **Readiness means usable:** `/ready` is 200 only when configuration, resources, database, schema and signing are all ready | Always (a release gate, not a percentage) | `src/journalpulse/api.py` (`ready`), `tests/test_research_beta_api.py`, `make onboard` |
| **Nothing half-saved:** a write that cannot reach the database is refused with 503 and stores nothing; retrying the same request stores exactly one record | Always | `docs/INCIDENT_DRILL.md` (run locally), idempotency tests in `tests/` |
| **A turn ends:** every chat turn returns a reply or a clear failure within `JOURNALPULSE_CHAT_TIMEOUT_SECONDS` (45 s) | Always | enforced in code; **not yet measured in production** |

Error budget: 0.5% of hourly checks, about 3.6 checks in 30 days. When it is spent, the next
change is a reliability fix, not a feature.

## Alerting

Two consecutive failed checks (about two hours) open one GitHub issue, "Synthetic check failing
on production", which links the run and docs/INCIDENT_DRILL.md. No second issue is opened while
one is open. There is no paging: this is a research beta, and the hosted app's own failure
behaviour (docs/OPERATIONS.md) is to refuse safely with 503, never to save half a record.

## What the check does and does not cover

* It calls only public URLs, with no account and no secret, and never sends journal text.
* `/ready` reports configuration, schema and signing as well as reachability, so a deploy whose
  migration was not applied counts as down, as it should.
* The browser journey runs only once the demo is live and the repository variable
  `JOURNALPULSE_SUPABASE_HOST` names the Supabase host it signs in through (public, not a secret).
  Each run creates one anonymous demo account, which the demo's own erasure job removes after
  24 hours (docs/DEPLOY_DEMO.md).
* Model quality, safety routing and latency of AI replies are measured offline
  (docs/evaluations/portfolio/), not by this check.

## Status

The workflow runs on a schedule only from the default branch, so **no production measurement
exists yet**: the badge says "no data" until the upgrade is merged and the first run pushes the
`status` branch. Until then the numbers above are targets, not results.
