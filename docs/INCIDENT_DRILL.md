# Incident drill: database unreachable

A drill, run locally, of the failure that matters most for a journal: the database disappears
while people are writing. It checks three promises from [SLO.md](SLO.md) and
[OPERATIONS.md](OPERATIONS.md): readiness tells the truth, a write that cannot be stored is refused
with nothing half-saved, and a retry after recovery stores exactly one record.

## How to run it

```bash
export JOURNALPULSE_PG_DSN=postgresql://postgres:postgres@localhost:5432/postgres   # disposable
uv run python scripts/incident_drill.py --db-container jp-pg      # host runtime
uv run python scripts/incident_drill.py --runtime compose         # or the Compose environment
```

`scripts/incident_drill.py` stands up an onboarding environment (docs/ONBOARDING.md), signs in an
anonymous drill identity, pauses the PostgreSQL container (`docker pause`), watches `/ready`,
attempts a save, resumes the container, and retries the same save with the same
`client_request_id`. It writes the timeline to `outputs/incident-drill/<date>.json`. It pauses only
the container you name; never point it at a shared database.

## Recorded run: 2026-11-18

Commit `761ab6f`, host runtime (Docker Compose could not pull its build images here), PostgreSQL 16
in a local container, PostgREST 12, the API with AI off. Timeline from
[`outputs/incident-drill/2026-11-18.json`](../outputs/incident-drill/2026-11-18.json), seconds from
the start of the drill:

| t (s) | Event |
|---:|---|
| 14.0 | Environment ready (`/ready` 200); anonymous drill identity signed in |
| 14.1 | **Database paused** (connections hang; nothing is refused outright) |
| 49.2 | `/ready` returns **503** with `database: unreachable`, 35 s after the outage began |
| 64.2 | A journal save during the outage returns **503**: "Your journal could not be reached. We could not confirm its latest state; please retry." |
| 64.3 | **Database resumed** |
| 80.4 | `/ready` returns **200** again, 16 s after recovery |
| 80.6 | The same save, retried with the same `client_request_id`, returns **201** |
| 80.8 | The drill identity has exactly **1** entry: the refused attempt stored nothing |

The drill checks what the runbook's steps promise rather than replaying them by hand. Step 1:
`/ready` reported `database: unreachable`. Step 2: the write failed closed
with 503 and stored nothing. Step 4: `/ready` returned 200 after recovery, and one save appeared
exactly once. Step 3 (the Supabase dashboard) and step 5 (closing the synthetic-check issue) have no
local equivalent and were not exercised.

## What the drill found

1. **Detection is slow by design: 35 s.** `/ready` caches its database probe for 30 s
   (`database_checks` in `src/journalpulse/api.py`) and then waits for the probe's timeout. That is
   fine against an hourly check, but a person can see failures before `/ready` does. Recorded,
   not changed: the cache protects the database from probe storms.
2. **A refused write took 15 s.** The save was sent after `/ready` turned 503 and came back 503
   15 s later: the API's database client waits up to 15 s before giving up (`timeout=15` in
   `_request`, `src/journalpulse/persistence.py`). The message is clear and nothing is saved, but
   15 s is a long wait in a writing app. A shorter write timeout is worth considering; recorded,
   not changed in this upgrade.
3. **OPERATIONS.md had no step-by-step procedure for this incident**, only the failure-behaviour
   table. Added "Incident: database unreachable" (confirm, leave the app alone, check the database
   side, verify with a disposable account, record).
4. **The local auth stand-in dropped the connection on an upstream timeout** instead of answering
   with a 5xx. Fixed (it now answers 504 and ignores a caller that has already gone). This affects
   only `scripts/onboarding_auth.py`, not production, where Supabase's gateway answers.

## Not covered

The hosted path (Supabase's own gateway, connection pooler and a paused free-tier project) was not
drilled; that needs the production project and is an owner decision. Recovery from data loss is
covered separately by `scripts/verify_recovery.py` (OPERATIONS.md, "Recovery verification evidence").
