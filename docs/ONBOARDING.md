# Onboarding: a whole environment in one command

```bash
make setup                      # once: locked Python and Node dependencies
make onboard NAME=<env>         # e.g. NAME=alex; lower-case letters, digits and hyphens
```

`make onboard` builds a complete local JournalPulse environment, checks it end to end, writes
`outputs/onboarding/<env>/REPORT.md` with the time of every step, and tears it down
(`uv run python scripts/onboard.py <env> --keep` leaves it running until Ctrl+C). The first
recorded run is [outputs/onboarding/first-run/REPORT.md](../outputs/onboarding/first-run/REPORT.md).

## What one environment is

| Part | Real or stand-in | Where |
|---|---|---|
| PostgreSQL 16 with **every migration** in `supabase/migrations/`, applied in filename order on a fresh database | Real | `scripts/onboard.py` (`prepare_database`), `scripts/pg_harness.sql` for the Supabase-provided `auth` schema and roles |
| PostgREST 12, the REST layer Supabase uses, enforcing row-level security | Real | `postgrest/postgrest:v12.2.12` image, or the SHA-256-pinned binary on the host |
| The API (`journalpulse.api:app`) serving the exported web app | Real (the production code paths for Supabase) | `Dockerfile.api` image, or uvicorn on the host |
| Supabase Auth | Stand-in: anonymous demo sign-ins and token lookup only | `scripts/onboarding_auth.py`, `deploy/onboarding/Dockerfile.auth` |
| AI and search | Off (`JOURNALPULSE_LLM_ENABLED=false`, no provider key) | — |

Every port binds 127.0.0.1 and is chosen at random, so environments with different names can
exist side by side. Secrets (JWT secret, signing key, database passwords) are generated per run,
passed only through process environments, and never printed or written to the report.

## What it checks, in order

1. **Database:** a fresh `jp_onboard_<env>` with all migrations, a PostgREST login and the signing key.
2. **PostgreSQL verifiers:** the seven scripts of `make test-postgres` (schema and RLS, journal,
   integrity, activities, sentinel repairs, erasure, demo sandbox) against the same cluster.
3. **Build:** the API and auth images (Compose), or the web export (host).
4. **`/ready` returns 200**, with `database: reachable`, `schema: schema_ready`, `signing: valid`.
5. **Build identity:** `/health` names the commit that was checked out (`X-JournalPulse-Commit`).
6. **Demo journey** in a real browser (`web/tests/demo/demo-journey.spec.ts`): Try the demo, save an
   entry, Simple Luna activity, check-in, Journey. Requests to anything but the environment's own
   API and auth ports are refused and fail the run. Pictures go to `outputs/onboarding/<env>/journey/`.

## Runtimes

* **Compose** (`--runtime compose`): everything runs from `deploy/onboarding/compose.yaml`. Needs
  Docker with Compose and access to Docker Hub, PyPI and npm for the first build. CI runs this
  path on every pull request (job `portfolio-onboard`).
* **Host** (`--runtime host`): the same processes on this machine, against the disposable local
  PostgreSQL 16 named by `JOURNALPULSE_PG_DSN` (scripts refuse any non-local address). For example:

  ```bash
  docker run -d --name jp-pg -e POSTGRES_PASSWORD=postgres -p 127.0.0.1:5432:5432 pgvector/pgvector:pg16
  export JOURNALPULSE_PG_DSN=postgresql://postgres:postgres@localhost:5432/postgres
  ```

* **Auto** (the default): tries Compose, and falls back to host when Docker or its images are not
  available. The report's Runtime row says which ran and why.

The browser journey needs Playwright's Chromium once: `npx --prefix web playwright install chromium`.

## Timings

From `outputs/onboarding/first-run/REPORT.md` (host runtime, one laptop-class container): the whole
run took about a minute, most of it the PostgreSQL verifiers and the web export; the Compose
runtime adds the image builds on the first run. Read the report for the current numbers rather
than this paragraph.

## When it fails

The report is written even on failure, with the failing step and its message. Common causes:
`JOURNALPULSE_PG_DSN` unset (host runtime), Playwright's browser not installed, a port race
(rerun), or Docker Hub rate limits (Compose runtime; auto falls back to host).
