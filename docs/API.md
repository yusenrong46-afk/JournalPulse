# JournalPulse HTTP API

The API is versioned by path: every product route is under `/v1/`, and the contract is the
OpenAPI document the app publishes. `web/openapi.json` is the committed copy and
`web/lib/generated-api.ts` the TypeScript types generated from it; CI fails when either
differs from the running code (`uv run python scripts/export_openapi.py --check`, then
`npm --prefix web run generate:api` and `git diff --exit-code`), so a contract change is
always a reviewed diff. Health routes (`/health`, `/ready`) are unversioned on purpose: hosts
and the hourly synthetic check (docs/SLO.md) call them.

Interactive docs: `/docs` on any non-production run (they are switched off in production).

## Identity

* **Hosted (Supabase):** `Authorization: Bearer <Supabase access token>`. The API asks Supabase
  who the token belongs to, then reads and writes through PostgREST with that same token, so
  row-level security decides what each person can see (`supabase/migrations/`).
* **Local development (`make run`, `make demo`):** `X-JournalPulse-User: <UUID>` names a local
  identity in SQLite. This header is ignored when Supabase is configured.

Writes may send `X-JournalPulse-Data-Revision` (default `0`): after "Delete all my data" the
revision rises and older tabs are refused instead of writing into the erased account.

## Three examples

Start the API with `make run` (AI and search off, SQLite, local identity). The data below is
fictional. `tests/test_api_examples.py` runs these exact commands against the app on every
test run, so they stay true; the `# expect:` line is the status code it checks.

**1. Is it up, and which commit is it?**

```bash
curl -si http://127.0.0.1:8000/health
# expect: 200
```

The body is `{"status":"ok"}`; the `X-JournalPulse-Commit` header names the deployed commit
(`unknown` when the host does not provide one).

**2. What is this server configured to do?** It needs an identity but returns no personal data.

```bash
curl -s http://127.0.0.1:8000/v1/system/status \
  -H 'X-JournalPulse-User: 00000000-0000-4000-8000-0000000000a1'
# expect: 200
```

Returns `analysis_mode` (`ai_configured`, `local_fallback` or `local_only`), `persistence_mode`
(`account` with Supabase, `server_sqlite` locally), a plain-language `message`, and
`build: {"commit": ..., "source": ...}`.

**3. Save a journal entry** (no AI is involved in saving).

```bash
curl -s -X POST http://127.0.0.1:8000/v1/journal/entries \
  -H 'Content-Type: application/json' \
  -H 'X-JournalPulse-User: 00000000-0000-4000-8000-0000000000a1' \
  -d '{"text": "Fictional example: a short walk after lunch helped.", "client_request_id": "6f1d2c3e-0000-4000-8000-000000000001"}'
# expect: 201
```

Returns the saved entry. Sending the same `client_request_id` with the same text again is
safe (the command is idempotent); with different text it is refused with 409, so a retried
request can never overwrite an entry. Only the owner can read it back
(`GET /v1/journal/entries` with the same identity).

## Errors

Errors are JSON with a `detail` string. 401 means no valid identity, 404 is also used for
"not yours" (ownership is never confirmed to someone else), 409 is a conflicting retry or a
stale data revision, 413 a request over `JOURNALPULSE_MAX_REQUEST_BYTES`, 429 a rate limit,
and 503 a dependency that is not ready ("Nothing new was saved"). docs/OPERATIONS.md lists
the failure behaviour of each dependency.
