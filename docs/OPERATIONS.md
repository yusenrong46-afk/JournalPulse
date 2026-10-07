# Operations Runbook

This runbook covers running JournalPulse for a small beta. It makes no claim of clinical safety,
therapeutic effect, or large-scale availability.

## Configuration reference

Use [.env.example](../.env.example) and [Settings](../src/journalpulse/config.py) as the source of defaults. Server environment variables take precedence over the local `.env` fallback.

| Variable | Default | Purpose |
|---|---|---|
| `JOURNALPULSE_ENV` | `local` | `production` requires hosted auth/storage, signing and explicit HTTPS origins |
| `JOURNALPULSE_LLM_ENABLED` | `true` | Disables model use when `false`; a key and user consent are still required when enabled |
| `JOURNALPULSE_LLM_API_KEY` / `OPENROUTER_API_KEY` | empty | Existing OpenRouter credential, server only |
| `JOURNALPULSE_CHAT_MODEL`, `JOURNALPULSE_LLM_MODEL` | `openai/gpt-6-luna` | Chat and legacy structured-analysis models |
| `JOURNALPULSE_LLM_ZDR` | `true` | Required zero-data-retention routing |
| `JOURNALPULSE_CHAT_TIMEOUT_SECONDS` | `45` | Per-attempt chat timeout; attempts share the provider deadline |
| `JOURNALPULSE_ANALYSIS_RATE_LIMIT_PER_MINUTE` | `20` | Per-person model-request limit |
| `JOURNALPULSE_SEARCH_ENABLED` | `false` | Opts the deployment into external discovery after all required providers are configured |
| `JOURNALPULSE_SEARCH_API_KEY` | empty | Existing Brave credential, server only |
| `JOURNALPULSE_SEARCH_TIMEOUT_SECONDS` | `8` | Search timeout, constrained to 1–20 seconds |
| `SUPABASE_URL`, `SUPABASE_ANON_KEY` | empty | Hosted Auth/PostgREST connection; anon key is public and relies on RLS |
| `JOURNALPULSE_WRITE_SIGNING_KEY` | empty | Server provenance signing; must match `private.server_secrets` and is required in production |
| `JOURNALPULSE_CORS_ORIGINS` | platform URL or loopback locally | Explicit allowed origins |
| `JOURNALPULSE_WEB_DIST` | empty | Exported frontend directory; empty permits API-only development |
| `JOURNALPULSE_DB_PATH` | `artifacts/research_beta.db` | Local SQLite file |
| `JOURNALPULSE_MEMORY_ENABLED`, `JOURNALPULSE_ADAPTIVE_POLICY_ENABLED` | `false` | Disabled research features |

Browser values are build inputs: `NEXT_PUBLIC_SUPABASE_URL`, `NEXT_PUBLIC_SUPABASE_ANON_KEY`, and local-only `NEXT_PUBLIC_API_BASE_URL` / `NEXT_PUBLIC_DEV_USER_ID`. The Vercel and Docker build scripts copy the public Supabase values. Never expose model, search or signing secrets through a `NEXT_PUBLIC_` variable.

`GET /v1/capabilities` makes no provider call. Discovery `configured` means the required flags and credentials are present, not that live search has succeeded. Enable search in the intended Vercel environment; Preview configuration does not configure Production. Keep scoped consent and the reviewed-resource fallback even after enabling it.

## Runtime checks

- `/health` only proves the API process answers.
- `/ready` returns 503 unless every required layer is ready, and reports each one separately:

  | Check | Values |
  |---|---|
  | `configuration` | `ready`, or `not_ready:<issues>` (for example `signing_key_missing`) |
  | `resources` | `ready`, or `not_ready:<error>` |
  | `llm` | `configured:not_probed`, `not_configured`, or `disabled`. `/ready` never spends a model call |
  | `web` | When `JOURNALPULSE_WEB_DIST` is set: `ready` or `not_ready:export_missing`. An index page is required; API-only deployments omit this check |
  | `database` | `reachable`, `unreachable`, `error:<status>`, or `local_sqlite` |
  | `schema` | `schema_ready`, `schema_missing`, or `schema_outdated:<version>` |
  | `signing` | `valid`, `invalid` (the API and database keys differ), `missing` (none in the database), or `not_configured` (none in the API) |
  | `retention_job` | `scheduled`, `not_scheduled`, or `unknown` |
  | `retention_state` | `healthy`, `missing`, `never_succeeded`, `failed`, `overdue`, or `unknown` |
  | `retention_last_success` | Last corroborated cron completion time, `never`, or `unknown` |
  | `retention_last_run_status` | Bounded cron status; no error messages or journal text |
  | `retention_overdue` | `true`, `false`, or `unknown`; 30-minute threshold |

  The database probe is cached for 30 seconds per instance.
- Production CORS accepts explicit HTTPS origins, with an optional port. Wildcards,
  credentials, paths, query strings, fragments, and loopback origins fail readiness.
- Every response carries `X-Request-ID`. Logs hold method, path, status, latency, and that ID, never
  request bodies, authorization headers, keys, or journal text.

## Deploy

1. Rotate any key that has appeared outside a secret store.
2. Run the checks in [Engineering reproduction](ENGINEERING.md#reproduction). CI runs the release gates on pushes to main and pull requests.
3. Apply only new `supabase/migrations/` in filename order; never replay or rewrite applied migrations.
   `scripts/verify_postgres_schema.py` rebuilds a
   scratch database from the migrations and checks RLS, provenance signing, lifecycle races, retention,
   the rate limit, and deletion. Scratch verification and integration helpers accept
   only local loopback or Unix-socket PostgreSQL routing and reject hosted DSNs.
4. Create one signing key per environment (at least 32 characters, for example
   `openssl rand -hex 32`) and store it in both places:
   - in the database, as the postgres role:
     `insert into private.server_secrets (name, value) values ('write_signing_key', '<key>')
     on conflict (name) do update set value = excluded.value;`
   - on the host, as `JOURNALPULSE_WRITE_SIGNING_KEY`.
5. Set the other production variables on the host: `JOURNALPULSE_ENV=production`, `SUPABASE_URL`,
   `SUPABASE_ANON_KEY`, `JOURNALPULSE_LLM_API_KEY`, `JOURNALPULSE_LLM_ENABLED=true`, and the model names.
6. Deploy:
   - **Vercel (production):** also set `JOURNALPULSE_WEB_DIST=web-dist`, then `vercel deploy --prod`.
     The build runs `scripts/build_vercel_web.py`, which exports the site with the Supabase values baked
     in.
   - **Render (alternative):** create a Blueprint from `render.yaml`; it asks for the model key,
     public Supabase anon key, and signing key. The signing key must match `private.server_secrets`.
     `Dockerfile.api` builds the site and serves it from FastAPI.
7. In Supabase, under **Authentication → URL configuration**, set the Site URL to the public address and
   add `https://<address>/**` as a redirect URL. Without this, sign-in links return to localhost.
8. Confirm `/ready` reports `database: reachable`, `schema: schema_ready`, `signing: valid`, and
   `retention_job: scheduled`. Observe an actual completed run and `retention_state: healthy`;
   a schedule alone does not verify cleanup.
9. Smoke test with a disposable account: sign in, chat to a saved step, check in, view Journey, export,
   then delete the test entry.

The migration and the code must ship together: the migration removes the old unsigned write functions
that earlier releases call.

`scripts/verify_openrouter.py` and `scripts/verify_conversation.py` make paid live calls. Run them only
when a live check is intended.

## Retention

- `jp_purge_expired_conversations` runs every 15 minutes through `pg_cron` as the `journalpulse-retention`
  job. It closes chats idle for more than 24 hours, clears text the person did not choose to keep, repairs
  any closed chat that still holds such text, and removes usage counters older than a day.
- The migration schedules the job when `pg_cron` is available. On Supabase, if it is not enabled, turn on
  **Database → Extensions → pg_cron** and rerun the scheduling block at the end of
  `202609280001_phase_a_integrity.sql`.
- Check this job's recent runs, without exposing error text:
  `select r.status, r.start_time, r.end_time from cron.job_run_details r join cron.job j using (jobid)
  where j.jobname = 'journalpulse-retention' order by r.start_time desc limit 10;`
- `202610040002_retention_diagnostics.sql` adds one private timestamp row. The scheduled entry point
  writes completion after successful cleanup in the same transaction; failures roll it back.
  Readiness corroborates completion with successful cron history after monitoring began. A manual
  purge and request-time cleanup cannot make a broken scheduler appear healthy. Missing/unreadable
  history, an unexpected schedule/command, or inconsistent evidence reports `unknown`.
- The proposed alert threshold is **more than 30 minutes** since successful scheduled completion.
  A new installation gets 30 minutes before `retention_overdue` becomes true, but reports
  `never_succeeded` until a run succeeds. A later cron failure reports `failed`; a later successful
  run recovers to `healthy`. The schema verifier checks both 30-minute boundaries with a fixed clock.
- Retention degradation remains visible without making `/ready` fail: privacy actions and the app
  remain available. Monitor these fields separately; investigate `missing`, `failed`, `unknown`, or
  overdue cleanup. Do not rely on HTTP 200 alone for retention health.
- When scheduled cleanup stops, each person is still swept when they next use chat. Text for people
  who never return stays until cleanup succeeds. An operator can run the entry point as postgres:
  `select public.jp_purge_expired_conversations();` This repairs data but does not prove cron works.
- The cutoff is 24 hours **of inactivity**, plus scheduling delay (normally up to 15 minutes), locks,
  and any outage delay. Cleanup skips rows currently locked and retries on later runs. It is not an
  exact 24-hour deletion promise.
- Chats with **Keep my messages** on can still be closed for inactivity; their text is retained.
- The private singleton retains only monitoring-start and latest entry-point completion timestamps.
  Cron history follows the host's history-retention policy. Clearing it may temporarily make a
  formerly healthy scheduler appear unverified until a new run completes.

### Release the preference and diagnostics slice

Apply `202610040001_conversation_preference.sql`, then `202610040002_retention_diagnostics.sql`.
These are additive replacements; do not edit an already applied migration. Deploy the matching API
and frontend together. The new API calls `jp_readiness_v2` and expects `phase-a-2`;
the original `jp_readiness` keeps its `phase-a-1` response so the existing live API
continues reporting readiness during a preview rollout. A missing v2 function still
makes the new API not ready; it never falls back to the older schema contract.
For a release without a preference endpoint mismatch, use a maintenance window or the host's atomic
deployment switch, then verify a disposable chat, reload, Just talk, resumption, export, and deletion.

Older API code is insufficient for preference-aware chats even though migrations preserve canonical
preference and block unsafe acceptance. Keep this compatible API/frontend pair as the rollback target;
repair forward rather than promoting an older frontend/API across this contract. A database restore
must be coordinated with the application version and the host's verified restore procedure.

When a preview shares the existing Supabase project, these migrations still update shared lifecycle
functions. Use a disposable account and fictional messages; keep preference-aware preview chats
away from the older application. Preview-scoped `JOURNALPULSE_LLM_ENABLED=false` disables paid
calls without changing production settings. Apply migrations only with authorized database access,
then check both the existing site's readiness and the preview's readiness and user flow.

## Failure behaviour

| Failure | Behaviour |
|---|---|
| Model timeout or transient 5xx | Retry within the configured attempts, then fail the turn with 502 or 503. Nothing is saved and the browser keeps the text for "Try again" |
| Model reply breaks the schema, is cut off, or names an unknown feeling | Reject the whole reply; save nothing for that turn |
| Chat closed, accepted, or answered elsewhere while a reply was running | The commit is refused: 409 for a closed or changed chat, 404 for a deleted one. Nothing is saved; the page reloads the chat |
| Two accepts at once | Exactly one is saved. The same request ID gets the saved record; a different one gets 409 |
| Safety check matches | Support mode for the rest of the chat; the model is never called and every reply is the support message |
| Supabase Auth unavailable | 503; never fall back to a development identity |
| Database unreachable | 503 with "Nothing new was saved" |
| Shared rate-limit counter unreachable | 503 before any model call (fail closed) |
| Rate limit reached | 429 with `Retry-After`; reading history and privacy actions still work |
| Signing key missing or mismatched | Writes fail with 503 and `/ready` reports `signing` as not valid |
| Expired session | Refresh once, then send the person to sign in |
| Response lost after a write | The browser retries with the same client UUID and gets the original record |
| Oversized request | 413 before any model or database work |
| Device offline | A banner says nothing new will be sent; the offline page is the only cached page |

## Data recovery and deletion

- Enable and test Supabase backups and point-in-time recovery for the chosen plan. Before inviting more
  people, restore a backup into a separate project and check record counts and RLS.
- `DELETE /v1/reflections/{id}` removes the decision, outcome, observation, model run, and safety event
  through foreign-key cascades.
- `DELETE /v1/account/data` removes every journal row: outcomes, observations, decisions, model runs,
  safety events, messages, memories, reflections, conversations, consents, and profiles. Usage counters
  are left to expire so deletion cannot reset the generation limit. The sign-in identity is kept;
  deleting it is a separate privileged task and must not be described as done until it is built.

## Rollback

- Keep the previous deployment available. On Vercel, promote the last good deployment.
- Never roll code back across an incompatible migration; add a forward repair migration instead. Code from
  before `202609280001_phase_a_integrity.sql` cannot write to a database that has it.
- Set `JOURNALPULSE_LLM_ENABLED=false` to stop all model calls. Chats continue with Simple Luna.
- The adaptive-policy and memory flags are independent kill switches and stay off until their research
  gates pass.
