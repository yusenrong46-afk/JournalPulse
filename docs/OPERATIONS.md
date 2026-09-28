# Operations Runbook

This runbook covers running JournalPulse for a small beta. It makes no claim of clinical safety,
therapeutic effect, or large-scale availability.

## Runtime checks

- `/health` only proves the API process answers.
- `/ready` returns 503 unless every required layer is ready, and reports each one separately:

  | Check | Values |
  |---|---|
  | `configuration` | `ready`, or `not_ready:<issues>` (for example `signing_key_missing`) |
  | `resources` | `ready`, or `not_ready:<error>` |
  | `llm` | `configured:not_probed`, `not_configured`, or `disabled`. `/ready` never spends a model call |
  | `database` | `reachable`, `unreachable`, `error:<status>`, or `local_sqlite` |
  | `schema` | `schema_ready`, `schema_missing`, or `schema_outdated:<version>` |
  | `signing` | `valid`, `invalid` (the API and database keys differ), `missing` (none in the database), or `not_configured` (none in the API) |
  | `retention_job` | `scheduled` or `not_scheduled`. Reported, not required; see Retention |

  The database probe is cached for 30 seconds per instance.
- Every response carries `X-Request-ID`. Logs hold method, path, status, latency, and that ID, never
  request bodies, authorization headers, keys, or journal text.

## Deploy

1. Rotate any key that has appeared outside a secret store.
2. Run the checks listed in the README's **Test** section. CI runs them on every push.
3. Apply `supabase/migrations/` in filename order. `scripts/verify_postgres_schema.py` rebuilds a
   scratch database from the migrations and checks RLS, provenance signing, lifecycle races, retention,
   the rate limit, and deletion.
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
   - **Render (alternative):** create a Blueprint from `render.yaml`; it asks for the secret values.
     `Dockerfile.api` builds the site and serves it from FastAPI.
7. In Supabase, under **Authentication → URL configuration**, set the Site URL to the public address and
   add `https://<address>/**` as a redirect URL. Without this, sign-in links return to localhost.
8. Confirm `/ready` reports `database: reachable`, `schema: schema_ready`, `signing: valid`, and
   `retention_job: scheduled`.
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
- Check recent runs: `select status, return_message, start_time from cron.job_run_details
  order by start_time desc limit 10;`
- If the job fails or is missing, `/ready` shows `retention_job: not_scheduled` and chats are still swept
  for each person when they next use the chat. Text in chats of people who never return stays until the
  job runs again. It can be run by hand as the postgres role: `select public.jp_purge_expired_conversations();`
- Chats with **Keep my messages** on are never purged automatically.

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
