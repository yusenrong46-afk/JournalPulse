# Operations Runbook

This runbook covers running JournalPulse for a small beta. It makes no claim of clinical safety,
therapeutic effect, or large-scale availability.

## Runtime checks

- `/health` only proves the API process answers.
- `/ready` checks configuration, the catalog, the model, and storage, and returns 503 when any is not
  ready. In production it requires Supabase, a model key when AI is on, and a non-local CORS origin.
- Every response carries `X-Request-ID`. Logs hold method, path, status, latency, and that ID, never
  request bodies, authorization headers, keys, or journal text.

## Deploy

1. Rotate any key that has appeared outside a secret store.
2. Run the checks listed in the README's **Test** section. CI runs them on every push.
3. Apply `supabase/migrations/` in filename order. `scripts/verify_postgres_schema.py` rebuilds a
   scratch database from the migrations and checks RLS and the write functions.
4. Set the production variables on the host: `JOURNALPULSE_ENV=production`, `SUPABASE_URL`,
   `SUPABASE_ANON_KEY`, `JOURNALPULSE_LLM_API_KEY`, `JOURNALPULSE_LLM_ENABLED=true`, and the model names.
5. Deploy:
   - **Vercel (production):** also set `JOURNALPULSE_WEB_DIST=web-dist`, then `vercel deploy --prod`.
     The build runs `scripts/build_vercel_web.py`, which exports the site with the Supabase values baked
     in.
   - **Render (alternative):** create a Blueprint from `render.yaml`; it asks for the two secret values.
     `Dockerfile.api` builds the site and serves it from FastAPI.
6. In Supabase, under **Authentication → URL configuration**, set the Site URL to the public address and
   add `https://<address>/**` as a redirect URL. Without this, sign-in links return to localhost.
7. Confirm `/ready` reports `persistence: supabase` and a configured model.
8. Smoke test with a disposable account: sign in, chat to a saved step, check in, view Journey, export,
   then delete the test entry.

`scripts/verify_openrouter.py` and `scripts/verify_conversation.py` make paid live calls. Run them only
when a live check is intended.

## Failure behaviour

| Failure | Behaviour |
|---|---|
| Model timeout or transient 5xx | Retry within the configured attempts, then fail the turn with 502 or 503. Nothing is saved and the browser keeps the text for "Try again" |
| Model reply breaks the schema, is cut off, or names an unknown feeling | Reject the whole reply; save nothing for that turn |
| Safety check matches | Support mode for the rest of the chat; the model is never called |
| Supabase Auth unavailable | Return 503; never fall back to a development identity |
| Expired session | Refresh once, then send the person to sign in |
| Response lost after a write | The browser retries with the same client UUID and gets the original record |
| Oversized request | 413 before any model or database work |
| Rate limit | 429 with `Retry-After`; reading history and privacy actions still work |
| Chat idle for 24 hours | Closed on the person's next chat request, clearing text unless they chose to keep it |
| Device offline | A banner says nothing new will be sent; the offline page is the only cached page |

## Data recovery and deletion

- Enable and test Supabase backups and point-in-time recovery for the chosen plan. Before inviting more
  people, restore a backup into a separate project and check record counts and RLS.
- `DELETE /v1/reflections/{id}` removes the decision, outcome, observation, model run, and safety event
  through foreign-key cascades.
- `DELETE /v1/account/data` removes every user-owned record and keeps the sign-in identity. Deleting the
  identity is a separate privileged task and must not be described as done until it is built and tested.

## Rollback

- Keep the previous deployment available. On Vercel, promote the last good deployment.
- Never roll code back across an incompatible migration; add a forward repair migration instead.
- Set `JOURNALPULSE_LLM_ENABLED=false` to stop all model calls. Chats continue with Simple Luna.
- The adaptive-policy and memory flags are independent kill switches and stay off until their research
  gates pass.
