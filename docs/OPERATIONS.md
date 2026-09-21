# JournalPulse Operations Runbook

This runbook covers the research-beta foundation. It does not claim clinical safety, therapeutic effect,
or production-scale availability.

## Runtime contract

- `/health` proves only that the API process can answer.
- `/ready` checks configuration, the curated resource catalog, AI configuration, and persistence mode.
- Production readiness requires Supabase, an explicitly configured non-local CORS origin, and a rotated
  OpenRouter key whenever AI processing is enabled.
- `scripts/verify_openrouter.py` performs the separately controlled live schema-constrained provider call.
- Every HTTP response receives `X-Request-ID`. Logs contain method, path, status, latency, and that ID;
  they must never contain request bodies, authorization headers, credentials, or journal text.

## Deployment sequence

1. Rotate any credential exposed outside the deployment secret store.
2. Run backend tests, frontend unit tests, generated-contract checks, Playwright, static export, and resource
   validation.
3. Build `Dockerfile.api`. Its first stage exports the Next.js PWA; the final Python image serves the UI
   and API from one origin with gzip compression.
4. For a personal preview, deploy `render.yaml`, confirm the persistent disk is mounted at `/var/data`,
   and verify `/ready` reports `server_sqlite` with AI explicitly disabled.
5. For a multi-user beta, apply Supabase migrations in filename order and confirm both atomic write
   functions are executable only by `authenticated` users.
6. Add Supabase and rotated OpenRouter values through Render secrets, change the environment to
   `production`, enable AI, and require every `/ready` dependency to be ready.
7. Complete one disposable-user reflection, outcome, export, single deletion, and bulk deletion.

The `preview` identity is a browser-generated UUID carried in `X-JournalPulse-User`. It prevents normal
browsers from sharing a timeline, but it is intentionally not treated as secure authentication. Never
invite external testers until Supabase Auth and RLS are enabled.

## Failure behavior

| Failure | Required behavior |
|---|---|
| OpenRouter timeout or transient 5xx | Retry at most the configured bounded attempts, then use the deterministic fallback |
| Invalid AI schema | Reject the model output and use deterministic fallback; never partially trust fields |
| Supabase Auth unavailable | Return 503 without accepting a development identity |
| Expired browser session | Refresh once, then return to private sign-in without discarding an opted-in draft |
| Lost response after a write | Retry with the same client request UUID and return the original record |
| Oversized body | Reject before model or persistence work with 413 |
| Analysis-rate limit | Return 429 and `Retry-After`; ordinary history and privacy operations remain available |
| Browser offline | Keep writing on-page, show offline status, and never claim the entry was saved |
| First PWA load | Cache only the static offline page; never prefetch every route or cache API responses |

## Data recovery and deletion

- Supabase backups and point-in-time recovery must be enabled and tested according to the selected plan.
- Before beta promotion, restore a backup into a non-production project and verify record counts plus RLS.
- `DELETE /v1/reflections/{id}` removes its dependent decision, outcome, observations, model run, and safety
  event through foreign-key cascades.
- `DELETE /v1/account/data` removes user-owned journal records but not the Supabase Auth identity.
- The Privacy screen also removes the encrypted local draft. Account-identity deletion remains a separate
  privileged operation and must not be represented as complete until implemented and tested.

## Rollback

- Keep the previous API image and PWA deployment available for immediate rollback.
- Never roll application code back across an incompatible migration. Add a forward repair migration instead.
- Disable AI with `JOURNALPULSE_LLM_ENABLED=false` if provider behavior is unsafe or unstable.
- Adaptive policy and memory flags remain independent kill switches and must stay off until their evidence
  gates are satisfied.
