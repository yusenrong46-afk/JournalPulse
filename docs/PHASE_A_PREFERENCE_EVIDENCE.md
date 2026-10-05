# Persistent preference and retention evidence

Measured locally on 2026-10-04. This is an engineering slice; no new NLP model was trained or evaluated.

## Source and scope

Implementation branch: `codex/persistent-conversation-preference`, in the existing isolated checkout.
The recorded base is `8f49fa63b06b79c2a4af82d2dd2f56c46ca3a02d`, the reliability candidate from
`cursor/phase-a-reliability-07c8`. Its seven commits above main
`e30095fee1f061eee14012bd76ca6c41505285c6` are inherited changes. This slice's patch is measured
against the reliability base, not against main. The changes are local and uncommitted.

The base was tested before implementation: 119 backend tests, 89.23% coverage, 15 frontend unit
tests, Ruff/mypy/OpenAPI, and the existing PostgreSQL checks passed. The resource catalog held 35
resources with zero validation errors. PR #8 was observed as draft; the public summary of Actions
run 36435590702 reported success for this candidate. Individual job logs were not inspected.
The older run number in `RELEASE_EVIDENCE.md` remains a historical record, not new evidence.

The product slice persists Just talk, withdraws ordinary cards, supports explicit action resumption,
and protects choice changes against retries and late replies. The operations slice distinguishes a
schedule from successful scheduled cleanup. Tests and comments explain revision, ownership,
idempotency, support precedence, and retention boundaries.

Launch preparation found that replacing the original readiness RPC would make the existing
production API report an outdated schema. The new API now uses additive `jp_readiness_v2`;
the legacy RPC preserves `phase-a-1`. A failing transport assertion was recorded before the
fix, and the database verifier checks both schema contracts and both signing probes. This
preserves readiness compatibility, not the older browser's support for explicit preferences.

## Checks and environment

| Check | Local result |
|---|---|
| Ruff; mypy | Passed; 15 source files checked by mypy |
| OpenAPI generation check | Current; preference and acceptance contracts regenerated |
| Backend pytest | 135 passed; latest launch preflight 89.89% coverage; existing 80% gate passed |
| Resource catalog | 35 resources, zero validation errors |
| PostgreSQL schema verifier | 100 passing behavioral checks, including legacy/preview readiness compatibility and three real two-session races |
| Frontend lint; typecheck | Passed |
| Frontend unit tests | 18 passed |
| Next.js production build/static export | Passed using normal Google font downloads |
| Browser/API-intercepted suite | 38 passed on phone and desktop Chromium, including automated accessibility |
| Browser → API → PostgREST → PostgreSQL suite | 8 passed in the final sequential run |
| Live model calls | Unrun; paid AI disabled on the preview |
| Hosted Vercel preview | Ready; real Supabase sessions and the preference/resumption/export/deletion flow passed |

## Hosted preview — October 4, 2026

At the user's request, the preview uses the existing Supabase project. Both new migrations were
applied together through the management API after checking that neither was present. Previous
definitions of the three replaced shared functions were saved for operator review; no journal
content or signing-key value was copied. The legacy readiness RPC still reports `phase-a-1`, and
the preview's versioned RPC reports `phase-a-2`.

Deployment `dpl_6swSRJfHcrUjd2DiGKboS2xpQmyp` is a preview of 97 uploaded source files from the
locally reviewed candidate. Its metadata records patch SHA-256
`c1a2c42bb8d2028caadf9b3ba704a15599136b23c1af9f63cef590f4fa6f773c`.
These hosted evidence notes were added afterward and are excluded from deployment inputs.
The production address continues pointing to deployment `dpl_57Rut4UoR24LsTRjsA7nEpiBQycU`.

The exact preview address was added to Supabase's redirect allowlist; the production Site URL
was preserved. Production's model flag was preserved while the preview flag was set to false.
Both sites reported ready. The preview reported valid signing, the new schema, AI disabled,
and healthy retention corroborated by a completed real cron run.

The mobile browser check used two disposable accounts with real Supabase password-issued
sessions and no mocked API/model responses. It passed Simple Luna onboarding, guided messages,
a reviewed card, cross-account read denial, Just talk/reload/later-message persistence, fresh
action resumption, stale acceptance rejection, exactly one saved reflection, export, temporary
text clearing, local model provenance, and journal deletion. Both temporary auth accounts were
removed. Their credentials stayed in memory, and the saved logs are redacted.

Chrome initially could not initialize its existing NSS trust database because the sandbox made
that directory read-only. A targeted write grant fixed initialization; TLS verification stayed
enabled and no new root certificate was imported. Early checks also caught premature clicking
before hydration and incorrect test assumptions about empty-message Send state and screen-reader
text. The corrected checks passed; those unsuccessful attempts were retained as diagnostic logs.

Vercel deployment protection remains enabled. Browser service workers were blocked in this
live check. Magic-link email delivery, real-model response quality, seeded aged-record cleanup,
and backup/restore were not exercised by this preview verification.

Python 3.12, uv 0.12.19, Node 22.23.3, PostgreSQL 16.15 with pgvector, PostgREST 12.2.12,
and system Chromium 151.0.7922.173 were used. Dependency lockfiles were preserved.
Playwright used an external configuration override selecting system Chromium; the pinned browser
distribution was not used. The repository's test configuration and assertions remain active.

PostgREST came from the versioned official GitHub release asset over verified TLS. The archive's
observed SHA-256 is `5de4092f1719da3353c40bf96c8dec6913f2254a7cd0b61cc05f233153b557d5`;
this is a recorded download fingerprint, not an independently verified publisher signature.
The scratch database image was `pgvector/pgvector:pg16` at digest
`sha256:7b822b0aac60967beb1ea5e576b8602c94c300a157d187f385ae3e0da199b90a`.
Only scratch databases `jp_verify` and `jp_integration` were reset; no hosted data was changed.

## Reproduce

From the repository root, with Python/Node dependencies installed from the lockfiles:

```bash
uv run ruff check src tests scripts
uv run mypy src
uv run python scripts/export_openapi.py --check
uv run pytest --cov=journalpulse --cov-report=term-missing
uv run python scripts/validate_resources.py
uv run python scripts/verify_postgres_schema.py
```

The schema and integration scripts need `psql`, PostgREST, and a PostgreSQL server with pgvector.
`JOURNALPULSE_PG_DSN` must point to a disposable server's maintenance database. These scripts rebuild
their named scratch databases. Local onboarding startup instructions describe the prepared server
on port 55432 and tools in `/workspace/.tools/bin`.

From `web`, with Node 22 active:

```bash
npm run lint
npm run typecheck
npm run test:unit
npm run build
npm run test:e2e
npm run test:integration
```

This cloud machine used `/workspace/.onboarding/playwright.config.ts` and
`/workspace/.onboarding/playwright.integration.config.ts` overrides with `npx playwright test
--config <path>`. Browser suites should run sequentially on this machine. One concurrent run produced
seven passes and one timeout waiting for the first guided turn: the trace showed a successful create
request of about 0.8 seconds followed by a successful turn of about 4.1 seconds, crossing the five-second
UI assertion budget under load. No assertion or application timeout was relaxed; the result and trace
diagnosis remain in the review package. The sequential rerun passed all eight tests in 42.8 seconds.

## What the tests establish

- API tests cover card withdrawal, later guided turns, trusted AI context, suppression of provider
  offer flags, explicit resumption, stale acceptance, receipt replay/conflicts, another owner's access,
  support precedence, close/delete/expiry, exports, and rollback after an injected receipt failure.
- Real PostgreSQL checks use payloads captured from the production Supabase adapter. They cover
  signed writes, RLS, preservation of canonical preference across older writers, acceptance revision
  requirements, receipt cascades, and a preference-versus-turn row-lock race.
- Integrated browser cases exercise a reload and fresh action acceptance, automatic and explicit
  retry after responses are deliberately lost, and delivery of an older HTTP response after a newer
  listening choice. Request interception in these fault cases delays/drops delivery after real writes;
  it does not replace the API or database.
- The integration model and authentication token issuer are deterministic stand-ins. The fake model
  observes the trusted listening instruction while deliberately retaining an offer flag, exercising
  the server's independent enforcement. This does not measure real-model language quality.
- Retention checks execute real PostgreSQL cleanup and diagnostic functions with controlled clock
  values and disposable cron metadata. They test missing/never-successful/failed/overdue/unknown/healthy
  states, recovery, manual/request-time cleanup separation, grants, and both 30-minute boundaries.
  The local pg_cron worker was unavailable; actual background scheduling is **unrun**. Installing
  pg_cron was blocked by package-repository access; no package verification was bypassed.
- Integrated export and deletion operate while local readiness reports missing scheduled retention.
  Retention diagnostics remain advisory; they do not gate privacy routes or disable the application.

Review was performed by the implementing assistant; no independent agent review or human QA is claimed.

## Manual review and release prerequisites

Use a disposable fictional chat:

1. Send two ordinary messages, open feelings/goal selection, and obtain a card.
2. Select Just talk. Check that the card disappears, no reflection is saved, and unsent text remains.
3. Reload, send another message, and confirm automatic action controls stay absent.
4. Select Find a small step, confirm feelings/goal, and accept the fresh card.
5. Export the data, inspect preference receipts, then delete the disposable journal.
6. Repeat without AI consent; check the listening continuation. Check saving/error wording and keyboard
   access. Human feedback should become a concrete follow-up issue.

Before releasing, apply both new migrations in order and ship the matching API/frontend pair; follow
`OPERATIONS.md`. Verify the deployed revision, real Supabase sign-in/refresh, a real scheduled purge
and its healthy diagnostics, failure/recovery observations, backups/test restore, and the disposable
user journey. These hosted checks remain outstanding. Real provider adherence needs separately
authorized live checks; this slice does not supply NLP benchmarks or evidence of wellbeing benefit.

Legacy `auto` chats retain acceptance without a client revision. After an explicit choice, old clients
without the revised acceptance contract receive conflicts. Do not roll back to an API/frontend that
cannot handle preference-aware chats; use a compatible deployment or a forward repair.
