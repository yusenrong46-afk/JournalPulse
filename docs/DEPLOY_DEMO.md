# Deploying the portfolio demo

The demo lets a visitor press **Try the demo** on the sign-in screen and use JournalPulse
without an email address. It is **off by default** (`JOURNALPULSE_DEMO_ENABLED=false`), and
nothing in this repository turns it on for a hosted environment. The steps below are owner
steps: they change hosted settings and the production database.

## What the demo guarantees (enforced in code, tested)

| Guarantee | Where | Test |
| --- | --- | --- |
| The route `POST /v1/demo/sessions` exists only when the flag is on | `src/journalpulse/demo.py` | `tests/test_demo_sandbox.py::test_demo_route_is_absent_unless_enabled` |
| Hosted demo identities are Supabase anonymous users; row-level security applies as for any account | `src/journalpulse/auth.py` (`is_anonymous` claim) | `scripts/verify_demo_sandbox.py`, `web/tests/integration/demo-sandbox.spec.ts` |
| A demo identity never causes a model or search request; asking for AI gives Simple mode | `conversations.py`, `journals.py`, `activity_sessions.py`, `discovery.py`, `inline_discovery.py`, `api.py` (`auth.demo`) | `tests/test_demo_sandbox.py` (a provider double fails the test if called) |
| Three fictional entries and one completed Simple Luna activity are seeded through the public routes | `demo.py::seed_demo_account` | `test_demo_session_is_seeded_with_three_entries_and_one_completed_simple_loop` |
| Creation is rate-limited per client address (default 20 per hour per instance) | `demo.py::register_demo_routes` | `test_demo_creation_is_rate_limited` |
| Demo data is erased after 24 hours through `public.delete_my_journalpulse_data()` | `supabase/migrations/202611180001_demo_sandbox.sql` | `scripts/verify_demo_sandbox.py` |
| Every demo view says "Demo: fictional data, AI off" and keeps the 9-8-8 line visible | `web/components/demo-banner.tsx` | `web/tests/unit/demo-sandbox.test.tsx`, `make demo-check` |

The per-instance limiter does not share state between serverless instances, so the hosted
limit that matters is Supabase's own anonymous sign-in rate limit (step 2).

## Owner steps (about 20 minutes)

1. **Merge** the upgrade pull request after review. Deploy only from `main`.
2. **Supabase → Authentication → Sign In / Providers:** turn on *Allow anonymous sign-ins*.
   Then **Authentication → Rate Limits:** set *anonymous users* to a small number per hour
   (for example 30). Anonymous users get the `authenticated` role, so the existing RLS policies
   apply to them unchanged.
3. **Apply the new migration** (additive; it changes no earlier migration). With the Supabase CLI
   linked to the project, from the repository root:

   ```bash
   supabase db push
   ```

   Or paste `supabase/migrations/202611180001_demo_sandbox.sql` into the SQL editor and run it once.
   Confirm the job is scheduled:

   ```sql
   select jobname, schedule, active from cron.job where jobname = 'journalpulse-demo-erasure';
   ```

   If `pg_cron` is not enabled, enable **Database → Extensions → pg_cron** and rerun the `do $$ … $$`
   block at the end of that migration.
4. **Vercel → Project → Settings → Environment Variables (Production):**

   | Name | Value |
   | --- | --- |
   | `JOURNALPULSE_DEMO_ENABLED` | `true` |
   | `NEXT_PUBLIC_JOURNALPULSE_DEMO_ENABLED` | `true` (optional: `scripts/build_vercel_web.py` copies the first flag) |
   | `JOURNALPULSE_DEMO_CREATION_LIMIT_PER_HOUR` | `20` (optional) |

   Then redeploy production (`vercel deploy --prod`), because the button is a build input.
5. **Check:** open the sign-in page in a private window, press **Try the demo**, and confirm the
   banner, the three fictional entries, and that a new chat says "Simple mode · no AI". Run one
   erasure by hand to see it work (it only touches anonymous accounts older than 24 hours):

   ```sql
   select private.jp_erase_expired_demo_accounts();
   ```

## Turning it off

Set `JOURNALPULSE_DEMO_ENABLED=false` (and the `NEXT_PUBLIC_` copy) and redeploy; the route
returns 404 and the button disappears. Leave the erasure job running until existing demo
accounts are gone, then optionally `select cron.unschedule('journalpulse-demo-erasure');`.

## Local demo

`make demo` serves everything on http://127.0.0.1:8000 with SQLite and local identities (a
`de300000-` prefixed UUID marks a demo identity); provider keys are blanked for that process.
`make demo-check` drives the journey and refuses any request that would leave the machine.
