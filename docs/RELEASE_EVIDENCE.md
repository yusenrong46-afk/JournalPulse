# Release Evidence

Measured on 2026-09-27 for the Luna redesign. These are engineering checks, not claims that the product
improves anyone's wellbeing.

## Automated checks

| Check | Result |
|---|---|
| Backend tests (pytest) | 62 passed |
| Backend coverage | 85% (branch-aware; gate is 80%) |
| Ruff, mypy | Passed |
| OpenAPI contract | Current |
| Catalog | 35 resources, 0 validation errors |
| PostgreSQL 16 migration check | Runs in CI on every push |
| Next.js lint, typecheck, production build, static export | Passed |
| Frontend unit tests (Vitest) | 15 passed |
| Browser tests (Playwright, phone and desktop) | 34 passed, including axe on every page |

## Live checks

- **Supabase:** all four migrations applied to the hosted project. A disposable account completed sign-in
  by magic link, a two-turn chat, a saved step, a check-in, export, and deletion.
- **OpenRouter:** `openai/gpt-6-luna` accepted the strict chat schema, including the `feelings` list.
  Luna suggested feelings that matched the message and handed off to the catalog instead of inventing its
  own activity.
- **Production (journalpulse.vercel.app):** signed in, chatted with Smart Luna, saved a step, checked in,
  and viewed Journey. `/ready` reported configuration, catalog, model, and Supabase as ready. The test
  entry was deleted afterwards.

## Still open

- The rate limiter is in memory, so each Vercel instance counts separately. Move it into Postgres before a
  wider launch.
- Supabase backups and a test restore have not been verified.
- Deleting a person's sign-in identity is not built.
- No model comparison has been run against frozen cases; `openai/gpt-6-luna` is the only model in use.
- Adaptive policies and memory remain off pending the [research track](RESEARCH_TRACK.md).
