# Release Evidence

For the follow-on implementation measured locally on 2026-10-04, see
[Persistent preference and retention evidence](PHASE_A_PREFERENCE_EVIDENCE.md).
The production observations below are historical records from 2026-09-28; they have not been
reverified for the new code or migrations.

Phase A (reliability and data integrity), measured on 2026-09-28. These are engineering checks, not
claims that the product improves anyone's wellbeing.

## Verified

### Automated, in CI (GitHub Actions run 36434661714, pull request #8)

| Check | Result |
|---|---|
| Ruff, mypy | Passed |
| OpenAPI contract | Current |
| Backend tests (pytest) | 119 passed |
| Backend branch coverage | 89.51% (gate 80%; was 85% before Phase A) |
| Catalog | 35 resources, 0 validation errors |
| PostgreSQL 16 schema verification | Passed: 59 checks, including two real two-session races |
| Next.js lint, typecheck, production build | Passed |
| Frontend unit tests (Vitest) | 15 passed |
| Mocked browser tests (Playwright, phone and desktop, API intercepted in the browser) | 38 passed, including axe on every page |
| Integrated browser tests (Playwright → FastAPI → PostgREST 12 → PostgreSQL with all migrations) | 5 passed |

The integrated suite uses a deterministic model provider and a local token issuer in place of
OpenRouter and Supabase Auth. Everything else is the production code path.

### Manual, on production (journalpulse.vercel.app, 2026-09-28)

- Applied `202609280001_phase_a_integrity.sql` to the hosted Supabase database in one transaction.
  `pg_cron` was created and the `journalpulse-retention` job is scheduled every 15 minutes.
- Stored the signing key in the database and on Vercel, deployed, and confirmed `/ready` reports
  `database: reachable`, `schema: schema_ready`, `signing: valid`, `retention_job: scheduled`.
- With a real magic-link sign-in and the real `openai/gpt-6-luna` model: two chat turns, feelings
  corrected from Luna's suggestion (tired, anxious) to the person's own (tired, sad), a reload, accept,
  and a check-in. The saved record held `self_report_input: {feelings: [tired, sad], mood_score: 2}`,
  a derived state with no confidence, the goal, and the model name. The chat was closed with its text
  cleared and linked to the reflection.
- A late reply to that chat returned 409; a second accept with a new request ID returned 409; a direct
  insert into `model_runs` with the person's own token returned 403 from PostgreSQL.
- Export returned all eight tables. The test entry was deleted afterwards; existing entries were not
  touched.
- The scheduled purge has not yet been observed running on production; it has been exercised in the
  schema verification and the integrated suite.

## Fixed

| Defect | Failure it caused | Regression test |
|---|---|---|
| Turn commits were unconditional and guarded only by an in-process lock | A slow reply could reopen a closed chat, overwrite newer state, restore cleared text, or be attached after another instance's reply | `tests/test_phase_a_lifecycle.py` (close, delete, accept, second instance), schema checks "delayed reply after close is rejected" and the concurrent-close race |
| Accept was three separate writes | A failure could leave a reflection with an open chat, or a closed chat without its reflection; two accepts could both save | Lifecycle tests for retry, two instances, and an injected mid-accept failure; schema checks and the concurrent-accept race |
| `_support_text()` returned the first assistant message in the chat | After ordinary replies, every support-mode reply repeated Luna's first ordinary reply | `test_support_mode_never_reuses_an_earlier_ordinary_reply` |
| Confirmed feelings and the mood face lived only in the browser | A reload replaced the person's corrections with Luna's suggestions, and accept saved them | `test_confirmed_feelings_and_mood_survive_reload_and_decide_the_saved_state`, the browser reload tests in both suites |
| Button-derived states carried an invented `confidence` | Heuristic numbers looked like measured certainty | `tests/test_self_report.py`, `web/tests/unit/feelings.test.ts` |
| Export used single PostgREST requests | Rows past the server's page limit were silently missing | `test_export_reads_every_page_of_every_table` (1,203 and 1,777 rows) |
| Deletion skipped decisions, model runs, safety events, and observations without a reflection link | Rows could survive a journal deletion | Schema check "A deletion removes every A journal row" |
| Idle-chat cleanup ran only when the person returned | Open-chat text could outlive the 24 hours indefinitely | Schema checks for the purge and repair pass, the integrated purge test |
| The rate limiter was per instance | Each Vercel instance allowed its own quota | `test_rate_limit_is_shared_by_every_instance`, schema checks, the integrated limit test |
| Deleting a journal also cleared usage counters | Deleting and retrying reset the generation limit | Schema check "deleting a journal does not reset the generation limit" |
| Signed-in users could insert and update every table | A person could forge policy, model, and safety records, or edit a conversation's card or support state | Schema checks for direct inserts and updates, forged and misdirected signatures, the integrated provenance test |
| `/ready` reported ready when environment variables existed | A missing schema, unreachable database, or mismatched key still looked ready | `tests/test_readiness.py` |
| The browser used port 8000 for any localhost origin | The exported site served on another local port could not reach its own API | The integrated suite |

## Still open

- **Auth in the integrated suite is a stand-in.** Tokens come from a local issuer, not Supabase Auth, so
  magic-link delivery and token refresh are verified only manually.
- **The safety gate is a phrase list.** Indirect language such as "ending it all" is not detected; the
  tests record this gap rather than hide it.
- **The retention job depends on `pg_cron`.** If it stops, text in chats of people who never return
  persists until it runs again. `/ready` reports this but does not fail.
- **Sign-in identity deletion is not built.** Journal deletion keeps the Supabase Auth account.
- **Supabase backups and a test restore have not been verified.**
- **The signing key is a single shared secret.** Rotating it means updating the database and every
  deployment together; there is no overlap window.
- **Timestamps come from two clocks.** Turn times come from the API and idle expiry uses the database
  clock; large skew would shift the 24-hour window.
- **No model evaluation has been run against frozen cases.** `openai/gpt-6-luna` is the only model in use.
- **Adaptive policies and memory remain off** pending the [research track](RESEARCH_TRACK.md).
