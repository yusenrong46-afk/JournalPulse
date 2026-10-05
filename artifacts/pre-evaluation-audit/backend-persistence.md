# Backend persistence findings — 2026-10-05

Read current dirty source, including `docs/LUNA_SEARCH_CONTRACT_FIX.md`. Initial findings were reproduced without product changes against all current migrations in isolated local PostgreSQL, then the database was dropped. Root subsequently authorized repairs only in the never-applied candidate migration and its verifier; the repairs below are now implemented and validated. No provider or external application calls.

## Repair status

- `supabase/migrations/202610050001_guided_activity_sessions.sql:50` revokes individual authenticated message deletion. Supported full conversation RPC, signed source deletion, and account deletion still remove message metadata.
- The candidate's replacement `jp_change_preference` at line 393 clears both card fields and returns `UPDATE ... RETURNING record` (line 443). Signed request, owner, revision, and idempotency guards remain.
- The lifecycle trigger at line 373 withdraws only unstarted offered sessions when recommendation card, constraints, goal, or public search topic changes. It records `stopped`, not a participant `declined`, and preserves active/paused/awaiting-report sessions unless the existing explicit withdrawal/safety/privacy branch applies. The predicate matches the backend parent's SQLite repair.
- `scripts/verify_activity_schema.py:166`, `:217`, and `:267` add actual PostgreSQL regressions for preference response/read parity, partial deletion bounds/full privacy deletion, recommendation supersession, preservation of ongoing activities, and a completed session's pause follow-up stopping a second active timer. The pause follow-up also proves mutation response equals persisted state and remains idempotent.
- Validation: all **72 activity PostgreSQL assertions** pass; the pre-existing `verify_postgres_schema.behavior_sql()` suite also passes against the repaired candidate schema. Ruff and mypy pass for the modified verifier. Log: `persistence-repairs-postgres.log`; repeatable isolated driver: `verify_persistence_repairs.py`.
- The nine earlier migration files are unchanged, verified with `sha256sum --check immutable-migrations-before.sha256` (all nine OK). Only candidate SQL and `scripts/verify_activity_schema.py` were edited in the product checkout by this agent. Both specifically named audit databases were dropped; local PostgreSQL is free for integration.

The original observations below describe the pre-repair state and are retained as regression evidence. `repro_persistence.py` intentionally describes that state and now stops at the denied partial message deletion; use `verify_persistence_repairs.py` for the fixed behavior.

## 1. P2 — Direct message deletion resets the per-conversation safety bound

- Location: `supabase/migrations/202609280001_phase_a_integrity.sql:124` grants authenticated users `DELETE` on `conversation_messages`; the existing owner policy permits it. The application determines its 20-message bound from the surviving rows at `src/journalpulse/conversations.py:234`. Activity offer/start/final-follow-up guards use that same row count at `supabase/migrations/202610050001_guided_activity_sessions.sql:126`, `:177`, and `:264`.
- Trigger: Create a conversation and commit twenty ordinary signed turns. As the same authenticated owner, delete its `conversation_messages` directly through the exposed Supabase table endpoint (or equivalent SQL), retaining the parent conversation. The database accepts this deletion. Its user-message count falls from 20 to 0 without advancing the parent revision. The next API turn passes the 20-message guard; activity offers and the special final-follow-up guard also see a fresh count.
- Impact: An ordinary session JWT can bypass the documented conversation length/final-follow-up bounds. The separate per-minute generation limit remains effective; this finding does not assert cross-account access or bypass of that limiter.
- Repro evidence: `repro_persistence.py` prints `cap_before_direct_delete=20` and `cap_after_direct_delete=0`, using actual owner role and production adapter-signed turns.
- Proposed fix: Revoke direct message deletion and require lifecycle deletion of the whole owned conversation, or maintain an immutable parent counter which message deletion cannot reduce. Keep whole-conversation/account deletion available.
- Regression checks: At 20 committed turns, an authenticated table DELETE should fail, or the independent count must remain 20; the next ordinary API turn and new activity offer must remain blocked. Verify owner conversation/account deletion still cascades successfully.

## 2. P2 — PostgreSQL preference changes return an activity card removed by the trigger

- Location: `supabase/migrations/202610040001_conversation_preference.sql:66` builds `changed` by preserving `activity_card`; `:75` returns that pre-trigger object. `supabase/migrations/202610050001_guided_activity_sessions.sql:337` clears the stored card on Listen. SQLite explicitly clears both card fields at `src/journalpulse/persistence.py:1206`.
- Trigger: A conversation has a valid additive `activity_card`. Send a signed preference change to `listen`. The successful RPC/API response still contains the card, while a subsequent GET returns no card. A direct `act` preference request with an existing card also preserves that card in PostgreSQL, whereas SQLite clears it.
- Impact: Preference responses disagree with committed state and with the SQLite contract. A client treating the mutation response as authoritative retains a withdrawn recommendation until refreshed. The current Talk UI separately hides the activity workspace in Listen mode, so this is not evidence that that UI visibly displays the card after withdrawal.
- Repro evidence: `repro_persistence.py` prints `listen_response_has_activity_card=true`, `listen_stored_has_activity_card=false`, `act_response_has_activity_card=true`, `act_stored_has_activity_card=true`.
- Proposed fix: Add an additive SQL replacement that clears `activity_card` alongside `card` for each preference change and returns the actual updated row with `UPDATE ... RETURNING record`.
- Regression checks: Through real PostgreSQL and SQLite, assert mutation response equals subsequent read for both preferences, beginning with an activity recommendation. Verify retries return current state and existing activity invalidation still occurs.

Run the retained local-only repro with:

```sh
PATH=/workspace/.tools/bin:$PATH JOURNALPULSE_PG_DSN=postgresql://postgres@127.0.0.1:55432/postgres /workspace/JournalPulse/.venv/bin/python /workspace/journalpulse-planning/pre-evaluation-audit-2026-10-05/repro_persistence.py
```

The script uses the repository's existing local-DSN safeguards and creates/drops only its specifically named audit database.
