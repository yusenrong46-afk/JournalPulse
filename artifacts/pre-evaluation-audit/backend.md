# Backend review and targeted repairs — October 5, 2026

Read the current dirty source and `docs/LUNA_SEARCH_CONTRACT_FIX.md`. No paid model/search calls, deployments, commits, resets, or hosted writes. The inherited changes and earlier evaluation evidence are preserved. The SQL reviewer used and removed one isolated local scratch database; its evidence is in `backend-persistence.md` and `persistence-observations.json`.

## P2: expired follow-up claims never become retryable in the UI — repaired

An activity follow-up claims a 90-second generation lease before invoking the model. If the worker exits after that claim, the stored session remains `generating`. The repository permits takeover after lease expiry, but both activity GET routes previously returned `generating` indefinitely. The panel only exposes retry for `pending`/`failed`, so refreshing or waiting did not recover the saved check-in.

Relevant source: `src/journalpulse/activity_sessions.py:139` (session response); `src/journalpulse/persistence.py:718` (atomic claim); `web/components/activity-session-panel.tsx:119` (retry UI).

The repair projects an expired lease as `failed` in the response, using the server clock. It does not mutate the stored claim, trigger model generation on GET, or change revisions. An explicit retry still passes the existing owner/source checks, transaction, request identity, and three-attempt maximum.

`tests/test_activity_follow_up_recovery.py` deliberately abandons a real repository claim, then checks both API reads at 89 and 90 seconds, canonical storage equality, explicit takeover, duplicate replay, and the three-attempt limit. Before the repair: **2 failed, 1 passed**. After the repair: these tests plus existing lifecycle/boundary/chat/stop tests **40 passed**. Ruff and mypy on the changed source pass.

A separate exploratory repro used an injected client returning an invalid selected ID and observed an uncleared claim after HTTP 502. The production parser rejects that ID before the later validator, and its normal provider-error cleanup works. That injected-only case is **not** reported as a production parser bypass, and no cleanup change was made on its basis.

## P2: a superseded unstarted activity can still start with obsolete constraints — repaired

Source: `src/journalpulse/persistence.py:1386` invalidates sessions for support/pause but leaves an existing offered session on normal negotiation. `:633` starts it without comparing the saved descriptor with the latest recommendation/constraints. `web/components/activity-session-workspace.tsx:279` hides the new recommendation while any nonterminal session exists.

Actual API/production-parser repro, using only an `httpx.MockTransport`:

1. Luna proposes `guided_meditation_2m`, with `time_minutes=2`.
2. Save that offer without starting. This is the normal state after choosing a search resource, and also a recoverable state when create succeeds but start is interrupted.
3. User asks, “Actually I only have one minute. Please make it shorter.” The second accepted model completion selects `guided_meditation_1m`, with `time_minutes=1`.
4. GET still returns the old session as `offered`, now with current conversation revision 2. Starting that refreshed session returns `active`, old resource `guided_meditation_2m`, duration **120 seconds**.

Observed output: `production_parser_calls=2`, `new_recommendation=guided_meditation_1m`, `new_time_limit=1`, `old_session_after_negotiation=offered`, `started_resource=guided_meditation_2m`, `started_duration=120`, `start_conversation_revision=2`.

The SQLite repair atomically stops only unstarted offered sessions when card/recommendation, constraints, goal, or search topic changes. It covers ordinary turns, preference changes, and follow-up recommendations. Automatic withdrawal uses `stopped` rather than `declined`, so it is not misrecorded as participant rejection and does not permanently exclude a resource. Already active/paused/awaiting-report sessions keep their current controls.

`tests/test_activity_offer_supersession.py` uses the actual production parser with local mock transport. It verifies the old offer is stopped, both an old-revision start and a refreshed start on that withdrawn offer fail, and the new one-minute offer starts with a 60-second deadline. It separately verifies active, paused, and awaiting-report preservation. Before repair: **1 failed, 3 passed**. Matching candidate SQL now passes real PostgreSQL checks covering each changed recommendation field, unchanged offers, and active/paused/awaiting preservation. The frontend reviewer added a real-stack browser case for replacing a saved unstarted search offer.

## P2: follow-up stop did not stop another active session in SQLite — repaired

Actual-parser repro: report A remains pending; session B starts; A's follow-up returns valid `move=pause`. The conversation enters pause, but SQLite left B active with a timer deadline, while the existing PostgreSQL lifecycle trigger stopped B. The UI correctly disabled controls due to the conversation pause, leaving inconsistent authoritative session state.

`src/journalpulse/persistence.py:874` now invokes the existing lifecycle invalidation on an explicit transition to pause, matching PostgreSQL. Ordinary recommendation changes still preserve active sessions. The final API regression in `tests/test_activity_offer_supersession.py` failed before this repair and passes afterward; it asserts B stops/deadline clears/check-in stays suppressed, while A's saved report and reply remain intact. The real PostgreSQL test additionally confirms replay creates no duplicate reply.

Final focused Python verification: **55 lifecycle/preference tests passed**, plus an earlier **68 activity/integrity/adapter tests passed**. Ruff and mypy on both changed Python source files pass. Final actual PostgreSQL verification: **72 activity assertions passed**, and the existing legacy SQL behavior suite passed. The isolated scratch database was dropped; the server is free for the full integration run.

## Other confirmed persistence findings

The companion SQL review found two issues using actual authenticated roles and the current migration chain:

- Direct owner deletion of conversation messages resets the row-count-based 20-turn bound from 20 to 0. This is an enforcement bypass, not a cross-owner access claim. Preserve legitimate conversation/account deletion while removing or independently accounting for direct message deletion.
- PostgreSQL preference changes return a pre-trigger activity card on Listen, and preserve the card on Act where SQLite clears it. Return canonical updated state and clear both card fields consistently.

Both persistence findings are repaired in the unpublished candidate migration: direct authenticated partial message deletion is revoked while full-chat/source/account deletion remains verified, and preference updates clear both cards and return the updated row. See `backend-persistence.md` for precise migration locations and retained before/after evidence. All nine earlier migration hashes remain unchanged, recorded in `immutable-migrations-before.sha256`. Product files changed by this backend team: `src/journalpulse/activity_sessions.py`, `src/journalpulse/persistence.py`, `tests/test_activity_follow_up_recovery.py`, `tests/test_activity_offer_supersession.py`, `supabase/migrations/202610050001_guided_activity_sessions.sql`, and `scripts/verify_activity_schema.py`.
