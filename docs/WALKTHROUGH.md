# Code walkthrough: ten questions

Short answers for a reviewer, each with the file and symbol to open. `make check-walkthrough`
fails if a question is missing, an answer reaches 120 words, or a pointer no longer resolves.

## Q1. What happens when a late or duplicate AI reply arrives after a newer save?

It is refused or replayed, never applied twice. Each chat has an incarnation and a revision:
a turn for an old incarnation, or one that finds the chat at a different revision, raises
`ConversationStale` and returns 409 before anything is written. A retried `client_message_id`
returns the stored turn without calling the model again; the same id with different text is
refused. Activity commands are keyed by a receipt (user plus request id): a repeat returns the
session's current row, a changed payload gets 409. Tests:
`tests/test_conversations_api.py::test_retried_message_id_does_not_call_the_model_again`,
`tests/test_activity_sessions_repository.py::test_delayed_followup_cannot_commit_into_recreated_ids`.

Where: `src/journalpulse/persistence.py` → `commit_turn`, `src/journalpulse/persistence.py` → `require_same_turn_request`, `supabase/migrations/202610060001_sentinel_boundaries.sql` → `jp_commit_turn`

## Q2. Where is the consent and safety gate, and what are its limits?

In `_take_turn`, before any provider call. Order: ownership (404), incarnation and replay,
closed chat, message limit, then `assess_safety` on the message and linked entry. A match
commits a fixed support reply and keeps the chat in support mode; the model is never called.
Only after that do consent, demo identity, provider and zero-data-retention checks run, then
the shared rate limit, then the model. Limits: the safety check is English phrase patterns,
not a classifier. On its 34-case development set (labelled NOT_QUALIFIED, not held out) it
passes 29, misses 3 and leaves 2 for review.

Where: `src/journalpulse/conversations.py` → `_take_turn`, `src/journalpulse/safety.py` → `assess_safety`, `src/journalpulse/safety.py` → `HIGH_RISK_PATTERNS`

## Q3. How is it enforced that one person can never see another's data?

Twice. In the API, every query filters on the signed-in user, and "someone else's" and
"missing" return the same 404, so ownership is never confirmed. In PostgreSQL, row-level
security requires `auth.uid() = user_id`; signed-in users keep only SELECT and DELETE on
their tables, and every write goes through security-definer functions that use `auth.uid()`.
The API's own Supabase client signs each write for one user, and the database checks that
user against the token. Tests: `tests/test_auth_and_rls.py::test_migration_enables_rls_and_owner_policy_for_every_user_table`
and the RLS checks in `scripts/verify_postgres_schema.py`.

Where: `src/journalpulse/conversations.py` → `require_owned`, `supabase/migrations/202609240001_conversations.sql` → `conversations_owner`, `src/journalpulse/activity_sessions.py` → `owned_chat`

## Q4. What did the 60-scenario benchmark show, and why was the release held?

It compared the guided-action prompt `2026-10-05.3` with its predecessor on 60 fictional
scenarios (36 development, 24 held out), graded by a model, not people. On 20 eligible
held-out pairs: 8 wins, 5 losses, 7 ties, mean gain +0.35 against a +0.30 target, but the
approximate 95% interval (-0.13 to 0.83) includes zero. It was held because the hard
safety-privacy-lifecycle check came out "uncertain" on a chest-pain case, and the release gate
passes only when every check passes. `make eval` recomputes these figures offline.

Where: `src/journalpulse/guided_evaluation.py` → `_paired_summary`, `src/journalpulse/guided_evaluation.py` → `build_comparison`, `scripts/portfolio_eval.py` → `benchmark_part`

## Q5. What does "Delete all my data" remove?

Under a per-account lock it raises the account's erasure revision, then deletes journal
entries, conversations and messages, reflections, outcomes, activity sessions and receipts,
safety events, model-run records, memories, consents and profile. It keeps identity-only
tombstones (a deleted id cannot be reused), the revision number, rate-limit events (deleting
does not reset the limit) and the sign-in account. Every POST compares
`X-JournalPulse-Data-Revision` with the stored revision, so an old tab cannot write into the
erased account; signed writes check it again in the database. Test:
`tests/test_erasure_replay.py::test_request_already_waiting_in_server_auth_cannot_save_after_erasure`.

Where: `supabase/migrations/202610060002_erasure_boundaries.sql` → `delete_my_journalpulse_data`, `src/journalpulse/api.py` → `auth_dependency`

## Q6. How are writes protected from forgery?

On the Supabase path the API signs every write payload with HMAC-SHA256 over sorted, compact
JSON that includes its purpose, user id and issue time. The database recomputes the signature
with the key in `private.server_secrets` (or the previous key until its deadline), and checks
the purpose, that the user is `auth.uid()`, and that the payload was issued within the last 15
minutes. A browser holding only its own token therefore cannot call the write functions
directly. Without a key the API refuses to write (503) rather than writing unsigned.

Where: `src/journalpulse/signing.py` → `signed_payload`, `supabase/migrations/202610140001_signing_key_overlap.sql` → `signature_valid`

## Q7. How long is chat text kept?

Chats idle for more than 24 hours are closed, and closing clears message text unless the
person chose to keep that chat. A scheduled database job runs the purge every 15 minutes where
pg_cron is available and records its last success, which `/ready` reports as retention health;
every request also sweeps the caller's own stale chats. Reflection text follows
`raw_text_retention_default`, which is off. Test:
`tests/test_phase_a_lifecycle.py::test_scheduled_purge_closes_idle_chats_and_keeps_retained_text`.

Where: `supabase/migrations/202609280001_phase_a_integrity.sql` → `purge_conversations`, `supabase/migrations/202610040002_retention_diagnostics.sql` → `jp_purge_expired_conversations`

## Q8. How does the rate limit behave when its storage is down?

It fails closed. A per-process sliding window catches bursts, then a counter shared by every
instance (an advisory-locked table in PostgreSQL) decides. Over the limit returns 429 with
`Retry-After`. If the shared counter cannot be reached the request gets 503 and the model is
never called. Only AI and search paths are limited; reading history and privacy actions are
not. Tests: `tests/test_phase_a_lifecycle.py::test_rate_limit_fails_closed_when_the_shared_counter_is_unavailable`
and `::test_rate_limit_is_shared_by_every_instance`.

Where: `src/journalpulse/api.py` → `enforce_generation_limit`, `supabase/migrations/202610040004_audit_integrity.sql` → `consume_rate_limit`

## Q9. How does Simple mode work with no AI?

A chat starts in guided mode when consent is off, the identity is a demo, or no provider is
configured. Guided replies are fixed text chosen by simple keyword and pattern rules; a goal
tap builds a card from the reviewed activity catalog. Each reply is recorded as model
`luna-guided`, provider `local`, so evaluation can tell it apart. No rate limit applies and no
provider is constructed. Test:
`tests/test_conversations_api.py::test_guided_luna_runs_without_consent_or_a_model`.

Where: `src/journalpulse/guided.py` → `guided_completion`, `src/journalpulse/guided.py` → `guided_model_run`, `src/journalpulse/conversations.py` → `start_conversation`

## Q10. How do you know which prompt produced a reply?

Every AI reply's model-run record stores the prompt version, the skill version and the SHA-256
of the packaged skill file, and each conversation stores the prompt version it started with.
The skill is loaded with size and front-matter checks. One honest wrinkle: the runtime prompt
constant is `guided-action-2026-10-06.2` while the packaged skill file says `2026-10-06.1`;
both are recorded, and the 60-scenario benchmark tested `2026-10-05.3`, so it does not
qualify the current runtime. Test:
`tests/test_guided_action_prompt.py::test_runtime_skill_is_packaged_hashed_and_present_in_the_actual_provider_request`.

Where: `src/journalpulse/guided_action.py` → `load_guided_action_skill`, `src/journalpulse/guided_action.py` → `GUIDED_ACTION_PROMPT_VERSION`
