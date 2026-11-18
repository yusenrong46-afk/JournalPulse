# Hints: exercise 2

1. The tests pass a model that raises if called. Your job is to make sure every early return
   happens before that line can run.
2. Order matters for the rate limit too: someone in distress, or someone who has not consented
   to AI, should not use up their allowance.
3. "Fails closed" means an error in a safety mechanism leads to the safe outcome (no model call),
   not to skipping the mechanism. Catch `LimiterUnavailable` around `take()` only.
4. Keep `answer` a straight line of guard clauses; it is easier to read the order that way.

Compare with the real thing: `src/journalpulse/conversations.py`, `_take_turn`, runs
`assess_safety` and commits a fixed support reply first; consent, demo identity and provider
checks come next; then `enforce_generation_limit` in `src/journalpulse/api.py`, which returns
503 when the shared counter is unavailable; only then the provider. Tests such as
`test_support_mode_never_calls_the_model_and_stays_there` use a provider that records calls.
