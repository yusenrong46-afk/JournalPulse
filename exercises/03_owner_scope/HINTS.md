# Hints: exercise 3

1. Write one private helper that fetches a row only if it exists *and* belongs to the owner,
   and raises `NotFound` otherwise. Use it in both `get` and `delete`.
2. Raise the same message in both cases. A different message (or a 403 in an API) tells an
   attacker that an id exists, which is already a leak.
3. `list` should filter by owner, never filter later in the caller.
4. Ids come from `store.ids` (`next(self.store.ids)`), so they increase in save order.

Compare with the real thing: `src/journalpulse/conversations.py`, `require_owned`, turns both
"missing" and "someone else's" into the same 404, and the database enforces it again with
row-level security (`conversations_owner` in `supabase/migrations/202609240001_conversations.sql`).
