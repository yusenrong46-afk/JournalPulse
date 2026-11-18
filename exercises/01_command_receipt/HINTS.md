# Hints: exercise 1

1. Check the receipt *before* checking the turn. A retry of a command that was already applied
   will carry an old `expected_turn`, and it must still get its first receipt, not `StaleCommand`.
2. "Same command" means same text. Compare the stored receipt's text with the new command's.
3. Change the chat only after every check has passed, so a refused command leaves no trace.
4. A late reply is not an error: the person did nothing wrong, so return `False` quietly.
5. Does a stored reply advance the turn? Read the last test carefully before deciding.

Compare with the real thing: `src/journalpulse/persistence.py`, `commit_turn` (fences on the
chat's incarnation, replays a stored turn for a repeated `client_message_id`, then requires the
expected revision) and `require_same_turn_request` (refuses a reused id with different text).
