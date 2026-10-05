# Before and after: search fix

`search-contract-fix.patch` compares the prior frozen, locally implemented v2
candidate with the current v3 candidate: eleven changed source/test/document
files. `combined-upgrade.patch` contains the full integrated upgrade: 54 changed
files against the complete 202-file dirty audited workspace before the upgrade.
Neither patch is against clean Git HEAD. Inherited audit work is preserved.

1. Before: the provider accepted arbitrary search prose, while the application
   rejected words outside its public vocabulary. After: provider and Pydantic
   share twelve finite categories; the server compiles the actual public query.
   See `activity_resources.py`, `guided_action.py` and `activity_chat.py`.
2. Before: the packaged skill allowed broad search phrases. After: version .2
   explicitly selects only a category and never copies private wording into it.
   The actual wheel includes that exact skill.
3. Before: a reasonable walking proposal could become HTTP502 with no saved turn.
   After: all twelve categories and constraint combinations are checked; private
   prose remains rejected. An actual walking completion saves through HTTP200.
4. Before: the failed v2 heldout was the only walking evidence. After: that result
   is preserved, a fresh 24-family heldout was frozen before baseline capture and
   implementation, and exact production replays verify all fresh observations.
   The historical walking rerun is labeled regression, not fresh quality evidence.
5. Before: rate-limited teacher evaluation blocked full release. After: unchanged
   bounded retries and alternate same-model routing retain 16 fresh judgments,
   including 15/20 quality pairs. The benchmark remains incomplete. Four separate
   post-hoc paired follow-ups have eight actual Luna replies and no teacher calls;
   Chromium validates the two offline review reports. No scores are invented.

Original ordered implementation patches are copied into `original-step-diffs/`,
with their historical notes. They group changes for review and can depend on later
shared contracts; they are not independently tested build checkpoints. Current
validation covers the integrated source: 617 backend tests, 90.67% coverage with
branch measurement, 16 real-stack browser integrations, static/OpenAPI and wheel
checks. Fix and combined patches reverse-check; forward application of the fix
reconstructs all eleven changed files from the frozen prior candidate.

No commit, reset, push or merge was performed. No product deployment, alias change
or shared migration was performed. Temporary evaluation assets were removed.
