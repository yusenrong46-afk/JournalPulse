# Exercises

Three small, stand-alone exercises on ideas this codebase depends on. Each folder has a
stub (`*.py`), failing tests (`test_*.py`) and `HINTS.md`. No solutions are included, and
nothing here imports the app: you can do them without reading the rest of the repository,
then compare with the real code that each `HINTS.md` names.

```bash
uv run pytest -o addopts="" exercises/01_command_receipt    # one exercise
uv run pytest -o addopts="" exercises                       # all three
```

| Exercise | Idea | Real code |
|---|---|---|
| [01_command_receipt](01_command_receipt/) | A retried command is applied once; a reply for an older turn is ignored | `src/journalpulse/persistence.py` (`commit_turn`) |
| [02_gate_order](02_gate_order/) | Safety, consent and the rate limit all run before the model can be called | `src/journalpulse/conversations.py` (`_take_turn`) |
| [03_owner_scope](03_owner_scope/) | A store bound to one person hides everyone else's rows, with no tell-tale 403 | `src/journalpulse/conversations.py` (`require_owned`) |

`make check-exercises` (part of `make gate-l5`) checks that every test **fails** against the
stubs as shipped, so a solution can never be committed by accident.
