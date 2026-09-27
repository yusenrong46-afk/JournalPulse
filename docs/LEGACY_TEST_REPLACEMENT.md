# Legacy Test Replacement

The `v0.4-demo` tag preserves the retired classifier and Streamlit product, with its tests. Those tests were
not carried forward because their central assumptions no longer exist.

| Retired coverage | Current replacement |
|---|---|
| TensorFlow/LSTM and transformer artifact smoke tests | OpenRouter strict-schema, ZDR, retry, and consent tests |
| Six-label preprocessing and phrase chips | Fixed feelings list, validated on the server and in the chat |
| Heuristic recommendation blending | Catalog validation and deterministic policy tests |
| Finite-state coaching | Chat tests: scripted Luna, goal cards, and the Playwright mood-to-saved-step flow |
| SQLite journal CRUD | User-isolated CRUD, export, bulk deletion, and Supabase request tests |
| Generic crisis fallback | Safety precedence, negation, locale, support-mode, and goal-turn safety tests |
| Streamlit consumer rendering | Next.js phone and desktop browser tests with axe accessibility checks |
| Training dry-run configuration | Research artifact registry that rejects fabricated evidence |

A matching test count is not treated as equivalence. The release gate is behaviour, branch coverage,
security boundaries, and browser-flow coverage for the current architecture. The historical tests still
run from the `v0.4-demo` tag.
