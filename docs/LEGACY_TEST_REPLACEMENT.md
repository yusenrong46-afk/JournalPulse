# Legacy Test Replacement

The `v0.4-demo` tag preserves 42 passing tests for the retired classifier and Streamlit product. They are
not silently carried forward because their central assumptions no longer exist.

| Retired coverage | Research-beta replacement |
|---|---|
| TensorFlow/LSTM and transformer artifact smoke tests | OpenRouter strict-schema, ZDR, fallback, and consent tests |
| Six-label preprocessing and phrase chips | Bounded affective-state validation and user-correction flow |
| Heuristic recommendation blending | Approved-catalog validation and deterministic policy tests |
| Finite-state coaching | Five-step Playwright guided-reflection flow |
| SQLite journal CRUD | User-isolated CRUD, export, bulk deletion, and normalized Supabase request tests |
| Generic crisis fallback | Safety precedence, negation, locale, and exploration-shutdown tests |
| Streamlit consumer rendering | Next.js mobile/desktop E2E and mobile visual baseline |
| Training dry-run configuration | Research artifact registry that rejects fabricated evidence |

Coverage count is not treated as equivalence. The release gate is behavior, branch coverage, security
boundaries, and browser-flow coverage for the new architecture. The historical tests remain executable by
checking out `v0.4-demo`.
