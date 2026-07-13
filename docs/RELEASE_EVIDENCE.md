# Research Beta Release Evidence

Measured locally on 2026-07-13 from the commit-ready working tree, Python 3.12.13, a compressed
single-origin static export, and Chromium. These results are reproducible engineering checks, not
product-effectiveness claims.

| Gate | Result |
|---|---|
| Backend tests | 30 passed |
| Backend coverage | 86.74% branch-aware coverage |
| Ruff / mypy | Passed |
| Curated catalog | 35 resources, 0 validation errors |
| npm audit | 0 known vulnerabilities |
| Next.js lint / typecheck / production build | Passed |
| Frontend unit tests | 4 passed |
| Playwright | 15 passed, 1 expected desktop-only visual skip |
| Lighthouse performance | 98 |
| Lighthouse accessibility | 100 |
| Lighthouse best practices / SEO | 100 / 100 |
| Mobile LCP | 2.409 seconds |
| CLS | 0.016 |
| Static deployment smoke | `/`, `/history/`, `/health`, and `/ready` served from FastAPI |

## Gates that remain open

- No live OpenRouter comparison is claimed. The previously exposed credential must be rotated before
  GPT, Qwen, and Gemini candidates are run against frozen structured-extraction cases.
- The migration and PostgREST contracts are tested, but the SQL has not yet been exercised against a
  connected Supabase project in this release run.
- The Render image is configured for a persistent preview deployment, but a paid Render disk and account
  authorization are external deployment prerequisites.
- Adaptive policy and episodic memory are disabled. There are no simulator, regret, off-policy, retrieval,
  or self-study results yet.
- No therapeutic, diagnostic, clinical-effectiveness, or personalization claim is supported.

## Functionality verified locally

- First-run processing preferences persist locally and remain overridable per reflection.
- A person can correct affective dimensions and tags before any decision is stored.
- The baseline recommendation and two safe alternatives come only from the reviewed catalog.
- User overrides preserve the original recommendation and are excluded from off-policy evaluation.
- Today surfaces unfinished decisions; standalone check-in captures post-state, helpfulness, and effort.
- Duplicate outcomes and cross-user outcome attachment are rejected.
- History search, loop-status filters, and descriptive trajectory views operate on stored observations.
