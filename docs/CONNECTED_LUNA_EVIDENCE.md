# Connected Luna implementation evidence

Recorded 2026-10-04 in this cloud workspace. Three implementation agents used
Ultra reasoning concurrently; functional verification proceeded slice 1, then
slice 2, then slice 3. Existing uncommitted Phase A work was preserved. This is
local implementation evidence, not a new hosted release.

Subsequent hosted deployment and real-provider evidence are recorded separately
in `CONNECTED_LUNA_PREVIEW.md`; the local-only statements below describe the
earlier implementation checkpoint.

## Sequential slice gates

| Slice | Focused checks | Real browser integration |
| --- | --- | --- |
| Saved writing and reflection | 24 backend tests; 17 new PostgreSQL assertions | Save, export exact writing, reload, consent, reflect, delete: 1 passed |
| Selected-entry chat | 19 backend tests; 7 mounted frontend tests | Start with visible source, reload, invalidate after source deletion: 1 passed |
| Discovery and refinement | 62 backend/config tests; 6 frontend tests | Approve topic, inspect results, refine while preserving goal and excluding earlier links: 1 passed |

Focused counts overlap the full regression suites below and must not be added
to them to claim a larger number of distinct tests.

## Shared regression

- Backend: **234 passed**, **90.72%** measured coverage.
- Frontend unit tests: **31 passed** in 8 files.
- Existing PostgreSQL verification: **100 checks passed**, including actual
  concurrency checks; journal extension: **17 additional checks passed**.
- Existing browser integration: **8 passed**; the 3 new slice browser tests above
  passed separately in order.
- Mobile/desktop browser and automated accessibility: **42 passed**, including
  the new `/journal` and `/discover` routes.
- Ruff, mypy, frontend lint/types, generated OpenAPI consistency, resource
  catalog validation, and production static export passed. Catalog: 35 resources,
  3 support records, 0 errors. `git diff --check` passed.

Browser integration exercised real FastAPI, PostgREST, PostgreSQL migrations,
and the exported UI, using deterministic AI/search and auth-issuer substitutes.
General browser/accessibility checks use intercepted API fixtures. This machine
used system Chromium rather than Playwright's downloaded pinned browser.
Local scheduler diagnostics use an explicitly labelled cron metadata substitute;
this run did not verify hosted scheduler execution or email delivery. Local CI
commands passed; no new remote CI run was triggered.

## Failures investigated and corrected

- The saved baseline had no standalone journal endpoint: the required save
  returned 404 instead of 201.
- A real browser test found successful deletion left the saved-entry pane visible.
  The UI now clears the entry and its consent before navigating; the original
  assertion was retained and passed on rerun.
- A stale Next.js font cache caused build errors. Clearing only ignored
  `web/.next` restored the build; TLS verification was preserved.
- New test fixtures initially omitted required safety metadata or the new empty
  export field, and a discovery locator omitted its visible label. Fixtures and
  locator expectations were corrected; functional assertions remain enforced.
- A final chat label polish was rechecked with 7 mounted frontend tests, the
  slice 2 real browser test, and 2 mobile/desktop accessibility checks. Its new
  exact-link assertion initially omitted Next.js's configured trailing slash;
  the expected route was corrected while retaining the exact source UUID check.

Original failure and final passing logs are retained in
`/workspace/journalpulse-planning/connected-luna/` alongside mobile screenshots.

## Remaining live evidence

The new migration `202610040003_journal_entries.sql` has been verified locally
only. The previously applied Phase A migrations were not edited or replayed.
No new deployment, hosted migration, paid model call, or live Brave search was
performed. Local provider credentials are absent and the cloud's newly proposed
provider network rules have not been activated.

Brave's official public reference source was inspected at commit
`a75d7eff34c260c7bba3bdc0163fd6982c34e370` to confirm the endpoint, header,
parameters, and result shape. That source inspection is not a successful live API
call. Discovery reads snippets, not full pages; it does not establish source
accuracy, relevance, or clinical benefit.

Follow `CONNECTED_LUNA_TEST_GUIDE.md` for the next preview and human acceptance
sequence. Live checks still need provider configuration, a declared budget,
reply/source quality assessment, and latency/cost measurements.
