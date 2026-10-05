# F6: dependency advisories, rechecked 2026-10-05

- `npm audit` (all dependencies): 5 high findings. They are one chain:
  `eslint-config-next` → `@next/eslint-plugin-next` → `fast-glob` → `micromatch` → `braces`
  3.0.3 (GHSA-vfj7-8cjw-p6xm). Every package in the chain is marked `dev` in `package-lock.json`.
- `npm audit --omit=dev` (what the site ships): **0** findings. The site is a static export
  whose runtime dependencies are `next`, `react`, `react-dom` and `@supabase/supabase-js`.
- The npm registry has no `braces` release newer than 3.0.3, so no override can fix it. npm's
  only suggested "fix" is a major downgrade of `eslint-config-next` to 14.2.35, which was **not**
  applied. Exposure is limited to linting trusted repository files on developer and CI machines.
- Python (`pip-audit` on the frozen `uv export` with dev extras, 35 packages): no known
  vulnerabilities.
- No lockfile was changed. Recheck when `braces` or `eslint-config-next` publishes a release.

Raw output: `F6-npm-audit-all.json`, `F6-npm-audit-production.json`, `F6-pip-audit.json`.
