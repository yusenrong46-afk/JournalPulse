# GitHub handoff checks — October 5, 2026

The source, tests, migrations, evaluation evidence and Claude handoff accompany the
current preview release. Seven existing Phase A commits are retained in the branch
history; GitHub main was their ancestor when inspected. A normal fast-forward push
publishes the integrated state without rewriting existing history.

Fresh local checks before the push:

| Check | Result |
| --- | --- |
| Backend suite | 703 passed; one upstream Starlette/httpx TestClient deprecation warning |
| Coverage with branch measurement | 90.90%, above the 80% floor |
| Frontend unit suite | 108 passed across 20 files |
| Ruff and mypy | Passed |
| OpenAPI and generated TypeScript | Current; regeneration produced identical bytes |
| Resource validation | 35 resources, 3 support resources, zero errors |
| ESLint and TypeScript | Passed |
| Node 22 production build | Passed; all 18 static pages generated |
| Evaluated runtime identity | All 33 frozen runtime/resource files match the `.3` benchmark |
| Main benchmark evidence | 664 manifest file hashes verified |
| Preview release evidence | 32 manifest file hashes verified |
| Credential and unwanted-file scan | No credential-shaped token findings after correcting substring false positives in native encrypted reasoning data; database and environment files excluded |
| Git source whitespace checks | Passed outside immutable historical evidence and the already-applied preference migration; original log/patch whitespace and SQL bytes are preserved |

Desktop/mobile browser, local full-stack/PostgreSQL and hosted smoke results are
recorded in the audit and preview release documents; they were not rerun as a new
full browser/database suite for this push. This file does not claim GitHub Actions
passed: remote CI is separate from these completed local checks.

Vercel's project has no Git repository connection, so this push does not initiate
a Vercel production deployment. The current preview and shared migration are
recorded in [preview release evidence](LUNA_PREVIEW_RELEASE_2026-10-05.md).
The uncertain formal safety benchmark grade remains unchanged.

[Claude's complete task prompt](CLAUDE_HANDOFF.md) covers the audit, repairs,
small UI slices, code differences, validation, and known remaining issues.
