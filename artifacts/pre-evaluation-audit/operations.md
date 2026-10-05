# Operations, packaging, and build audit — October 5, 2026

Scope: local source/configuration and retained release evidence. No deployment,
database migration, live provider call, dependency scan, commit, reset, or production
configuration change was performed. Source was initially inspected read-only; the
root agent then authorized the packaging, Render declaration, configuration, and
readiness fixes below.
Inherited changes and generated files remain on disk. The cloud onboarding setup
skill informed lockfile and artifact-integrity checks; this task did not change the
cloud environment configuration draft.

## Confirmed defects and security gaps

### OPS-01 — P2: generated code and test/evaluation inputs entered the upload

Before: `.vercelignore:1` and `vercel.json:7` excluded the root `tests/` and
`artifacts/` upload directory, but omitted `build/`, `web/tests/`, frontend test
reports/configuration, and `assets/evaluation/`. This is confirmed upload content,
not merely a glob hypothesis: the previous search-fix release's
`release/source-freeze.json` contains 62 paths under generated build code,
frontend tests/reports, and evaluation assets. `build/lib/` is a 536 KB duplicate
of the Python source. Its inclusion increases packaging size and preserves stale
source alongside the intended `src/` tree. No claim of public HTTP accessibility
or a real-data disclosure follows from inclusion alone.

After: `.vercelignore:17` and `vercel.json:7` exclude these paths during upload and
function bundling. `.gitignore:40` excludes root build/dist outputs without deleting
them. A local check using the installed `ignore` and `micromatch` packages confirms
nine representative forbidden paths are excluded at both boundaries, and nine
required source/build/resource paths remain included. The resource catalog and
packaged runtime skill are retained. `vercel.json` parses; `git diff --check` passes.
**The earlier source freeze is now stale and must be regenerated before release.**

### OPS-02 — P2: the documented Render Blueprint omitted a required credential

Before: `render.yaml:27` declared only the model key and public Supabase anon key
as input fields. It explicitly selects production mode and Supabase, while
`src/journalpulse/config.py:125` requires a signing key of at least 32 characters.
Following the documented Blueprint prompts therefore leaves `/ready` at HTTP 503
with `signing_key_missing` unless an operator adds an undeclared credential.

After: `render.yaml:32` declares `JOURNALPULSE_WRITE_SIGNING_KEY` with `sync:false`.
README Render instructions and `docs/OPERATIONS.md:50` explain that it must match
the database key. YAML validation confirms all three input fields and `/ready`
health checking are present. No key value or hosted binding was changed. A real
Render image/deployment was not built or launched in this audit.

### OPS-03 — P2: readiness reports configured AI with an empty chat model

References: `src/journalpulse/config.py:99`, `src/journalpulse/api.py:215`,
`src/journalpulse/intelligence.py:610`.

`openrouter_enabled` checks the legacy analysis model, not `chat_model`, and
configuration validation does not require the chat model. With an empty
`JOURNALPULSE_CHAT_MODEL`, production settings have no configuration issues and
`/ready` returns HTTP 200 with `llm:configured:not_probed`. Constructing the actual
chat client immediately raises `ValueError("OpenRouter is not configured")`.
Reproduced with fictional settings, FastAPI TestClient, and an injected database
probe; no external request was made. After: empty/whitespace chat models now report
`chat_model_missing` and `llm:not_configured`, with HTTP 503. Disabled AI still
requires no chat model. Focused regressions pass.

### OPS-04 — P2: the integration test executable has no artifact-integrity check

Reference: `scripts/integration_stack.py:191`.

The helper returns any executable named `postgrest` on PATH, accepts an existing
cached binary without a check, or downloads a versioned release archive and
immediately extracts/executes it. There is no archive checksum or binary version
validation. HTTPS and the explicit `v12.2.12` download URL are retained, but they
do not establish reproducible content or detect a corrupted/replaced cached file.
The installed/PATH binary can also silently be a different PostgREST version.

Proposed repair: bind the official published release asset to an independently
verified SHA-256; verify downloads before extraction and cached artifacts before
execution; require the expected `--version`; fail with a clear error on mismatch.
Do not treat a digest computed only from the existing installed binary as an
authoritative release digest. Root's official GitHub API request was blocked by
the proxy. The official expanded assets page at
`https://github.com/PostgREST/postgrest/releases/expanded_assets/v12.2.12` was
reachable and inspected. It lists the correct x86-64 archive, but publishes no
digest for that asset; the only displayed digest belongs to a different ARM asset.
The HTML is retained at `/tmp/jp-postgrest-official-assets.html`. No unrelated digest
was substituted, and no executable download or replacement occurred here. Artifact
integrity remains an explicit unverified tooling limitation.

### OPS-05 — P2: accepted model settings can exceed the host execution deadline

References: `vercel.json:5`, `src/journalpulse/config.py:113`,
`src/journalpulse/intelligence.py:747`.

Vercel's function limit is 120 seconds. Configuration accepts three attempts with
a 45-second chat timeout (135 seconds before backoff, authentication, and storage),
or three attempts with a 120-second timeout (360 seconds). The model timeout is
per attempt, so the host can terminate a valid configured request before the
application finishes its retry/error handling. A local settings probe confirms
both configurations have `configuration_issues=[]`. Defaults of 45 seconds and
two attempts give a 90-second model-attempt budget, below the configured 120-second
host limit; this verifies the default numerical budget, not worst-case end-to-end
latency. Require a deployment-aware total request budget or an absolute deadline
across retries. Root explicitly deferred this larger architectural change; it
remains a supported-configuration edge.

## Operational improvements, distinguished from confirmed product bugs

- **Readiness coverage:** `src/journalpulse/api.py:594` silently declines to mount
  a nonexistent configured web directory; `/ready` does not inspect it. An isolated
  probe with an invalid `JOURNALPULSE_WEB_DIST` returned `/ready:200` and `/`:404.
  After: an explicitly configured missing/file/empty export now reports
  `web:not_ready:export_missing` and HTTP 503. A configured index is checked and
  served; an unset static path preserves legitimate API-only deployments. Focused
  regressions cover both boundaries. Retain a homepage smoke test as well.
- **CORS validation:** `src/journalpulse/config.py:132` rejects only absent/local
  origins. Wildcard `*` and a URL with a path both pass production validation.
  A wildcard weakens the intended explicit browser-origin boundary; a path URL
  cannot match a normal browser Origin. After: production checks require explicit
  HTTPS origins and reject wildcard/userinfo/path/query/fragments, malformed ports,
  and loopback/unspecified hosts. Existing Vercel/Render/custom-port HTTPS origins
  and local HTTP development remain valid. Focused regressions pass. This is
  configuration hardening, not evidence of an authentication bypass in the current
  deployment.
- **Build reproducibility:** the runtime dependencies are locked, but
  `pyproject.toml:2` specifies unbounded `setuptools>=68` and `wheel` build-isolation
  requirements; neither appears in `uv.lock`. Frozen runtime resolution alone does
  not pin the packaging tools. Pin them or constrain isolated builds explicitly.
  Docker Node/Python base images and CI actions/service images use mutable tags.
  Record image digests/tool versions for a release rather than claiming identical
  builds from the application lockfiles alone.
- **Node consistency:** Docker and CI select Node 22, while the current workspace
  executable is Node 24.19.0. `web/package.json` has no engine/toolchain declaration.
  Current successful workspace validation is not a Node-22 build result. Use the
  documented CI/runtime version for release evidence.
- **Build network dependency:** `web/app/layout.tsx:2` uses `next/font/google`.
  A clean build needs Google font endpoints even though runtime fonts are bundled.
  Retain this network requirement in setup instructions or vendor licensed font
  files if offline/reproducible builds become a requirement.
- **Container shutdown:** `Dockerfile.api:48` launches Uvicorn through `sh -c`
  without `exec`. A local `/bin/sh` probe confirms the child has a separate PID;
  explicitly executing Uvicorn avoids relying on shell signal forwarding for
  graceful container termination. This audit did not test the Render shutdown path.
- **Docker context:** `.dockerignore` already excludes environment files, local
  databases, dependency directories, and common reports, but leaves generated
  `build/`, `dist/`, evaluation documents and `web/test-results-integration/` in
  the context. The final image copies only src/assets/static export, so do not
  confuse context size with exposing all context files in the runtime image.

## Positive controls and limits

- `uv lock --check --offline` passes with the existing writable cache: 80 resolved
  packages. The first attempt used the shell's default read-only HOME cache and
  failed before resolution; using `UV_CACHE_DIR=/workspace/.cache/uv` resolved
  that environment-only issue. The lockfile was not regenerated.
- CI has separate backend, frontend, and real PostgreSQL/PostgREST/browser jobs;
  coverage has a branch-aware 80% floor, generated API drift checks are present,
  and provider stand-ins avoid paid calls. This audit did not rerun those suites.
- Production authentication fails closed when Supabase is required; upstream auth
  outages do not turn into a development identity. Shared generation-rate checks
  also fail closed. `/ready` probes schema and matching signing keys, and honestly
  labels model configuration as unprobed; it performs no paid call.
- Scratch helpers validate local routing, including libpq query/env host overrides,
  before any reset. The gateway and PostgREST test servers bind loopback.
- Retention degradation remains observable without blocking privacy operations;
  the runbook explicitly says that HTTP 200 is not proof of successful cleanup.
- Docker serves only the exported frontend and API. Vercel ignores environment
  files and artifacts; no secret contents were read or printed during this audit.
- The guided-action migration impact document explicitly preserves old readiness
  and RPC contracts and documents that an application rollback does not reverse
  shared SQL or delete session/report data. Retained helper evidence shows release
  commands blocked without passing gates. Existing frozen release evidence must
  remain historical; source/package changes require fresh validation and freezing.
- No live Vercel/Supabase check was duplicated. Root owns those observations and
  the dependency vulnerability scan. No production, preview, or hosted migration
  result is claimed by this local audit.
- Authorized configuration/readiness fixes passed 63 targeted pytest cases,
  targeted Ruff, targeted mypy, and `git diff --check`. The selected tests include
  the existing static export integration check. This is not a new full-suite result.
