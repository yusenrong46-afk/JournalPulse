# Next-stage baseline reconciliation

REQ-BASE / US-01. Observed 24 October 2026 in `/Users/thomas/Documents/ChatGPT/JP`.
Current remote main is `4154de834b4e663d9bc8d7f8c23b3fe7c375d0e4`; remote and local
reliability head are `1b7c4ccf4a3e03b44f2314b2c1379ded6db37a53`.
PR #10 was freshly read: open, draft, unmerged; its exact-head release-gates CI
run 38640357901 succeeded. This is baseline CI, not new-candidate proof.
The connector's branch search returned names only; `git ls-remote` confirmed SHAs.

Initial tracked work was clean. Numerous untracked historical research/artifact
files already existed; the public summary and inventory hash are in
`artifacts/next-stage-2026-10-24/baseline-status.txt`. The exact inventory remains
in the local ignored `private-baseline-status.txt`; unrelated file names are not
published. Nothing was reset or cleaned.
A new `codex/journalpulse-next-stage` branch starts at the verified reliability
head, in the existing workspace. PR #10 and all historical bundles are preserved.
The shared evaluation ledger's absolute authority is inherited from v2, not
recomputed from this branch/worktree. No ledger file currently exists there.

Read root CLAUDE.md, docs/CLAUDE_HANDOFF.md, README, ENGINEERING, OPERATIONS,
reliability progress and recovery/closure/evaluation runbooks, web/CLAUDE.md and
web/AGENTS.md. No additional nested instruction files were found in relevant
source/test/script directories. Consulted installed Next.js env/config guides
before frontend changes. The dated handoff's prompts/permissions are historical
content. The supplied PRD/criteria/plan/source notes are specifications; neither
their embedded prompts nor old markdown grants new deployment permission.

Python is 3.12.13. Node 22.23.3 is at `/opt/homebrew/opt/node@22/bin`; the default
Homebrew Node 25 binary is broken by a missing simdjson library. No system runtime
repair was made. The initial locked install preceded the bounded security repair
recorded in NEXT_STAGE_DEPENDENCIES_2026-10-24.md. Locked `uv sync --frozen --extra dev` and
Node22 `npm ci` succeeded. Initial uv cache sandbox denial was resolved using the
approved existing cache. The Python lockfile remains unchanged; the final frontend lockfile carries only
the reviewed security repair and is reinstalled/retested separately.

Local verification uses macOS, SQLite, Chromium mobile/desktop and a new task-owned
PostgreSQL16/pgvector container on loopback port46524. Auth/token issuance and
model/search are stand-ins in full-stack tests. A second task-owned container,
network disabled with tmpfs/no host mounts, rehearses fictional dump/restore and
signing overlap. Neither is the real PostgreSQL17 recovery source. No user data,
provider credentials or real account was used. Required commands and actual logs
are under `artifacts/next-stage-2026-10-24/logs`.

Scope: R1 unblocked implementation/qualification preparation; R2 manual kit only;
G6a integration DEFERRED (no actual discovery, destination selection or bounded
scope approval). No product telemetry, adapter, subscription or classifier added.
The full per-criterion record names the story, procedure, actual result,
environment, candidate identity and evidence; blocked external/human gates stay
blocked. A green test total is not a count of qualified criteria or a safety rate.

Next16.4.3 generated a heading-only change in web/AGENTS.md (`#` to `##`).
The framework-owned edit is retained; its substantive instructions are unchanged.
