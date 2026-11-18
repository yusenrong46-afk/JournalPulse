# Upgrade spec (verbatim copy)

Copied verbatim from Thomas's prompt "Upgrade five portfolio projects to the
Portfolio Standard (L1 to L5)", dated 2026-11-18: section 2, section 3 and the P3
project spec. This file is the reference for the gates; `upgrade/STATUS.md` holds
the results.

## 2. Rules

1. **Branch and PR.** In each repo work on `upgrade/2026-11` and open one draft PR titled `Upgrade to Portfolio Standard L1 to L5`. Commit at least once per level with a conventional commit message.
2. **Repo instructions stay in force.** Read each repo's `CLAUDE.md`, `AGENTS.md` and any protected-files list first. Where they are stricter than this prompt, they win. Where this prompt is stricter, it wins.
3. **Do not ask, decide.** If something is ambiguous, take the most conservative reading, write it under Decisions in `upgrade/STATUS.md`, and continue.
4. **Investigate before you change.** When a check fails, read the whole output, reproduce it in isolation, find the cause, fix the cause. Never edit a test to match the code unless the test contradicts its own spec. After three distinct failed attempts, mark the gate BLOCKED with the evidence and continue with work that does not depend on it.
5. **Never weaken.** Do not remove, relax, skip or bypass an existing test, threshold, lint rule, type setting, coverage gate, mutation gate, frozen evaluation file or protected file to make something pass. You may add.
6. **Report only what you ran.** Every PASS in a status file carries the command and the output line that shows it. A result you did not produce in this session is not a result.
7. **Invent nothing.** No invented metric, user, customer, quote, licence, permission or cost. Unknown stays "unknown" or "not recorded".
8. **Fictional data only.** Demo seeds, fixtures and screenshots use invented content. Never put real personal writing, credentials or keys in a file, log or screenshot. Never print a credential.
9. **Label authorship.** Keep each repo's existing statement about AI assistance. Anything you draft that needs human judgment (gold labels, benchmark questions, case-study facts) is marked `agent_draft_unreviewed` until Thomas reviews it.
10. **Stay in scope.** Build what the spec for that level says. If you find a real bug outside the spec, fix it only if it is small and has a test; otherwise list it under Findings.
11. **Messages.** Send Thomas at most one progress message a day, plus the final report. Message immediately only if you find a leaked secret or discover that a default branch or a production system changed.

## 3. The Portfolio Standard

A project is at level N when `make gate-lN` exits 0 in a fresh clone. Add `make gate-l1` through `make gate-l5` to every repo (create a `Makefile` where none exists; where Makefile targets are protected, add new targets only). Each gate target runs that level's automated checks and every lower level's. Level-specific details are in section 6.

**L1 Reachable. A stranger can open it and run it.**
- `make setup` then `make test` pass from a fresh clone on Linux. Every skipped test states its reason.
- CI runs setup, tests and the gate scripts on every pull request and is green on your branch.
- The README's first screen has: one sentence saying what it does and for whom, one dated result with its number, one image, and how to see it (demo link or one command). `scripts/check_readme.py` asserts these and that every relative link resolves.
- All finished work is on one branch. Fold in finished open branches, or list each with the reason it stays out.
- A secret scan is clean. No new file over 5 MB.

**L2 Demonstrable. Anyone can try it in two minutes with no account, no key and no cost.**
- `make demo` starts the product on localhost with fictional or synthetic data, no credentials and no paid call.
- `make demo-check` drives a browser through the main journey, asserts what a visitor should see, and saves three screenshots and one short video under `docs/demo/`.
- During `demo-check`, no request leaves the machine except to hosts on a short allowlist written in the test (map tiles, fonts). The test asserts this.
- `docs/DEPLOY_DEMO.md` gives Thomas the exact steps and environment variables to host the demo in under ten minutes.

**L3 Measured. Every number in the README regenerates from a command.**
- `make eval` runs offline, makes no paid call, and writes a results file. Two runs give byte-identical files. Timestamps and timings go in a sidecar file that this check excludes.
- The README results block is generated from that file. `make check-docs` fails when they differ.
- CI fails when a guarded number falls below its committed baseline.
- Every real-model evaluation has a documented command, a stated budget, and a status of RUN (with receipts) or NOT RUN.

**L4 Connectable. It plugs into another system and can be set up for a new customer from config.**
- One documented, versioned interface (HTTP with OpenAPI, MCP, or CLI with JSON Schema) with contract tests. It calls the existing code; it is never a second implementation.
- `make onboard NAME=<new>` sets up a new city, database, region or environment with no source edits and writes a report with per-step timings.
- A container image builds and passes a smoke test, or the status file records that Docker was unavailable.
- `docs/ONBOARDING.md`: prerequisites, steps, how to verify, how to roll back.

**L5 Explainable. Thomas can defend it live.**
- `docs/WALKTHROUGH.md`: ten questions an interviewer would ask, each answered in under 120 words with a pointer to file and symbol. `scripts/check_walkthrough.py` asserts every pointer exists.
- `docs/CASE_STUDY.md` in customer-loop form: problem, discovery, scope, build, demo-back, result, what I would change. Every factual claim cites a file or artifact in the repo.
- `exercises/`: three stand-alone hand-coding exercises, each a stub, failing tests and hints. No solution anywhere in the repo. The gate asserts the tests fail on the stubs.
- `docs/DEMO_SCRIPT.md`: a 90-second spoken demo and a five-minute deep-dive outline.
- `upgrade/RESUME_BULLETS.md`: three bullets backed only by numbers the repo now generates.

## 6. P3. JournalPulse (AI journaling app)

**Repo:** `yusenrong46-afk/JournalPulse` (public). Read `CLAUDE.md`, `docs/CLAUDE_HANDOFF.md` and `web/AGENTS.md` first.

**Context.** Next.js front end, FastAPI back end, Supabase Auth and Postgres with row-level security, deployed on Vercel at `journalpulse.vercel.app`. AI chat is optional; a scripted Simple mode makes no model call. A safety and consent gate runs before any provider request.

**Protected.** `artifacts/**` and `assets/evaluation/**` (evaluation evidence). Existing files in `supabase/migrations/` (add new migrations only). Safety routing: on the 34-case development set the result must not get worse than 29 pass, 3 missed support, 2 review required. Privacy boundaries: no journal text in logs, tests or seeds beyond fictional text. The hosted database and deployments.

**Known state.** `claude/epic-hawking-u8ijg4` holds four fix commits not on `main`; its backend suite gives 1,385 passed. The README quotes hand-typed counts from October 7 (1,270 backend tests). A first-time visitor to the live app meets a sign-in screen; there is no demo path.

**L1**
1. Merge `claude/epic-hawking-u8ijg4` into the upgrade branch. Do not squash.
2. Run each CI job's commands locally where you can: backend lint, mypy, `scripts/export_openapi.py --check`, pytest, `scripts/validate_resources.py`; web lint, typecheck, unit tests and build.
3. Date the hand-typed README counts now; L3 replaces them with a generated block.

**L2. A demo sandbox.**
1. Behind `JOURNALPULSE_DEMO_ENABLED` (default false), the sign-in screen offers `Try the demo`. It creates an isolated identity with no email. Row-level security still applies.
2. A demo identity can never cause a model or search request, enforced in the API and tested: a client that asks for AI as a demo identity gets Simple mode.
3. Seed three fictional entries and one completed Simple Luna activity loop. Erase demo data after 24 hours through the existing erasure path. Rate-limit creation. The Canadian crisis-support line stays visible.
4. Every demo view carries the label `Demo: fictional data, AI off`.
5. `make demo` locally uses the existing local-identity and SQLite path with AI and search disabled. `make demo-check`: Try the demo, write and save an entry, Simple Luna, choose an activity, check in, see it in Journey.
6. `docs/DEPLOY_DEMO.md`: the Supabase and environment settings and the deploy steps.

*Done when:* gate passes, and the integration job proves isolation between two demo identities on real Postgres.

**L3**
1. `make eval`, offline: (a) `scripts/evaluate_safety.py` on the development set, keeping its `NOT_QUALIFIED` label; (b) recompute the published October 5 benchmark aggregates from the stored judgments in `artifacts/luna-release-benchmark/` (documented as 8 wins, 5 losses, 7 ties; mean primary gain +0.35; interval -0.13 to 0.83) without calling a model; (c) test counts and coverage from this run.
2. If a recomputed aggregate disagrees with the docs, report the discrepancy. Do not edit the artifacts.
3. Generated README block with each number's date and prompt version (the benchmark tested `guided-action-2026-10-05.3`; the runtime prompt is `guided-action-2026-10-06.2`).

**L4. An operated service.**
1. Build identity: `/health` or `/v1/system/status` returns the deployed git commit. Test it.
2. `.github/workflows/synthetic.yml`, hourly: request `/health` and `/ready` on the production URL; once the demo is live, run the demo journey; append the result to `status/history.jsonl` on a `status` branch; open an issue after two consecutive failures. The README shows a 30-day availability line generated from that history. No secret is needed. It starts running when Thomas merges.
3. `docs/SLO.md` (target, what counts as down, error budget) and `docs/INCIDENT_DRILL.md`: run one local drill (database unreachable, `/ready` returns 503, follow `docs/OPERATIONS.md`, recover) and record the commands and output.
4. `make onboard NAME=<env>`: bring up a complete fresh environment from zero with Docker Compose (Postgres, PostgREST, API, exported web) using the real migrations, run the six schema and integrity verification scripts CI runs, pass `/ready`, run the demo journey against it, and write `outputs/onboarding/<env>/REPORT.md` with timings.
5. `docs/API.md`: three curl examples against the local stack. The OpenAPI check that already exists is the contract test.

**L5.** Walkthrough questions: how a late or duplicate AI reply is kept from overwriting a newer save; where the consent and safety gate sits and what it cannot catch; how ownership is enforced; what the 60-scenario benchmark showed and why the release was held; what erasure removes. Exercises (stand-alone): an idempotent command receipt that ignores a stale reply; a pre-model gate ordering test using a fake provider that fails the test if called; an owner-scoped data access check.

**Needs Thomas.** Review and merge. Apply any new migration to the hosted database, enable the demo in Supabase and Vercel, and approve the production deploy. Confirm from the build identity that production serves `main`.
