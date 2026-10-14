# JournalPulse reliability progress October 14 2026

Base: GitHub main `4154de834b4e663d9bc8d7f8c23b3fe7c375d0e4`.
Branch: `codex/journalpulse-reliability`. The built-in goal is active. Application
changes below are **not deployed**; the prior release remains live.

| Stage | State | Evidence and remaining dependency |
|---|---|---|
| A. Encrypted live recovery | Tool and fictional smoke verified; real export pending | Source PostgreSQL 17.6 / Free plan inspected. Secure database URI and encryption passphrase input still needed. Zero verified real backups recorded. |
| B. English safety | Local repair complete | 52 clear regression cases pass; 8 ambiguous cases remain review-only. 24 entry-point checks preserve no-provider support routing and consent. |
| C. Fresh AI evaluation | Frozen, not run | 100 fictional cases in 50 families; 40/60 split; exact source/packets hashed. Luna candidate and Sonnet 4.6 judge; shared USD10 cap. Secure OpenRouter key input pending. No paid attempts recorded. |
| D. Privacy and closure | Implemented and locally checked; hosted gate pending | Contact added. Fresh-auth endpoint, pending-write guards, private closure ledger, Admin deletion/verification, scoped retry receipt and UI are tested locally. Server-only Admin credential, migration and disposable hosted-account proof remain required. |
| E. Citco resume | Complete | Revised sibling PDF plus JSON/Python editable source. One page rendered and inspected; original hash unchanged, outside-project pixels identical, five hyperlinks preserved. |
| Release | Pending | No new live migration, deployment, merge or promotion. Real recovery must precede migration; hosted closure must pass before enablement. |

## Local evidence

Evidence folder: `artifacts/reliability-2026-10-14`.

- Full backend: 1,409 passed, 91.69% branch-measured coverage. Ruff, mypy and
  generated OpenAPI check pass. A dependency deprecation warning remains.
- Frontend: 230 unit tests pass; lint and TypeScript pass.
- Browser: 62 mocked checks pass across phone and desktop. Opening the closure
  panel exposed a 4.28:1 button contrast failure; the repaired danger color passes
  the expanded accessibility checks. The failed run is preserved separately.
- All PostgreSQL migration/lifecycle/erasure/closure verifier scripts pass against
  an isolated local PostgreSQL 17 container with fictional records. Concurrent
  write-versus-closure orderings both pass. Hosted Auth Admin calls are unverified.
- Browser → FastAPI → PostgREST → PostgreSQL: 22 passed, 4 optional tours skipped.
  This uses real migrations/database/API with stand-in Auth and a fake model.
  The production frontend build also passed during this run.
- Backup encryption/decryption and isolated restore mechanics passed a fictional
  smoke drill. The old smoke log's scope label includes “live”; that is a tool
  wording defect, now removed. Its input was fictional and it is **not live proof**.
- Resume evidence: `resume-verification.json`. Revised PDF and source are beside
  the selected October 5 original, outside Git. No new model-quality claim was added.

## Runbooks and evidence boundaries

- [Local recovery](LOCAL_RECOVERY_RUNBOOK.md): hidden prompts, encryption, local
  isolation, three verified generations, weekly manual cadence and deletion replay.
- [English safety](ENGLISH_SAFETY_2026-10-14.md): demonstrated defects and unresolved
  ambiguity. Passing these development cases is not general safety accuracy.
- [AI protocol](evaluations/LUNA_QUALITY_2026-10-14.md): frozen version, costs,
  denominators and uncertainty. Human review is NOT RUN; no clinical claim.
- [Account closure](ACCOUNT_CLOSURE_RUNBOOK.md): privilege boundary, retries,
  deployment/rollback and the remaining hosted gate.

If deletion history since an old backup cannot be reconciled, the restored records
must stay inaccessible. No independently replicated deletion history or automatic
daily backup has been claimed. No subscriptions, classifier or multilingual
expansion were added. Real credentials and journal records must remain outside Git,
evaluation packets, screenshots and reports.
