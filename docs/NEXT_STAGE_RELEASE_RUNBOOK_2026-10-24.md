# Next-stage release and support runbook

This candidate is not approved for hosted promotion. R1 remains incomplete while
real recovery, hosted closure, paid qualification and independent review are blocked.
Old handoff permissions do not authorize a new shared migration or deployment.

## Actual actions and reversible local work

Work is on `codex/journalpulse-next-stage`, based on the untouched reliability
candidate `1b7c4cc`; separate commits cover routing, browser policy, qualification
and preparation. No shared migration, manual deployment, account deletion or paid
provider request has occurred. The source/evidence records identify the final
commits and checks. Any GitHub branch/PR publication is review delivery, not a
production release. Inspect automatic checks separately if a preview is created.

Task-owned disposable PostgreSQL containers contain fictional data only. They
are removed at handoff. The fictional recovery receipt proves local mechanics
only; it explicitly does not prove hosted backup/PITR or later-deletion replay.

There is no newly deployed version to roll back. Keep the original reliability
branch available; use an ordinary reviewed revert if this local slice is rejected.
Preserve unrelated files and all old/new evaluation evidence. Do not reset/clean
the workspace, delete results, reset the ledger or reopen deleted data.

## Ordered external gates

1. Supply the existing intended Supabase session-pooler URI and a private backup
   passphrase through the recovery tool's hidden interactive prompts. Docker/GPG
   are available; the secure inputs and actual verified generation are missing.
   Use [the recovery runbook](LOCAL_RECOVERY_RUNBOOK.md), PostgreSQL17 and the
   128MiB guard. Never restore to production. Zero real verified generations are
   currently recorded; three is a rolling retention target, not three fake runs.
2. Verify the real encrypted export/local restore receipt and deletion-reconciliation
   limits. If source loss prevents complete later deletion history, recovered data
   remains inaccessible. No independently replicated deletion ledger is claimed.
3. With current explicit shared-migration/staging authorization and secure Admin
   configuration, review/apply only the compatible additive account-closure
   migration `202610140002_account_closure.sql`; capture its hash/state and the
   known-good hosted deployment identity first. Never replay scratch scripts on
   hosted Supabase. New API readiness expects v6/account-closure-1; v5 is retained.
4. Stage the exact tested source with closure enabled only for the scoped disposable
   smoke. Run `scripts/verify_hosted_closure.py` with a verified deployment record
   and a new output path. Real >5-minute refresh/fresh-sign-in and Auth deletion
   proof is required. Never substitute a local fake or existing user account.
5. Reconcile the authoritative missing USD10 ledger before paid evaluation. Supply
   a secure provider key through the explicit v4 wrapper, preserve endpoint/privacy,
   cost/no-repeat controls, then obtain the preselected independent blind review.
   Use the separate decision tool; a confirmed critical failure FAILS the candidate.
6. Reconcile current-candidate CI and all gates, capture actual hosted commit,
   deployment ID, schema/readiness, configuration and rollback target, and obtain
   current promotion authorization. A historical deployment ID is not a verified
   rollback target. Keep full R1 BLOCKED until its required evidence exists.
7. Execute the manual pilot kit only with real consent, approved capabilities and
   actual elapsed windows. G6a integration remains deferred. No real participant,
   human review, personal walkthrough or impact claim is supplied by these templates.

## Support triage without private writing

Collect request ID, build/deployment identity, operation class and bounded error
stage. Do not collect tokens, receipt secrets, journal text or action titles.
For an uncertain save, preserve exact local writing and its original receipt;
retry only that logical operation. Never tell a person Saved without confirmation.
For Auth failure, keep private content hidden; do not switch to development identity.
For provider failure, preserve saved writing/report and explain the unavailable
chosen mode. A Simple fallback is not an AI success. Revocation cannot recall a
request already transmitted.

Saved-data erasure keeps sign-in identity. Account closure is separate: return
success only after verified identity/dependent-row absence. Pending closure blocks
new owner writes; other owners continue. A receipt expires after15minutes; after
identity deletion with uncertain completion, reconcile privately rather than
promise indefinite ordinary retry. Preserve pending/completed markers.

## Future hosted rollback, not performed here

Disable new closure attempts, restore only a verified compatible prior app and
check readiness/ordinary saved-data operations. Retain the additive migration,
erasure/closure guards, deletion history and pending/complete markers. Never
resurrect identities or drop guards to hide an error. A live recovery or changed
external delivery state requires operator reconciliation before access/dispatch.
