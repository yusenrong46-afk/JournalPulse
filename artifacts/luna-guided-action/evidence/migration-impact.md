# Guided-action migration impact and release preparation

Prepared October 5, 2026 by read-only inspection. No deployment, alias, schema,
account, environment-binding, paid-model, or search mutation was performed by
this preparation task.

## Observed shared environment

The existing protected Preview alias resolves to
`dpl_Ey6QzvQBWwpXaqwyT64N8rhR9d5n`. Production `journalpulse.vercel.app`
resolves to `dpl_57Rut4UoR24LsTRjsA7nEpiBQycU`; neither was changed. The
Both existing addresses returned HTTP 200 with `status:ready` before migration,
including reachable schema and valid signing probes. The Vercel project is `prj_EOSYVlo1KjkdnOAT8Smm72LdgFrV` and the shared Supabase
project is `cggixttnyprrbaotmjxj`. All required Preview credential binding
**names** exist. Values and automation-bypass credentials were neither printed
nor saved. Existence is not verification of provider access or credits.

`schema-before.json` captures DDL/index/policy metadata and aggregate counts
only. It contains no journal passages, message bodies, emails, credentials, or
individual-user records. `FUNCTIONS_BEFORE_REFERENCE.sql` captures the one
replaced public function plus relevant unchanged legacy conversation/readiness
functions. This is an operator reference, not a backup or automatic rollback.

No activity table, activity RPC, new activity owner index, lifecycle trigger,
compatibility trigger, or readiness v3 was present at inspection. The additive
migration is **not already applied**. Recheck immediately before execution;
any existing target object or recorded version blocks replay. Existing integrity,
preference, journal, retention, and readiness-v2 objects are present.

The nine inherited migration files match the original dirty-workspace hashes in
`baseline-manifest.json`. The hosted project has **no**
`supabase_migrations.schema_migrations` table: earlier SQL was applied manually.
Therefore an empty migration-history result is not evidence that the database is
empty. Preserve the actual installed schema. Do not replay the nine files, create
fictional historical entries, or silently establish a new migration-history
schema. The release helper records this new manually applied migration in the
inspectable `migration-applied.json` artifact, alongside its hash and actual
post-migration object/security checks. If a history table is introduced by an
operator before execution, the helper records only the newly applied version.

## Proposed migration

File: `supabase/migrations/202610050001_guided_activity_sessions.sql`.
SHA-256 at preparation: `737a2de145a257a48bdf94b5ce946bbb22ca3fe1604a4c0ed51aaf1f150469f6`.
This hash must match the final source freeze and gate evidence at execution.
No earlier migration is modified or replayed.

| Change | Impact and boundary |
| --- | --- |
| New `activity_sessions` table | Owned session state, server deadline, bounded revision, participation report, honest OPE-ineligible selection, and follow-up message identity. Owner/chat and owner/journal foreign keys cascade on deletion. |
| New `activity_receipts` table | UUID request identity and request hash preserve retry semantics. Owner-scoped read policy; receipt count is bounded per session. Session deletion cascades receipts. |
| New indexes | Owner/chat uniqueness supports the foreign key; a partial unique index permits only one nonterminal session per owner/chat. Owner/session receipt indexes bound lookup cost. Creating the unique conversation index briefly locks existing small shared metadata; the helper sets a 5-second lock timeout and a 60-second statement timeout. |
| RLS and grants | Both new tables enable owner SELECT RLS. Public/anonymous/authenticated direct writes are revoked. Authenticated reads are permitted. Existing table policies/grants are unchanged. |
| Five versioned write RPCs | Offer, command, report, claim-follow-up, finish-follow-up. Existing private signed-payload verification plus authenticated owner checks guard writes. Source → chat → session locks, revisions, receipt hashes, and session incarnation reject races and stale results. |
| Private receipt/lifecycle helpers | Explicit search paths, no ordinary client execution grants. Controls are bounded at 48 receipts with 64 total; follow-up leases last 90 seconds and permit at most three attempts. |
| Compatibility trigger | New LLM cards occupy additive `activity_card`; legacy `card` remains null for nullable propensity. Rejects placing such a card in the legacy field. Clears new offers on closed/support/listen/ordinary-pause chats. |
| Lifecycle trigger | Existing close, sweep, accept, support, listen, and chat-pause updates also stop active new sessions, withdraw deadlines/check-ins, and cancel pending generations. Unretained closure clears participant notes and receipts. It does not rewrite existing rows during installation. |
| Replaced `delete_my_journalpulse_data()` | Adds the two new children first in the existing authenticated owner-only deletion list. Existing tables and owned deletion behavior are retained. This function runs only on an explicit account-data deletion request. |
| New readiness v3 | Requires activity RPC/lifecycle/compatibility objects. Readiness v2 and all existing conversation/signing RPC signatures remain available to old production. |

Installing the migration creates metadata/empty new tables and functions; it does
not issue a data deletion, rewrite current conversations, read secret values,
change production routing, or invoke a model. New triggers affect subsequent
ordinary updates, which is why compatibility and lifecycle checks are release
gates. Session controls do not extend the parent chat idle-retention timestamp.

## Old/new compatibility evidence

`backend-old-parser-compatibility.json` passed the **exact frozen pre-upgrade
parser**, whose domain SHA-256 is `7908ee078cfa4db02eea6de193097fba808fb130a4538719daff7a81fffe16f4`. It verified
that a new conversation has legacy `card:null`, nullable/OPE-ineligible
`activity_card`, and can be parsed by old models that ignore the additive field.
The same parser rejected nullable propensity inside the legacy card field,
showing the original failure and the purpose of the explicit additive field.
The legacy policy table still requires nonnull propensity; no ALTER weakens it.

`backend-session-evidence.md` records actual scratch PostgreSQL main/journal/
integrity/activity runs, including 42 final signed activity assertions. These
prove local behavior, not the shared hosted runtime. After migration and candidate
readiness, root must create only a disposable owned scenario and test the existing
production `/ready` and authenticated export against a new-card row. The final
`old-production-compatibility.json` must be tied to the candidate deployment/hash
and the unchanged production deployment ID. No real user data is benchmark input.

## Rollback limits

The compatible rollback target is the prior Preview deployment above. An alias
rollback changes the served application only. **It does not undo the shared SQL,
remove new session/report data, or rewind the unchanged production deployment.**
Keeping this additive schema installed is the intended safe code rollback: old
RPCs/readiness v2, nonnull legacy propensity, and parser-compatible JSON remain.
Do not execute the captured functions blindly after new data exists. Dropping new
tables would delete activity reports; dropping guards or reverting deletion
integration can weaken privacy. Any such destructive or security-boundary change
needs a separately reviewed concrete recovery plan and authorization. No hosted
reset, drop, backfill, automatic data restoration, or migration replay is included.
A forward fix is preferred if a runtime issue appears after schema installation.

## Release gates and execution order

The prepared helper is `/workspace/.onboarding/guided_action_release.py`.
It uses the existing injected-token Vercel/Supabase clients. Its `inspect`,
`freeze`, and `status` commands mutate only local evidence (status reads Vercel).
No mutation command was run during preparation.

1. Finish implementation and required software checks. Freeze the final model
   request/prompt/candidate before held-out comparison. Complete bounded actual
   baseline/candidate controlled/adaptive observations, blind judging, and the
   offline HTML/patch evidence. Report failures without deleting unfavorable cases.
2. Run helper `freeze` once the final upload is stable. It captures every actual
   upload file, including inherited dirty code, and checks all nine old migration
   hashes. Source changes invalidate this freeze.
3. Populate `release-gates.json` with the exact frozen `candidate_sha256`,
   `migration_sha256`, `overall_status:"pass"`, and the named gates below. Each
   gate needs `status:"pass"` plus `evidence:[{"path":"/absolute/artifact",
   "sha256":"actual-file-hash"}]`. The helper verifies all evidence hashes.
4. Root reviews the staged SQL/impact and executes `migrate` only after all gates
   pass. The helper rechecks no new objects/version, verifies unchanged routing,
   uses a bounded transaction/advisory lock, reloads PostgREST, and saves observed
   RLS/no-direct-write/legacy-readiness state. Never re-run it when objects exist.
5. `deploy` uploads exactly the frozen source to this project as a Preview
   deployment; it explicitly refuses a production target. Poll `status` until
   actual Vercel READY; do not assume deployment creation implies readiness.
6. Run read-only candidate health, a disposable owned browser/API activity and
   inline-search smoke, and unchanged-old-production readiness/export proof.
   Preserve only fictional data and sanitized evidence. Root handles bounded paid
   calls and owned-account cleanup; this helper performs no model/search calls.
7. Run `alias` only after all gates still pass and all three candidate artifacts
   below match the READY candidate ID/hash. The helper rechecks unchanged
   production and old Preview alias, updates only existing Preview, and verifies
   both aliases. Finally smoke the stable address and clean up owned QA data and
   temporary gateways. Aliasing alone is not final completion.

Required gate names:
`critical_safety_privacy_lifecycle`, `heldout_primary_gain`,
`heldout_candidate_win_rate`, `naturalness_non_regression`,
`adaptive_before_after_complete`, `new_functionality_acceptance`,
`required_software_checks`, `bounded_live_model_search_evidence`,
`old_new_parser_compatibility`, `inspectable_report_and_patches`,
`spend_within_budget`.

Required candidate artifacts all include `status:"pass"`, `deployment_id`, and
`candidate_sha256`: `candidate-health.json`, `candidate-smoke.json`, and
`old-production-compatibility.json`. Health also needs `homepage:200`,
`readiness.status:"ready"`, `unauthenticated_private_data:401`, and
`development_identity_rejected:401`. Compatibility also needs the observed
`production_deployment_id`, `owned_disposable_export_parsed:true`,
`legacy_readiness_ready:true`, and `new_activity_card_seen_in_export:true`.
Candidate smoke must contain the actual checks/transcript references; no teacher
score substitutes for timer, ownership, export, or deletion evidence.

The provisional thresholds are ≥0.3 held-out primary improvement, ≥60% candidate
wins among non-tied eligible held-out pairs, naturalness decline ≤0.2, all eight
before/after adaptive conversations observed, and all critical/software/new-flow
checks passing. They are engineering criteria, not clinical efficacy claims.

Helper validation: Python compilation and Ruff passed;
`release-guard-verification.json` records nine local failure/pass guard checks
with zero network calls. These are prepared-tool checks, not a completed release.
