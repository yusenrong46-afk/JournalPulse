# JournalPulse acceptance criteria
Version 1.0 - proposed definitions, not executed results
Prepared 24 October 2026. Read with PRD.md.

## How acceptance works

There are 65 criteria: 53 core-stage criteria and 12 conditional integration criteria. These are user/operational risk scenarios, not 65 requested new features or 65 new test files. Map existing tests where they prove the requirement; add only meaningful missing regression/failure checks.

Every criterion uses Given/When/Then plus an evidence requirement. If a criterion groups several required subcases, all applicable subcases must pass. A screenshot or green process exit is not a substitute for the specified behavior.

Allowed execution states: NOT_RUN, PASS, FAIL, BLOCKED, NOT_APPLICABLE. The definitions start NOT_RUN because they apply to the next implementation candidate. That does not assert the current code fails every criterion. NOT_APPLICABLE requires an explicit scope decision and a disabled conditional capability. Full R1 requires all its in-scope gates; missing credentials do not make a requirement irrelevant.

Evidence classes stay separate: deterministic/unit, API with fake providers, real disposable PostgreSQL, browser with mocked API, browser through actual API/database, hosted disposable account, actual paid-model evaluation, and real human observation.

For every execution record capture criterion ID, source commit and relevant inference fingerprint, schema/build/environment, timestamp, exact command/procedure, actual result, artifact paths and blocker. Store no secrets or real journal contents. Use release_evidence_template.json as the starting record.

P0 failures block the affected release/capability and any wider pilot. P1 criteria remain required within their stated gate; priority does not silently waive them. See PRD Section 4 for G0-G7.

## Traceability index

| Requirement | Criteria | Gate(s) |
| --- | --- | --- |
| REQ-BASE | AC-BASE-01, AC-BASE-02, AC-BASE-03 | G0 |
| REQ-SAFE | AC-SAFE-01, AC-SAFE-02, AC-SAFE-03, AC-SAFE-04 | G1, G7 |
| REQ-WRITE | AC-WRITE-01, AC-WRITE-02, AC-WRITE-03, AC-WRITE-04 | G1 |
| REQ-CONSENT | AC-CONSENT-01, AC-CONSENT-02, AC-CONSENT-03, AC-CONSENT-04 | G1 |
| REQ-ACT | AC-ACT-01, AC-ACT-02, AC-ACT-03 | G1 |
| REQ-OWN | AC-OWN-01, AC-OWN-02, AC-OWN-03, AC-OWN-04 | G1 |
| REQ-CLOSE | AC-CLOSE-01, AC-CLOSE-02, AC-CLOSE-03, AC-CLOSE-04 | G1, G2 |
| REQ-REC | AC-REC-01, AC-REC-02, AC-REC-03, AC-REC-04 | G2 |
| REQ-WEB | AC-WEB-01, AC-WEB-02, AC-WEB-03 | G1 |
| REQ-EVAL | AC-EVAL-01, AC-EVAL-02, AC-EVAL-03, AC-EVAL-04, AC-EVAL-05, AC-EVAL-06 | G3 |
| REQ-REL | AC-REL-01, AC-REL-02, AC-REL-03 | G1, G2 |
| REQ-UX | AC-UX-01, AC-UX-02, AC-UX-03 | G4 |
| REQ-OBS | AC-OBS-01, AC-OBS-02, AC-OBS-03 | G4, G5 |
| REQ-PILOT | AC-PILOT-01, AC-PILOT-02, AC-PILOT-03 | G5 |
| REQ-INT | AC-INT-01, AC-INT-02, AC-INT-03, AC-INT-04, AC-INT-05, AC-INT-06, AC-INT-07, AC-INT-08, AC-INT-09, AC-INT-10, AC-INT-11, AC-INT-12 | G6a, G6b |
| REQ-HANDOFF | AC-HANDOFF-01, AC-HANDOFF-02 | G7 |

## REQ-BASE

### AC-BASE-01 - Current branch and instructions

**Gate:** G0 | **Priority:** P0 | **Applies:** Core next-stage scope

**Given:** The repository and existing PR10 work are available.

**When:** Inspect current refs, dirty work and relevant instructions before edits.

**Then:** Record actual main/candidate/base relation and preserve unrelated changes; use the current verified implementation, not an assumed historical branch.

**Verification:** Read-only Git and instruction review.

**Required evidence:** Baseline record with SHAs, worktree decision and relevant instruction paths.

**Initial execution state:** NOT_RUN. Record actual result separately.

### AC-BASE-02 - Reproducible locked setup

**Gate:** G0 | **Priority:** P1 | **Applies:** Core next-stage scope

**Given:** A clean supported Python/Node environment exists.

**When:** Install with the existing lockfiles and run a bounded offline smoke.

**Then:** Setup works or a precise environment blocker is recorded; no automatic unrelated dependency upgrade and no paid provider call.

**Verification:** Fresh local install/smoke.

**Required evidence:** Actual commands, environment versions, exit results and any bounded setup fix.

**Initial execution state:** NOT_RUN. Record actual result separately.

### AC-BASE-03 - Traceability and conditional scope

**Gate:** G0 | **Priority:** P1 | **Applies:** Core next-stage scope

**Given:** This PRD, catalog and plan are loaded.

**When:** Map requirements to existing modules, tests and stories.

**Then:** Every applicable requirement has an owner/story/evidence target; optional integration is deferred unless G6 is approved; existing functionality is not rebuilt merely to fill a checklist.

**Verification:** Document review.

**Required evidence:** Current traceability map and scope decision.

**Initial execution state:** NOT_RUN. Record actual result separately.

## REQ-SAFE

### AC-SAFE-01 - Delayed affirmative intent regression

**Gate:** G1 | **Priority:** P0 | **Applies:** Core next-stage scope

**Given:** The exact synthetic failing fixtures from the audit exist.

**When:** Run the classifier before and after the narrow temporal-negation correction.

**Then:** The previously failing present/future-intent fixtures route to support; the pre-fix reproduction and reason for change are preserved.

**Verification:** Focused deterministic regression.

**Required evidence:** Fixture IDs, pre-fix failure, post-fix result and explained diff.

**Initial execution state:** NOT_RUN. Record actual result separately.

### AC-SAFE-02 - Paired denial and context controls

**Gate:** G1 | **Priority:** P0 | **Applies:** Core next-stage scope

**Given:** Paired genuine denials and reviewed historical/quotation controls are predeclared.

**When:** Run the same classifier changes over the control set.

**Then:** Expected routes remain correct; no broad removal of negation or blanket support classification; ambiguous fixtures have explicit expected behavior and review rationale.

**Verification:** Focused unit/control tests.

**Required evidence:** Case-by-case expected/actual routes, including any unresolved ambiguity.

**Initial execution state:** NOT_RUN. Record actual result separately.

### AC-SAFE-03 - Real entry points suppress side effects

**Gate:** G1 | **Priority:** P0 | **Applies:** Core next-stage scope

**Given:** Instrumented candidate/search clients and supported API entry points are available.

**When:** Submit required support fixtures through the actual reflection/chat paths.

**Then:** Support response uses deterministic behavior; no candidate/judge/search call occurs for those paths and no inappropriate activity continuation is persisted.

**Verification:** API integration with captured fake clients and real local persistence where meaningful.

**Required evidence:** Call counts, response state and persistence assertions by entry point.

**Initial execution state:** NOT_RUN. Record actual result separately.

### AC-SAFE-04 - No unsupported safety claim

**Gate:** G7 | **Priority:** P1 | **Applies:** Core next-stage scope

**Given:** The bounded English regression set passes.

**When:** Review UI/help/README and evaluation reporting.

**Then:** Scope stays English and research beta; passing fixtures are not described as general safety accuracy, clinical validation or crisis-care capability.

**Verification:** Content and claim review.

**Required evidence:** Current disclosure/case-study excerpts with exact evidence scope.

**Initial execution state:** NOT_RUN. Record actual result separately.

## REQ-WRITE

### AC-WRITE-01 - Exact save without model availability

**Gate:** G1 | **Priority:** P0 | **Applies:** Core next-stage scope

**Given:** An owner has a fictional draft and model/search clients are disabled or failing.

**When:** Save then retrieve from authoritative persistence.

**Then:** Exact submitted text is returned; Saved appears only after a valid receipt; provider call count is zero.

**Verification:** API plus browser integration.

**Required evidence:** Persisted fixture equality, save receipt and zero-call assertion.

**Initial execution state:** NOT_RUN. Record actual result separately.

### AC-WRITE-02 - Navigation and newer typing

**Gate:** G1 | **Priority:** P0 | **Applies:** Core next-stage scope

**Given:** A save is delayed while newer text is entered or the user navigates away.

**When:** Resolve the earlier save and return/refresh the original tab.

**Then:** Newer typing is preserved; old replies do not navigate the user unexpectedly or replace newer text; account-scoped tab recovery matches current contract.

**Verification:** Browser test with controlled response order.

**Required evidence:** Before/after draft, receipt and route-state assertions.

**Initial execution state:** NOT_RUN. Record actual result separately.

### AC-WRITE-03 - Uncertain save and receipt replay

**Gate:** G1 | **Priority:** P0 | **Applies:** Core next-stage scope

**Given:** The server persists a save but the response is lost.

**When:** Retry the same logical save using its existing receipt.

**Then:** Exactly the original logical save is recovered without a duplicate entry; unrelated fresh intent remains distinct.

**Verification:** Fault-injected API/browser integration.

**Required evidence:** Persisted row count/identity and receipt replay trace.

**Initial execution state:** NOT_RUN. Record actual result separately.

### AC-WRITE-04 - Conflicts, account switches and erasure

**Gate:** G1 | **Priority:** P0 | **Applies:** Core next-stage scope

**Given:** Two tabs have competing revisions and an old request is in flight.

**When:** Commit a newer revision, switch account or erase data before the old response returns.

**Then:** Stale mutation/response is rejected or ignored per contract; no silent overwrite, cross-account resurfacing or erased-data resurrection.

**Verification:** Controlled concurrency and browser isolation tests.

**Required evidence:** Expected conflict/fence behavior and owner-specific final state.

**Initial execution state:** NOT_RUN. Record actual result separately.

## REQ-CONSENT

### AC-CONSENT-01 - Simple mode remains independent

**Gate:** G1 | **Priority:** P0 | **Applies:** Core next-stage scope

**Given:** AI/search are available but the user selects Simple or only writes.

**When:** Complete the supported scripted path.

**Then:** No model/search request occurs; mode labels are accurate and existing Simple/local-timer semantics are preserved.

**Verification:** Client capture and user-flow check.

**Required evidence:** Zero-call proof and actual resulting mode/state.

**Initial execution state:** NOT_RUN. Record actual result separately.

### AC-CONSENT-02 - Explicit owned-entry sharing

**Gate:** G1 | **Priority:** P0 | **Applies:** Core next-stage scope

**Given:** Owners A/B have distinct entries and chat can be linked or unlinked.

**When:** Try an unlinked chat, an explicit A link and an attempted B link.

**Then:** Unlinked chat receives no journal content; only explicitly selected owned content is shared; cross-owner reference is rejected before provider use.

**Verification:** API contract/security tests.

**Required evidence:** Captured provider payload and rejected cross-owner request.

**Initial execution state:** NOT_RUN. Record actual result separately.

### AC-CONSENT-03 - Separate external-search approval

**Gate:** G1 | **Priority:** P0 | **Applies:** Core next-stage scope

**Given:** AI consent is granted but search consent is absent.

**When:** Request a resource suggestion, then separately grant topic-search consent.

**Then:** No external search before its own approval; afterward only the disclosed topic fields leave; raw journal/history is absent; unavailable search offers the documented alternative.

**Verification:** Boundary tests with payload capture.

**Required evidence:** Consent transitions, approved field allowlist and zero-call denied path.

**Initial execution state:** NOT_RUN. Record actual result separately.

### AC-CONSENT-04 - Revocation and provider failures

**Gate:** G1 | **Priority:** P0 | **Applies:** Core next-stage scope

**Given:** A consented request is pending and the provider may timeout, refuse or return invalid output.

**When:** Revoke, erase or switch account while it is pending; also exercise each terminal provider failure.

**Then:** No new unauthorized dispatch or stale persistence; already transmitted calls are not falsely claimed recalled; saved writing/report survives; fallback is labeled and never counted as AI success.

**Verification:** Fault ordering, schema validation and UI test.

**Required evidence:** Request/result fence assertions and truthful error/fallback state.

**Initial execution state:** NOT_RUN. Record actual result separately.

## REQ-ACT

### AC-ACT-01 - Server lifecycle and optimistic concurrency

**Gate:** G1 | **Priority:** P0 | **Applies:** Core next-stage scope

**Given:** An owned modern activity has known session/conversation revisions.

**When:** Exercise start, pause, resume, finish_early, expire, stop and decline, including stale/repeated receipts.

**Then:** Allowed transitions use server time and expected revisions; invalid/stale commands cannot overwrite canonical state; replay does not duplicate effects; no invented reset path.

**Verification:** Lifecycle/API plus disposable PostgreSQL checks.

**Required evidence:** Transition/receipt evidence and final canonical revisions.

**Initial execution state:** NOT_RUN. Record actual result separately.

### AC-ACT-02 - Timer is not participation

**Gate:** G1 | **Priority:** P0 | **Applies:** Core next-stage scope

**Given:** An activity starts and later reaches its deadline.

**When:** Expire it without a participant report.

**Then:** It reaches awaiting_report as appropriate; no completed participation, positive outcome or Journey success is fabricated.

**Verification:** Deterministic clock test and UI assertion.

**Required evidence:** Session and Journey/report state before/after deadline.

**Initial execution state:** NOT_RUN. Record actual result separately.

### AC-ACT-03 - Report persists before follow-up

**Gate:** G1 | **Priority:** P0 | **Applies:** Core next-stage scope

**Given:** A user reports completed, partial, not_tried or stopped.

**When:** Save the report and force the optional follow-up to fail or be retried.

**Then:** The exact participation report remains; skipped ratings stay unknown; follow-up retries do not duplicate the report or imply activity performance.

**Verification:** API persistence and provider-failure test.

**Required evidence:** Saved report, retry behavior and honest follow-up state.

**Initial execution state:** NOT_RUN. Record actual result separately.

## REQ-OWN

### AC-OWN-01 - Cross-owner and hosted-auth boundary

**Gate:** G1 | **Priority:** P0 | **Applies:** Core next-stage scope

**Given:** Owners A/B have journals, conversations and activities; auth can fail.

**When:** Attempt unauthorized list/get/change/export and service-outage paths.

**Then:** No B content or mutation is available to A; owner comes from auth; production does not fall back to a development identity; responses follow existing non-enumerating behavior.

**Verification:** API/RLS tests plus scoped hosted disposable checks.

**Required evidence:** Two-owner results and actual auth-outage behavior.

**Initial execution state:** NOT_RUN. Record actual result separately.

### AC-OWN-02 - Export correctness

**Gate:** G1 | **Priority:** P0 | **Applies:** Core next-stage scope

**Given:** A has known supported fictional records and B has different records.

**When:** Export A's saved data.

**Then:** The export parses, includes expected supported A records, excludes B/secrets and clearly states scope and any intentionally excluded provider/external copies.

**Verification:** API plus manual export inspection.

**Required evidence:** Redacted schema/count comparison, not raw personal contents.

**Initial execution state:** NOT_RUN. Record actual result separately.

### AC-OWN-03 - Erasure and in-flight resurrection

**Gate:** G1 | **Priority:** P0 | **Applies:** Core next-stage scope

**Given:** A has saved data and delayed writes/results; B remains active.

**When:** Erase A's saved data and release delayed responses.

**Then:** A's supported data and new stage-owned data are removed/fenced; stale work cannot recreate them; B is unchanged; A's sign-in identity remains; replay is safe.

**Verification:** Repository/DB race tests and hosted disposable lifecycle.

**Required evidence:** Count/epoch/fence assertions and signed-in state after data erasure.

**Initial execution state:** NOT_RUN. Record actual result separately.

### AC-OWN-04 - Sensitive-data canaries

**Gate:** G1 | **Priority:** P0 | **Applies:** Core next-stage scope

**Given:** Fictional canary text/token values are placed in controlled inputs/errors.

**When:** Exercise validation, provider failure, telemetry and evidence generation.

**Then:** No journal/chat/action title, credential or raw sensitive payload is present in operational logs, error summaries or public evidence; intentional encrypted backups remain outside Git/evaluation artifacts.

**Verification:** Captured logs and artifact inspection.

**Required evidence:** Canary checks and documented content-free evidence fields.

**Initial execution state:** NOT_RUN. Record actual result separately.

## REQ-CLOSE

### AC-CLOSE-01 - Unavailable until actual prerequisites

**Gate:** G2 | **Priority:** P0 | **Applies:** Core next-stage scope

**Given:** Closure flag/config/migration/hosted gate is missing or incomplete.

**When:** Inspect capability and attempt account closure.

**Then:** Production closure stays unavailable without claiming completion; after verified recovery and compatible migration, the exact staged build may be configured for disposable smoke. Data erasure stays separate. No production promotion before the hosted gate.

**Verification:** Configuration tests and hosted gate review.

**Required evidence:** Capability/readiness evidence, migration identity and blocker if pending.

**Initial execution state:** NOT_RUN. Record actual result separately.

### AC-CLOSE-02 - Fresh sign-in and operation-bound receipt

**Gate:** G1 | **Priority:** P0 | **Applies:** Core next-stage scope

**Given:** A closure request has a bearer identity and controllable AMR/receipt fields.

**When:** Try fresh sign-in, refresh-only old sign-in, wrong owner, stale/invalid receipt and caller user_id.

**Then:** The initial request requires the intended authenticated owner with real sign-in within 5 minutes and exact confirmation/UUID4/256-bit receipt. Token refresh is insufficient. Existing pending/completed closure may continue via the valid 15-minute receipt after Auth deletion invalidates the session; a valid other-owner bearer is rejected. Expired receipt with unresolved post-deletion completion requires operator reconciliation. Admin credentials remain server-only.

**Verification:** Local auth/receipt fault tests plus actual hosted freshness smoke.

**Required evidence:** Case results and exact operation binding, with no secrets retained.

**Initial execution state:** NOT_RUN. Record actual result separately.

### AC-CLOSE-03 - Partial failure and concurrent writes

**Gate:** G1 | **Priority:** P0 | **Applies:** Core next-stage scope

**Given:** Closure can pause before/after Admin deletion while writes race.

**When:** Inject each supported uncertain/failure ordering and retry with the valid receipt.

**Then:** Pending/complete markers block A writes; B works; 204 is withheld until identity and dependent rows are absent; uncertainty remains truthful; no deleted data is restored.

**Verification:** Deterministic fault/concurrency tests.

**Required evidence:** Closure ledger state, dependent-row checks and response transitions.

**Initial execution state:** NOT_RUN. Record actual result separately.

### AC-CLOSE-04 - Actual disposable hosted lifecycle

**Gate:** G2 | **Priority:** P0 | **Applies:** Core next-stage scope

**Given:** Real recovery, compatible migration, staged deployment and scoped credentials are ready.

**When:** Run the existing hosted smoke on two newly tagged fictional accounts, with verified IDs and cleanup scope.

**Then:** Actual Auth/DB behavior and >5-minute refresh-only rejection are verified; fixture accounts are cleaned as authorized; local fakes are not substituted; rollback preserves guards.

**Verification:** Real hosted disposable-account verification.

**Required evidence:** Exact deployment/schema report, account fixture identifiers protected, cleanup result and timestamp.

**Initial execution state:** NOT_RUN. Record actual result separately.

## REQ-REC

### AC-REC-01 - Live encrypted export within guards

**Gate:** G2 | **Priority:** P0 | **Applies:** Core next-stage scope

**Given:** Secure credentials/passphrase, Docker/GPG and the approved source guard are ready.

**When:** Run the existing live export procedure through its secure input mechanism.

**Then:** A consistent intended-source snapshot is encrypted before disk write; size/version/dependency guards hold; no secret/content leaks into agent logs or Git. Enforce PostgreSQL 17 and 128 MiB guards; storage-object/Vault dependencies stop the procedure.

**Verification:** Actual guarded operation; fictional dry run is separate.

**Required evidence:** Content-free source identity, archive digest/location reference, snapshot time and encryption verification.

**Initial execution state:** NOT_RUN. Record actual result separately.

### AC-REC-02 - Isolated local restore verification

**Gate:** G2 | **Priority:** P0 | **Applies:** Core next-stage scope

**Given:** An actual encrypted export is available.

**When:** Restore only into the runbook's unique disposable isolated target and run its invariants.

**Then:** Decryption/schema/count/integrity checks pass without touching production; record duration and coverage; do not claim hosted Auth recovery/PITR.

**Verification:** Actual isolated restoration.

**Required evidence:** Verification receipt, target-isolation evidence, durations and cleanup result.

**Initial execution state:** NOT_RUN. Record actual result separately.

### AC-REC-03 - Deletion reconciliation and fail-closed reopening

**Gate:** G2 | **Priority:** P0 | **Applies:** Core next-stage scope

**Given:** A backup predates erasure/closure and deletion history may be incomplete.

**When:** Rehearse recovery with complete and incomplete post-snapshot history using fictional fixtures.

**Then:** Complete history removes later-deleted data before access; incomplete/unprovable history leaves recovered data inaccessible; no invented independent deletion-ledger claim.

**Verification:** Controlled restore/deletion drill plus actual process review.

**Required evidence:** Decision record and demonstrated access quarantine.

**Initial execution state:** NOT_RUN. Record actual result separately.

### AC-REC-04 - Cadence, generations and honest recovery scope

**Gate:** G2 | **Priority:** P1 | **Applies:** Core next-stage scope

**Given:** One or more actual verified generations exist.

**When:** Review retention and the recovery runbook.

**Then:** Manual weekly/pre-change cadence and three verified generations are accurately reported; zero/missing generations remain visible; actual latest snapshot defines exposure; no automatic-backup or full-DR claim without evidence. Three generations is a rolling target, not three duplicate pre-release runs; never displace the sole good generation with a failed one.

**Verification:** Operational evidence review.

**Required evidence:** Generation inventory, retention/cleanup procedure, latest verification and known limits.

**Initial execution state:** NOT_RUN. Record actual result separately.

## REQ-WEB

### AC-WEB-01 - Frontend origin and CSP agreement

**Gate:** G1 | **Priority:** P1 | **Applies:** Core next-stage scope

**Given:** Default and explicitly permitted distinct frontend/backend auth origins are configured.

**When:** Build/serve the exported frontend and exercise browser auth network destinations.

**Then:** The actual permitted origin is allowed, or unsupported mismatch fails build clearly; no wildcard introduced; disallowed origins remain blocked.

**Verification:** Build plus browser CSP checks.

**Required evidence:** Two configuration results and actual response CSP.

**Initial execution state:** NOT_RUN. Record actual result separately.

### AC-WEB-02 - Same-byte script policy and regression

**Gate:** G1 | **Priority:** P1 | **Applies:** Core next-stage scope

**Given:** The exported page is served under the updated policy.

**When:** Check allowed scripts, injected/unapproved script and relevant rolling/static cache behavior.

**Then:** Intended scripts work and unapproved script remains blocked; hashes apply to the same bytes served; no broad unsafe relaxation.

**Verification:** Focused public-web-security/browser test.

**Required evidence:** Policy/hash/assertion output tied to build.

**Initial execution state:** NOT_RUN. Record actual result separately.

### AC-WEB-03 - Plain public phone/SMS target

**Gate:** G1 | **Priority:** P1 | **Applies:** Core next-stage scope

**Given:** The public information component is rendered.

**When:** Inspect phone/SMS href and open the link chooser on a supported device without initiating a call.

**Then:** Displayed number and target agree; no HTTP attribution query in the phone target; no test call is placed.

**Verification:** Static/component check and manual device inspection.

**Required evidence:** Sanitized target and device/browser observation.

**Initial execution state:** NOT_RUN. Record actual result separately.

## REQ-EVAL

### AC-EVAL-01 - New source freeze and original controls

**Gate:** G3 | **Priority:** P0 | **Applies:** Core next-stage scope

**Given:** Inference-relevant repairs are complete and old bundles exist.

**When:** Compare source/packet, prepare a new named bundle and update the wrapper explicitly.

**Then:** Old bundles remain immutable; new hashes bind repaired source/prompts/catalog/runner/lockfiles/requests; 100 cases/50 families/40-60 split retained; route mix recomputed; ledger path/cap preserved. Authoritative ledger history survives worktree changes; frozen absolute path must match actual runner ledger, with no fresh allowance.

**Verification:** Prepare/guard tests and reviewed freeze diff.

**Required evidence:** Manifest, source/packet comparison and explicit wrapper target.

**Initial execution state:** NOT_RUN. Record actual result separately.

### AC-EVAL-02 - Cost, routing and non-repeat execution guards

**Gate:** G3 | **Priority:** P0 | **Applies:** Core next-stage scope

**Given:** Frozen runner can be exercised with fake endpoints/accounting failures.

**When:** Test unavailable ZDR/structured endpoint, identity mismatch, unresolved cost, budget breach, transport interruption and existing result file.

**Then:** No forbidden/extra call proceeds; no fallback, ledger reset, duplicate case/stage attempt or rerun of an existing bundle; USD10 total includes failed attempts.

**Verification:** Existing evaluator-control tests with failure injection.

**Required evidence:** Accounting/call-count traces and stop reasons without credentials.

**Initial execution state:** NOT_RUN. Record actual result separately.

### AC-EVAL-03 - Predeclared release policy

**Gate:** G3 | **Priority:** P0 | **Applies:** Core next-stage scope

**Given:** Reserved outputs have not been viewed.

**When:** Record PRD Section 9 thresholds, expected-route completeness, human sample rule and decision algorithm before execution.

**Then:** Policy is versioned and unambiguous; it is labeled new, not an existing runner gate; candidate report completion alone cannot produce PASS. Mean population and family weighting are distinct from all-expected-normal case denominator N; empty populations cannot pass.

**Verification:** Decision-validator unit examples and policy review.

**Required evidence:** Timestamped/versioned policy and examples where poor/critical/incomplete results fail or block.

**Initial execution state:** NOT_RUN. Record actual result separately.

### AC-EVAL-04 - Actual run and full denominators

**Gate:** G3 | **Priority:** P0 | **Applies:** Core next-stage scope

**Given:** Current session has paid-run authorization, secure credentials and a valid fresh bundle.

**When:** Execute the frozen candidate/judge run once under its existing controls.

**Then:** All planned cases/statuses and charges are accounted for; exceptions remain visible; no historical W/L/T substitution; missing/invalid results block qualification; actual models/providers/latencies recorded.

**Verification:** Actual paid run; no mocked substitute.

**Required evidence:** Immutable results, ledger reconciliation, exact source/provider identity and generated report.

**Initial execution state:** NOT_RUN. Record actual result separately.

### AC-EVAL-05 - Independent human review and quality decision

**Gate:** G3 | **Priority:** P0 | **Applies:** Core next-stage scope

**Given:** Actual run exists and a real independent reviewer is available.

**When:** Grade the preselected 20 outputs blinded to judge scores/model identity, then inspect support/flagged cases and apply Section 9.

**Then:** All 20 prespecified cases remain in the human denominator; blind grades precede adjudication. Apply Section 9 exactly: confirmed critical failure FAILS this candidate, indeterminate concern BLOCKS, only documented unsupported judge flags may clear with original grades preserved. Human review cannot repair invalid/missing model results. Frozen runner reports stay unchanged; attach separate human/decision records.

**Verification:** Human review plus deterministic decision computation.

**Required evidence:** Review sheet, sample-selection rule, unique reviewed counts, rationales and decision.

**Initial execution state:** NOT_RUN. Record actual result separately.

### AC-EVAL-06 - Holdout discipline and claims

**Gate:** G3 | **Priority:** P0 | **Applies:** Core next-stage scope

**Given:** Results may be weak, incomplete or inconclusive.

**When:** Review follow-up changes and publication text.

**Then:** No tuning/repeated run of reserved cases to manufacture a pass; revised development and genuinely new reserved evidence are planned when needed; AI-judged, human, hosted and clinical status stay distinct.

**Verification:** Evidence/version review.

**Required evidence:** Preserved negative results, change history and accurately scoped claims.

**Initial execution state:** NOT_RUN. Record actual result separately.

## REQ-REL

### AC-REL-01 - Required current-candidate verification

**Gate:** G1 | **Priority:** P0 | **Applies:** Core next-stage scope

**Given:** Applicable changes are ready.

**When:** Run focused checks, then the repository's required release/CI checks once for the candidate.

**Then:** All required checks pass or the candidate remains gated; real PostgreSQL checks use a proven disposable target; skipped/mocked checks are disclosed; generated API contracts match.

**Verification:** Actual commands and CI results.

**Required evidence:** SHAs, command/exit outputs, test IDs, environment and skips.

**Initial execution state:** NOT_RUN. Record actual result separately.

### AC-REL-02 - Final evidence manifest and hosted identity

**Gate:** G2 | **Priority:** P0 | **Applies:** Core next-stage scope

**Given:** Source and artifacts are finalized and authorized deployment checks are available.

**When:** Generate/verify the final manifest and inspect the actual deployment.

**Then:** Every claim binds the relevant source/inference/schema/environment; hosted ID is observed rather than assumed; historical manifests remain labeled; no final edits silently invalidate evidence.

**Verification:** Hash verification and hosted identity/smoke.

**Required evidence:** Current release record with CI/model/recovery/hosted evidence separately linked.

**Initial execution state:** NOT_RUN. Record actual result separately.

### AC-REL-03 - Rollback and support procedure

**Gate:** G2 | **Priority:** P0 | **Applies:** Core next-stage scope

**Given:** A candidate, previous application and additive data guards exist.

**When:** Rehearse/inspect rollback and one provider/auth/save failure support scenario.

**Then:** Affected optional capability can be disabled; data/closure guards and deletion history preserved; no rollback resurrects records or replays external writes; support does not require journal content.

**Verification:** Disposable rehearsal/runbook exercise.

**Required evidence:** Rollback target, exact procedure, observed scope and unresolved limits.

**Initial execution state:** NOT_RUN. Record actual result separately.

## REQ-UX

### AC-UX-01 - Accessible core paths

**Gate:** G4 | **Priority:** P1 | **Applies:** Core next-stage scope

**Given:** Core pages are available on the candidate.

**When:** Run keyboard-only, visible-focus, labels/status announcement, 320px layout and 200% zoom checks on the three tasks.

**Then:** Critical actions and recovery remain operable/readable; no clipped required controls or keyboard trap; scoped defects corrected and browser coverage reported.

**Verification:** Manual inspection plus meaningful existing browser checks.

**Required evidence:** Checklist with device/browser/build and observed issues.

**Initial execution state:** NOT_RUN. Record actual result separately.

### AC-UX-02 - Five real standardized sessions

**Gate:** G4 | **Priority:** P1 | **Applies:** Core next-stage scope

**Given:** Five adults have agreed to observed fictional-data sessions.

**When:** Run the same 5/7/6-minute task caps and neutral instructions; record assistance/stops.

**Then:** Five real participant records exist with familiarity, version, first exposure/retest, outcome/time and mode; no coaching is counted unassisted; no synthetic observations.

**Verification:** Actual moderated sessions.

**Required evidence:** Pseudonymous task sheet, consent accounting and raw counts.

**Initial execution state:** NOT_RUN. Record actual result separately.

### AC-UX-03 - First-use task and comprehension gates

**Gate:** G4 | **Priority:** P1 | **Applies:** Core next-stage scope

**Given:** The five sessions are complete on the assessed candidate or explicitly separated versions.

**When:** Apply PRD thresholds and repair repeated friction.

**Then:** 5/5 unassisted save/retrieve and >=4/5 chosen-mode reflection/action with persisted retrieval meet the first-use gate. All 5 ultimately complete export/open/erasure and understand the sign-in boundary; initial assistance stays visible. Simple is not counted as AI. Functional privacy failure or repeated unresolved navigation confusion blocks expansion.

**Verification:** Observed task scoring and separate retest.

**Required evidence:** Initial and retest tables, requested/actual modes, comprehension results and fix decisions.

**Initial execution state:** NOT_RUN. Record actual result separately.

## REQ-OBS

### AC-OBS-01 - Opt-in and minimal event allowlist

**Gate:** G4 | **Priority:** P0 | **Applies:** Only if product research telemetry is selected; manual-only measurement may mark this instrumentation criterion NOT_APPLICABLE with a scope record. Consent/withdrawal/retention for real notes/mappings remain required under UX/PILOT/OBS-03.

**Given:** Research telemetry is optional and a fictional canary fixture exists.

**When:** Decline consent, enable it, generate events and inspect the exported fields.

**Then:** No research events before consent or after revocation; app works without telemetry; no backfill; only allowed pseudonymous fields; no journal/action/mood/token/IP content.

**Verification:** Boundary and payload tests plus disclosure review.

**Required evidence:** Consent version/state, allowed-field schema and canary results.

**Initial execution state:** NOT_RUN. Record actual result separately.

### AC-OBS-02 - Metric semantics and deduplication

**Gate:** G5 | **Priority:** P1 | **Applies:** Core next-stage scope

**Given:** Fixtures include retries, Simple fallback, timer expiry, zero activity, missing events, late activation and changed builds.

**When:** Calculate pilot metrics from those fixtures.

**Then:** Retries do not multiply workflows; timer/export is not participation; mature/partial/unknown denominators are distinct; cost includes unsuccessful cohort attempts; zero denominator is undefined; mode/build preserved.

**Verification:** Meaningful calculation examples against explicit expected totals.

**Required evidence:** Metric definitions, fixture totals and reproducible calculation.

**Initial execution state:** NOT_RUN. Record actual result separately.

### AC-OBS-03 - Withdrawal and retention enforcement

**Gate:** G4 | **Priority:** P0 | **Applies:** Core next-stage scope

**Given:** A participant has consented events, notes and contact mapping.

**When:** Withdraw or expire the declared retention window.

**Then:** Collection stops; linked research events/notes are removed within 24h of withdrawal; raw research records/mapping deleted no later than 30 days after cohort close; summaries recomputed; product/provider retention not misrepresented.

**Verification:** Test-clock removal and actual operating procedure.

**Required evidence:** Deletion job/manual procedure, dry-run result and published retention wording.

**Initial execution state:** NOT_RUN. Record actual result separately.

## REQ-PILOT

### AC-PILOT-01 - Fresh enrollment and mature windows

**Gate:** G5 | **Priority:** P1 | **Applies:** Core next-stage scope

**Given:** Five session participants may accept or decline a new pilot.

**When:** Record invitations/enrollment and observe each 168-hour window.

**Then:** Actual enrolled/withdrawn/opted-out/fully observed/partial counts reconcile; five sessions are not assumed five pilot users; no daily-use pressure; zero-use participants remain in eligible denominators.

**Verification:** Actual recruitment and observation records.

**Required evidence:** Consent ledger and complete per-participant window accounting.

**Initial execution state:** NOT_RUN. Record actual result separately.

### AC-PILOT-02 - Real-use report and stop rules

**Gate:** G5 | **Priority:** P0 | **Applies:** Core next-stage scope

**Given:** A released candidate is used voluntarily.

**When:** Collect bounded support/usefulness observations and inspect critical issues.

**Then:** Report real positive/negative/non-use results and missingness; isolation, confirmed-save loss, erased-data resurrection or severe routing issue stops affected use/expansion; no clinical/causal/adoption exaggeration.

**Verification:** Human observations plus incident review.

**Required evidence:** Cohort report, issue log, build changes and actual stop/resume decisions.

**Initial execution state:** NOT_RUN. Record actual result separately.

### AC-PILOT-03 - Next-investment decision

**Gate:** G5 | **Priority:** P1 | **Applies:** Core next-stage scope

**Given:** The planned five-person enrolled cohort has complete accounting.

**When:** Apply the predeclared useful-outcome plus later-day-return rule.

**Then:** Continue only with >=3 of the same 5 enrolled participants each meeting both specific-usefulness and eligible later-day-return conditions, with no unresolved P0. Record met/not met/inconclusive/incomplete cohort and unknowns; no denominator changes to force success.

**Verification:** Documented product decision.

**Required evidence:** Counts meeting each/both conditions, unknowns and continue/simplify/stop rationale.

**Initial execution state:** NOT_RUN. Record actual result separately.

## REQ-INT

### AC-INT-01 - Discovery and provider feasibility

**Gate:** G6a | **Priority:** P1 | **Applies:** G6a scopes bounded implementation; G6b gates user enablement. Default integration deferred/disabled.

**Given:** Core blockers are resolved and an external-task need is being considered.

**When:** Review at least two actual matching user problems, select one destination/adapter and assess official account/API contract and access feasibility; any disposable spike requires its own scoped authorization.

**Then:** G6a scope decision records one bounded need and feasible implementation path. Permission scope is described honestly; no extra adapter/two-way sync. Implementation may proceed after scope approval; actual provider/acceptance proof is G6b, not a prerequisite for starting the build.

**Verification:** Discovery, official-contract/access review and optional scoped feasibility spike.

**Required evidence:** Scope decision, official provider contracts, chosen account type and constraints.

**Initial execution state:** NOT_RUN. Record actual result separately.

### AC-INT-02 - Owned approved immutable payload

**Gate:** G6b | **Priority:** P0 | **Applies:** G6a scopes bounded implementation; G6b gates user enablement. Default integration deferred/disabled.

**Given:** An owned activity and current connection/revisions exist.

**When:** Submit missing approval, wrong owner, stale revisions, unknown fields and a valid reviewed snapshot. Force a failure during transaction acceptance.

**Then:** Only the valid owned ActivitySession.id and expected activity/conversation/connection revisions are accepted; owner derived from auth; exact title/minimal fields frozen; no journal/history included or catalog mutation. Owner/revision/fence validation, local/semantic deduplication, frozen approval and outbox insertion commit atomically. Forced rollback creates no accepted intent or provider write. Only exact reviewed title, optional due fields and disclosed opaque correlation leave; no extra journal/chat context.

**Verification:** API transaction/payload tests.

**Required evidence:** Allowed/denied cases and captured exact adapter payload.

**Initial execution state:** NOT_RUN. Record actual result separately.

### AC-INT-03 - Local replay and two-tab duplicate guard

**Gate:** G6b | **Priority:** P0 | **Applies:** G6a scopes bounded implementation; G6b gates user enablement. Default integration deferred/disabled.

**Given:** One activity has a pending or succeeded export.

**When:** Double click, replay the same key with same/changed payload, and submit a new key from another tab.

**Then:** Same intent returns same receipt; changed key payload conflicts; cross-tab semantic guard prevents another active/uncertain/succeeded export for the same activity/account/list; no remote exactly-once claim.

**Verification:** Concurrent API/DB tests.

**Required evidence:** Receipt IDs, row uniqueness and adapter call counts.

**Initial execution state:** NOT_RUN. Record actual result separately.

### AC-INT-04 - Lost response and expired worker lease

**Gate:** G6b | **Priority:** P0 | **Applies:** G6a scopes bounded implementation; G6b gates user enablement. Default integration deferred/disabled.

**Given:** Provider creation may succeed before the response or worker is lost.

**When:** Race two workers for the same queued intent; drop a response or crash after the durable dispatch claim, expire its lease and release a stale worker result.

**Then:** Queued-to-sending atomically claims one attempt against current erasure epoch/connection generation. Only the valid claim dispatches; stale workers cannot overwrite newer state. Crash/lease expiry after claim is uncertain unless non-transmission is affirmatively proved; no automatic repeat-create/reset-to-queued. No DB transaction stays open across network I/O.

**Verification:** Controlled adapter/worker fault injection.

**Required evidence:** State/lease trace and zero additional POST assertions.

**Initial execution state:** NOT_RUN. Record actual result separately.

### AC-INT-05 - Positive-only reconciliation

**Gate:** G6b | **Priority:** P0 | **Applies:** G6a scopes bounded implementation; G6b gates user enablement. Default integration deferred/disabled.

**Given:** An uncertain intent has an opaque correlation marker.

**When:** Exercise one match, multiple matches, zero matches, partial pages, moved task and unavailable credentials.

**Then:** One exact positive match can establish creation; every other ambiguous result stays unresolved or reports anomaly; no negative lookup permits repeat POST; no silent remote deletion.

**Verification:** Adapter reconciliation tests plus real marker write/read probe.

**Required evidence:** Full marker/correlation behavior and bounded read/call trace.

**Initial execution state:** NOT_RUN. Record actual result separately.

### AC-INT-06 - Bounded known-safe retry

**Gate:** G6b | **Priority:** P0 | **Applies:** G6a scopes bounded implementation; G6b gates user enablement. Default integration deferred/disabled.

**Given:** Dispatcher has a predeclared approval lifetime and finite retry budget.

**When:** Return documented throttling, verified pre-send failure, ambiguous 5xx, timeout and expired approval.

**Then:** Only demonstrably non-creating failures can retry; Retry-After respected; expired approvals stop; SDK retries cannot bypass policy; ambiguous outcomes remain uncertain.

**Verification:** Fault injection and SDK/adapter configuration review.

**Required evidence:** Retry timing/attempt limits and no-hidden-retry evidence.

**Initial execution state:** NOT_RUN. Record actual result separately.

### AC-INT-07 - OAuth, account binding and disconnect

**Gate:** G6b | **Priority:** P0 | **Applies:** G6a scopes bounded implementation; G6b gates user enablement. Default integration deferred/disabled.

**Given:** An account-scoped connection exists.

**When:** Exercise invalid/replayed callback state, refresh races, revoked credentials, disconnect and reconnect to another account.

**Then:** Unsupported callbacks rejected; tokens server-side; late refresh cannot resurrect disconnected credentials; no silent retargeting; local disconnect does not claim provider grant revocation or call broad revokeSignInSessions.

**Verification:** OAuth/connection tests and disposable account flow.

**Required evidence:** Connection-generation/state results, safe scope disclosure and secret absence.

**Initial execution state:** NOT_RUN. Record actual result separately.

### AC-INT-08 - Revocation and erasure race

**Gate:** G6b | **Priority:** P0 | **Applies:** G6a scopes bounded implementation; G6b gates user enablement. Default integration deferred/disabled.

**Given:** An export is queued or in-flight and source/account can be erased.

**When:** Commit revocation/erasure before dispatch and again after dispatch, then release a late provider result.

**Then:** Pre-dispatch fences suppress sends; in-flight creation is not falsely cancelled; no late result restores erased title/activity; future work blocked; minimal suppression record only as policy permits.

**Verification:** Forced race ordering with erasure epochs.

**Required evidence:** State/transmission/persistence assertions and external-copy disclosure.

**Initial execution state:** NOT_RUN. Record actual result separately.

### AC-INT-09 - Restored outbox quarantine

**Gate:** G6b | **Priority:** P0 | **Applies:** G6a scopes bounded implementation; G6b gates user enablement. Default integration deferred/disabled.

**Given:** An old backup contains credentials and queued/sending intents predating deletion or disconnect.

**When:** Restore in isolation and attempt dispatch before reconciliation.

**Then:** Outbound work stays disabled/quarantined, credentials unusable; stale sending does not repost; incomplete deletion/consent history blocks reopening; only freshly authorized new work can proceed.

**Verification:** Recovery/dispatcher integration test.

**Required evidence:** Quarantine state, zero outbound calls and reconciliation decision.

**Initial execution state:** NOT_RUN. Record actual result separately.

### AC-INT-10 - Receipt wording and external copies

**Gate:** G6b | **Priority:** P1 | **Applies:** G6a scopes bounded implementation; G6b gates user enablement. Default integration deferred/disabled.

**Given:** An intent is queued, created, uncertain or later externally moved/deleted.

**When:** Read the UI and erase local source data.

**Then:** 202 means queued, succeeded means creation established; neither means activity completed or still synchronized; user was told remote copy remains; no implicit delete/recreate.

**Verification:** UI/content plus status-boundary test.

**Required evidence:** State labels, activity/report unchanged and no unapproved remote mutations.

**Initial execution state:** NOT_RUN. Record actual result separately.

### AC-INT-11 - Optional due time and privacy

**Gate:** G6b | **Priority:** P0 | **Applies:** G6a scopes bounded implementation; G6b gates user enablement. Default integration deferred/disabled.

**Given:** Due time may be enabled and payload/logging are instrumented.

**When:** Test omitted due time, timezone change, DST gap/overlap and content canaries.

**Then:** Title-only default works; enabled time fields round-trip reviewed meaning or reject ambiguity; no surprise reminders; logs omit title/token/private content; only approved remote fields leave.

**Verification:** Date/payload/log tests.

**Required evidence:** Timezone cases and captured allowlist/canary proof.

**Initial execution state:** NOT_RUN. Record actual result separately.

### AC-INT-12 - Actual provider and release evidence

**Gate:** G6b | **Priority:** P0 | **Applies:** G6a scopes bounded implementation; G6b gates user enablement. Default integration deferred/disabled.

**Given:** Offline checks pass and one disposable external account/task test is authorized.

**When:** Create/reconcile the reviewed fictional task, inspect actual permissions, disconnect and perform authorized fixture cleanup.

**Then:** Real provider behavior matches the contract or feature stays off; fake/live evidence separated; actual adapter/schema/deployment identified; rollback disables dispatch without replay.

**Verification:** Scoped real-provider smoke and release review.

**Required evidence:** Exact candidate/provider test, remote fixture receipt, cleanup and rollback record.

**Initial execution state:** NOT_RUN. Record actual result separately.

## REQ-HANDOFF

### AC-HANDOFF-01 - Current case study and operational package

**Gate:** G7 | **Priority:** P1 | **Applies:** Core next-stage scope

**Given:** Applicable release/pilot evidence exists or is explicitly blocked.

**When:** Prepare the current overview,90-second demo,guide,runbook and evidence matrix.

**Then:** All claims trace to exact build/data/denominator; unfinished gates visible; historical evidence separate; no private content; no invented user/time-saved/clinical metrics.

**Verification:** Artifact/claim review.

**Required evidence:** Published-ready drafts, source links and release-evidence record.

**Initial execution state:** NOT_RUN. Record actual result separately.

### AC-HANDOFF-02 - Personal ownership demonstration

**Gate:** G7 | **Priority:** P1 | **Applies:** Core next-stage scope

**Given:** Yusen has participated in a meaningful implementation slice.

**When:** Explain one request path and defect fix in five minutes, then make a small controlled variation and describe verification.

**Then:** Explanation covers owner/consent/state/provider/error boundaries and the actual change; AI-assisted work is described accurately; a recording/script alone is not proof of code ownership.

**Verification:** Actual walkthrough and small local exercise.

**Required evidence:** Concise self-record/reviewer notes, explained diff and observed exercise result.

**Initial execution state:** NOT_RUN. Record actual result separately.

## Worked metric acceptance example

This is a fictional calculation fixture, not pilot evidence. Assume three fully followed workflows on one build: W1 requests AI and succeeds after two attributable calls costing USD 0.03 and USD 0.04; W2 requests AI, incurs USD 0.05 and fails; W3 uses Simple mode and succeeds without a provider call. Both successful workflows have an explicit persisted action.

Expected: overall completion 2/3; AI completion 1/2; Simple completion 1/1; total provider cost USD 0.12; provider cost per completed AI workflow USD 0.12/1 = USD 0.12. If reporting cost across all completed modes, label USD 0.12/2 = USD 0.06 separately. Do not discard W2's cost or call W3 an AI success.

For participant metrics, a fully observed consented participant with zero activity remains in the eligible denominator. An opted-out/unobserved participant remains in enrollment accounting but is not classified as zero use. A late activation without a full 24-hour return opportunity is shown separately. If no AI workflow completes, cost per AI completion is undefined, not zero.

## Release decision

Do not turn the number of passing criteria into a model-quality percentage or user-success rate. Evaluate each gate with its own denominator and evidence. Preserve failed, blocked and not-run rows. A new code fix, schema change or altered inference path may invalidate part of the previous evidence; document the mapping and repeat the affected required checks.
