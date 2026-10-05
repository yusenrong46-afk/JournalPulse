# JournalPulse pre-evaluation audit

October 5, 2026 · America/Halifax · current workspace, not a new deployment

**Verdict:** the audited candidate passes the local software checks. Several material
bugs were repaired. The upgraded product is **not yet ready for release sign-off**:
its model-quality evaluation is incomplete, and the existing Preview still serves
the earlier product. Manual evaluation of that Preview will not exercise these fixes.

Start with [the audit-only code changes](../artifacts/pre-evaluation-audit/changes.html)
or the [combined patch](../artifacts/pre-evaluation-audit/audit-only.patch). The
[historical response comparison](../artifacts/pre-evaluation-audit/audit-corrected-prior-response-comparison.html)
contains previously captured responses with corrected evidence checks; it is **not**
a fresh run of the updated prompt.

## Scope and method

Reviewed frontend flows, API contracts, SQLite and PostgreSQL behavior, ownership
and deletion, authentication, timers and retries, model requests and output
validation, journal context, search consent and resource constraints, safety
routing, evaluation methodology, dependencies, packaging, CI and deployment
configuration. Parallel reviewers reproduced defects, added focused regressions,
and checked fixes before the combined application tests.

The starting workspace already contained substantial uncommitted work. A 236-file
source snapshot and hashes preserve that starting point. The linked patch contains
only this audit's changes. Earlier benchmark artifacts and the nine previously
applied migration files remain unchanged. Only the never-deployed activity migration
was amended. No deployment, hosted migration, user-data inspection, or paid model/search
call was made. The cloud-environment-onboarding setup skill guided environment and
verification practices.

## Before and after

P1 means a material user or evidence-integrity failure; P2 means a narrower reliability,
configuration, or correctness issue. These are engineering priorities, not clinical
risk ratings. Every repair below is local until a later verified deployment.

| Area | Before | After |
| --- | --- | --- |
| P1: inline search | The browser put constraint fields at the request root; the strict API returned 422 before searching. | Correct nested payload; a real browser/API/database test searches, refines, saves a signed result and starts it. |
| P1: activity replacement | An unstarted old offer could hide a new recommendation and still start with the old duration. | Changed recommendations withdraw only obsolete unstarted offers. Active sessions remain intact unless explicitly stopped. |
| P1: explicit safety language | Bare “I don't feel safe” and direct “I will/I'll kill myself/end my life” could enter ordinary chat. | Narrow phrase repairs route these cases to support with zero model calls; matched denials do not cancel separate affirmative risk. |
| P1: evaluation safety evidence | A successful parser check could stand in for an unobserved critical language judgment. | Accepted critical language requires an observed judgment tied to the actual response. Proven provider refusals remain a separate boundary result. |
| P2: follow-up recovery | An interrupted worker could leave a saved report displaying “generating” indefinitely. | An expired lease becomes visibly retryable; atomic takeover and the three-attempt bound remain enforced. |
| P2: follow-up context | Identical reports for different selected activities produced identical model requests. | Bounded context identifies the actual activity, goal and configured duration as untrusted data. It includes no owner IDs, receipts or URLs. |
| P2: stop parity | A follow-up's explicit pause could hide controls while another SQLite session kept running. | SQLite now stops affected sessions consistently with PostgreSQL. |
| P2: feeling correction | An empty new feelings list restored the previous inferred label. | Empty suggestions clear stale inferences; explicit user-confirmed feelings remain separate. |
| P2: resource constraints | A website containing videos could be treated as no-audio/no-video merely because its container was a website. | Unknown capabilities fail hard format constraints; explicit reviewed metadata is required. |
| P2: preference response | PostgreSQL could return a pre-trigger card or retain an old card after a mode switch. | Preference changes return the committed state and clear outdated cards, matching SQLite. |
| P2: message-count integrity | Direct owner deletion of individual message rows could reset the stored-turn count. | Authenticated partial-message DELETE is revoked. Supported whole-chat, source and account deletion still work. No cross-owner access was demonstrated. |
| P2: check-in retry | After a lost save response, edited answers could reuse an old receipt and appear saved while the original answers remained stored. | Ambiguous retries preserve the exact submitted answers and explain the retry; definite validation failures remain editable. |
| P2: input contracts | Search examples could fail validation; input limits differed from the API; validation arrays became `[object Object]`. | Examples and limits match the API; bounded, readable errors exclude submitted private values. |
| P2: evaluation gates | Nested statuses were silently ignored; naturalness pooled development and holdout cases; adaptive observations could pass without all judgments. | Strict status contract, complete heldout-only release metrics, and complete observed adaptive pairs plus judgments are required. |
| P2: judgment identity | Old scores could attach to changed responses sharing a case ID. | Judgments bind the scenario, exact outputs/transcripts and model request identities. Old evidence needs a verified adapter. |
| P2: readiness | Empty chat models and explicitly missing static exports could report ready; malformed production CORS settings passed validation. | These conditions fail readiness/configuration checks. Legitimate backend-only and local HTTP development remain supported. |
| P2: packaging/setup | Generated duplicate code and test/evaluation files entered Vercel uploads; Render omitted a required signing-key field. | Upload exclusions retain runtime assets while removing those extras; the Render Blueprint declares the required secret. |
| Dependencies | Optional research dependencies GitPython 3.1.51 and urllib3 2.7.0 had published advisories. | Targeted lock updates to 3.2.0 and 2.8.0; all other package records unchanged. All 79 locked Python registry packages now report zero advisories in the checked source. |
| Product documentation | Me and README overstated reviewed-only selection and safety detection, and described only the older flow. | Copy distinguishes AI/guided modes, snippet-only search, limited phrase detection, and local versus hosted versions. |

The activity-context correction advances the packaged prompt/skill to
`guided-action-2026-10-05.3`. Its SHA-256 is
`abb73f69089a8591e5f30f1ac62e4557cd0a62f45d408d8fdf718f1bd4808270`.
The built Python wheel contains these exact skill bytes. Existing live response
evidence was captured with `.2`, so it cannot establish `.3` language quality.

## Verification

| Check | Result |
| --- | --- |
| Full Python suite | **700 passed**; one upstream TestClient deprecation warning |
| Coverage with branch measurement | **90.83%**, above the 80% CI floor; coverage measures exercised code, not answer quality |
| Frontend unit tests | **108 passed**, 20 files |
| Desktop/mobile browser tests | **46 passed**, including 18 automated accessibility checks; no retries or skipped tests |
| Full browser → FastAPI → PostgREST → PostgreSQL tests | **18 passed**, including inline search and obsolete-offer replacement |
| PostgreSQL verifier suites | **208 passed assertions/checks** across base schema (100), journals (17), integrity (19) and activities (72) |
| Static checks | Ruff, mypy, ESLint, TypeScript and generated OpenAPI/type consistency pass |
| Build and resources | Node 22 production static export, Python wheel, packaged skill identity, and all 35 catalog resources pass |
| Corrected historical HTML | Chromium checks pass for 60 cases, eight adaptive sections, filters, visible evidence disclaimer, review export and mobile layout |
| Python dependencies | All 79 locked registry versions queried successfully: zero reported advisories after repair |
| JavaScript production dependencies | Zero reported advisories in `npm audit --omit=dev` |

The prior recorded backend run had 617 passing tests and 90.67% coverage; the
frontend audit baseline had 103 unit tests, and the prior integration run had 16.
These are different source versions, not a claim that a larger test count alone
proves better quality. The new tests reproduce specific failures.

Integration uses real local PostgreSQL, PostgREST, migrations and the API, with
named stand-ins for auth token issuance, Luna and Brave. It establishes application
behavior, not live provider quality. Retention SQL checks use cron metadata
stand-ins; hosted scheduled-job health was separately observed. Automated
accessibility checks do not establish full WCAG conformance. The optional research
app was not run; patched package APIs were smoke-tested in isolation and all-extras
resolution was checked without changing the project environment.

## Hosted state observed

Read-only checks at **15:25 Halifax / 18:25 UTC** found both existing sites READY,
with `/health` and `/ready` returning 200. Model status is explicitly
`configured:not_probed`; no paid generation was performed.

- Preview: `dpl_Ey6QzvQBWwpXaqwyT64N8rhR9d5n`,
  <https://journalpulse-preview-yusenrong46-9212s-projects.vercel.app>.
  Retention was scheduled and healthy, with a successful 18:15 UTC run.
- Production: `dpl_57Rut4UoR24LsTRjsA7nEpiBQycU`,
  <https://journalpulse.vercel.app>. Its older readiness response lacks the newer
  retention diagnostics.
- The shared Supabase project has the older working schema. The candidate activity
  tables, RPCs and triggers are absent: **the new migration is not applied**.

These observations do not mean the hosted sites contain this audit's fixes.

## Remaining issues and explicit limits

1. **Model evaluation is incomplete and historical.** The `.2` comparison has
   15/20 judged quality pairs, one accepted critical case without a language
   judgment, 7/8 baseline and 5/8 candidate adaptive trajectories, and zero fresh
   adaptive judgments. Partial gains cannot pass the release gate. `.3` needs
   appropriately frozen, current evidence and a holdout policy that accounts for
   cases already inspected. A browser test cannot replace the missing judgments.
2. **No fresh release freeze.** Audit fixes intentionally invalidate the previous
   candidate/upload hashes. Generate a new source and migration freeze as part of
   a later release; preserve the old evidence as history.
3. **Budget reconciliation.** The retained ledger records $22.40 reserved; current
   attempt rates imply $22.75. The $0.35 difference is inherited from the opening
   balance. Both are below $25, but the basis needs reconciliation before more paid
   work. Counts remain Luna 219/600, teacher 171/180 and Brave 2/6. No cap was raised.
4. **Safety remains a limited English phrase router.** Indirect distress can be
   missed and quoted/historical risk can be overdetected. The repaired explicit
   phrases do not create a clinical classifier or establish therapeutic efficacy.
5. **Outcome history is split.** New inline reports are saved in chat but do not
   populate the legacy Home/Journey garden. Journal-to-chat context selection and
   multiple discovery entry points still deserve a focused usability review.
6. **One unresolved JavaScript development-tool advisory.** `braces` 3.0.3 has a
   high-severity nested-pattern denial-of-service advisory, GHSA-vfj7-8cjw-p6xm,
   propagated through five lint-tool dependency records. The registry offers no
   patched version. This is one root advisory, not five distinct production bugs.
7. **Operational hardening remains.** Allowed retry/timeout combinations can exceed
   Vercel's 120-second function limit; an absolute end-to-end request budget needs
   separate work. The local PostgREST downloader lacks artifact/version verification;
   the official required x86-64 release asset exposes no checksum. Packaging tools,
   CI/container tags, network-fetched build fonts, Docker context exclusions and
   shell signal forwarding also limit reproducibility and need targeted follow-up.

No load/soak test, independent penetration test, fresh paid model/search run,
live Render deployment, or clinical evaluation is claimed. Dependency scans cover
published package advisories, not unknown vulnerabilities or the host OS.

## Evaluation handoff

You can review the historical side-by-side responses now, keeping the visible
version caveat. For hands-on evaluation of the **upgraded** flow, first complete
the remaining evidence/release work and deploy the verified candidate to Preview.
Do not treat testing the current older Preview as testing the local fixes.

Once that candidate is available, start with: quiet two-minute activity → change
to one minute → start/pause/reload → report unchanged/not tried; consented search →
refine → save → start; Just talk/stop; selected journal then current-state correction;
lost-response retry; and fictional safety cases. The implementation guide has the
[full manual checklist](LUNA_GUIDED_ACTION_IMPLEMENTATION.md#easiest-manual-review).

Detailed reviewer evidence is retained under
`/workspace/journalpulse-planning/pre-evaluation-audit-2026-10-05/`:
`ai.md`, `resource-safety.md`, `backend.md`, `backend-persistence.md`, `frontend.md`,
`evaluation.md`, `operations.md`, and `python-dependency-repairs.md`.
The repository review bundle includes the diffs, hashes, selected logs, dependency
results and corrected historical comparison in `artifacts/pre-evaluation-audit/`.
