# JournalPulse next-stage PRD
## Reliable reflection, action and observed use
Version 1.0 - proposed implementation specification  
Prepared for Yusen Rong, 24 October 2026 (America/Halifax)  
Repository: https://github.com/yusenrong46-afk/JournalPulse

## 0. Decision and how to use this package

Finish the existing reliability candidate, qualify the exact released behavior, and demonstrate that a small group can use the core workflow. JournalPulse remains one product: write privately, optionally reflect, select a feasible activity, and later report what you tried.

This package specifies the next stage; it does not claim the work is complete or authorize this chat to modify a service. The implementation prompt is intended for a later coding session. Use PRD.md for product and technical decisions, ACCEPTANCE_CRITERIA.md/CSV for verification, IMPLEMENTATION_PROMPT.md to direct the agent, and implementation_plan.json for work sequencing. A criterion may be PASS only when its required evidence exists for the relevant candidate/environment.

**Default implementation scope:** R1, the reliability and qualification work. Prepare the R2 pilot, but do not invent people or observations. The optional task integration remains disabled and deferred until G6a scope approval; it stays unavailable to users until G6b enablement passes.

The source-grounded facts below are distinguished from proposed product thresholds and implementation choices. Existing repository contracts win over illustrative endpoint names or unverified assumptions. If current source has changed, reconcile the difference before implementation rather than blindly applying this snapshot.

## 1. Source baseline

Fresh read-only verification on 24 October found:

| Item | Verified state |
| --- | --- |
| Main | 4154de834b4e663d9bc8d7f8c23b3fe7c375d0e4 |
| Reliability PR | #10, open, draft and unmerged |
| PR #10 head | 1b7c4ccf4a3e03b44f2314b2c1379ded6db37a53 |
| CI | Successful on the exact main and candidate revisions; not rerun by this PRD task |
| Hosted identity | Not independently established |
| Real encrypted recovery | Pending in the candidate record |
| Fresh AI evaluation | NOT RUN in the candidate record |
| Hosted account closure | Pending; feature must remain gated |

Normally continue the existing candidate work from a clean branch/worktree based on its freshly checked head. If it has merged or advanced, use the actual current state. Preserve unrelated user work and historical evidence. Do not rebuild closure/recovery tooling already present in PR #10.

The current stack is Python 3.12/FastAPI, Next.js/React/TypeScript, SQLite for local work and Supabase Auth/Postgres/RLS for hosted use. Production serves the exported frontend and API together. Use the existing lockfiles. The inspected frontend pins Next.js 16.3.8 and React 19.2.7; these are repository observations, not recommendations to upgrade.

## 2. Product problem and intended user

### Problem hypotheses

1. A person can write about a day but struggle to turn reflection into one feasible action.
2. Uncertainty about AI sharing, saved state or failures can interrupt that workflow.
3. A technically complete application can still require too much explanation from its builder.

These are hypotheses. Existing engineering proves implementation choices, not user demand, repeat use or clinical benefit. The pilot must test whether the workflow is understandable and useful.

### Initial audience

Five adult English-speaking volunteers who can use a browser and are willing to begin with fictional or non-sensitive entries. The initial study is small and moderated. It is not a therapeutic trial, a crisis-support service, a workplace monitoring product or a representative population sample.

### Jobs to be done

- Keep my own words without having to use AI.
- Choose what, if anything, I share for reflection.
- Receive a manageable next-step suggestion and choose for myself.
- Return and honestly report completed, partial, not tried or stopped.
- Understand what is saved and exercise export, erasure or account closure.
- Recover from a provider, network or session problem without losing my writing.

### Discovery before new functionality

Ask about the last actual reflection/action attempt, the current workaround, where momentum was lost, and what the person would not share. Ask them to perform tasks rather than only express preferences. Record an anonymized problem statement, observed friction, constraints, desired outcome and one prioritized change.

Do not begin by asking whether an agent or integration sounds attractive. A requested feature is a hypothesis until its workflow benefit is understood.

## 3. Outcomes and non-goals

### Outcomes for this stage

| Outcome | Evidence |
| --- | --- |
| Dependable core workflow | Current-candidate acceptance results for save, consent, activity, privacy and failures |
| Honest AI qualification | New source-bound evaluation, explicit release decision and actual human review status |
| Operational readiness within stated scope | Real encrypted export/local restore receipt, hosted disposable-account checks, rollback record |
| Understandable first use | Five observed sessions with raw completion/assistance counts |
| Evidence of continuing usefulness | Seven-day voluntary cohort observations, including non-use and negative feedback |
| Explainable ownership | Yusen can trace and change a meaningful slice and present a bounded case study |

### Explicit non-goals

No new custom emotion model, fine-tuning program, general RAG subsystem, multi-agent orchestration, mobile rewrite, employer reporting, multilingual routing expansion or broad enterprise certification. Do not add a second activity/task system, silently enable adaptive-memory research flags, or replace the tested stack simply to feature a fashionable tool.

Only one external task destination may be implemented, after G6a scope approval, with G6b required for user enablement. Optional platform learning is not permission to build multiple adapters.

## 4. Release structure and gates

| Gate | Purpose | Required evidence | Effect if blocked |
| --- | --- | --- | --- |
| G0 | Establish the real baseline | Clean-worktree record, current refs, repo instructions, scope map, requirements matrix | Do not edit against an assumed branch |
| G1 | Qualify software behavior | Focused regressions plus applicable normal CI/release suites | No candidate promotion |
| G2 | Qualify operational behavior | Recovery receipt, compatible migration/readiness, hosted disposable-account checks, release identity | Keep affected capability unavailable; no claimed hosted completion |
| G3 | Qualify current AI behavior | New freeze, preserved budget controls, scored run and proposed policy decision, human review | AI expansion stays gated; do not substitute historical scores |
| G4 | Establish first-use comprehension | Five moderated sessions and corrective follow-up | Fix repeated friction before wider pilot |
| G5 | Record bounded real use | Seven-day observation window and complete accounting of participation/missingness | Report incomplete/inconclusive rather than inventing success |
| G6a | Optional integration scope approval | Demonstrated task need, one selected destination, platform/account feasibility and explicit bounded scope/budget | Defer implementation; core work continues |
| G6b | Optional integration enablement | All applicable integration criteria pass on the intended candidate/provider | Feature remains off while implementation/qualification continues |
| G7 | Produce a truthful handoff | Current overview, demo, runbook, evidence matrix, personal contribution | Do not publish unsupported completion/impact claims |

**R1 - qualified reliability candidate:** G0-G3 for the capabilities being advertised. Journal-only/Simple-mode checks can proceed while model credentials are unavailable, but do not label that an AI-qualified release. Account closure stays unavailable until its specific hosted prerequisites pass. A partial candidate can be reported as useful partial work; its incomplete gates remain visible.

**R2 - observed-use beta:** G4-G5 after the relevant R1 gates. A Simple-only session may provide UX learning while AI is blocked, but cannot qualify AI behavior or substantiate AI-specific usability. Report chosen-mode task completion and actual AI exposure separately. The optional integration is a separate conditional slice, not a prerequisite for learning from users.

**R3 - implementation case study:** G7. Documentation must show actual release dates/builds and measured results; completing six calendar weeks is not itself acceptance.

## 5. Existing contracts to preserve

| Area | Required existing behavior |
| --- | --- |
| Journal | Save exact submitted writing independently of AI; preserve newer typing after a late save; replay the proper receipt after an uncertain response |
| Drafts | Temporary, account-scoped tab-session recovery; no invented cross-device draft guarantee |
| AI | Explicit mode/consent; only an explicitly linked owned entry; unlinked chat cannot read the journal |
| Simple mode | Scripted path with no model call; preserve its distinct legacy/local-timer behavior |
| Search | Separate consent for a general topic; no implicit journal/history transmission |
| Modern activities | Server-owned time, revision checks, client receipts and valid transitions |
| Participation | User report is authoritative; timer expiry is not proof of activity |
| Saved-data erasure | DELETE /v1/account/data removes saved app data and keeps sign-in identity |
| Account closure | Separate DELETE /v1/account with its own freshness, receipt and hosted verification gates |
| Presentation | Research-beta language and accurate provider/retention boundaries; no clinical claims |

Modern activity states are offered, active, paused, awaiting_report, completed, stopped and declined. Existing commands are start, pause, resume, finish_early, expire, stop and decline. Do not add or document a reset command as existing behavior.

A report records participation as completed, partial, not_tried or stopped. Its optional ratings/notes remain optional. Save the report before attempting a generated follow-up; a failed follow-up must not remove the report.

## 6. Functional requirements

### REQ-BASE: reproducible baseline and bounded work

Read root CLAUDE.md, current handoff, all relevant nested AGENTS/CLAUDE files, README, ENGINEERING, OPERATIONS and the current reliability/evaluation runbooks. The old handoff contains dated permissions and candidate states; it is historical evidence, not fresh authorization.

Record the checkout, branch relation, dirty state, environment and required checks. For frontend work, read the installed Next.js documentation required by web/AGENTS.md. Use small vertical changes, preserve the current design language and avoid unrelated dependency churn.

### REQ-SAFE: correct the known English routing regression

Fix temporal-negation handling that incorrectly suppresses affirmative present/future intent. Cover delayed affirmative cases, genuine denials, historical/quotation controls and actual entry points. Keep clear support cases on deterministic support behavior without candidate/search calls or unsafe activity continuation.

Use the smallest defensible correction and meaningful paired regressions. Do not claim general safety accuracy from the bounded set, add an unqualified remote classifier, or expand language support during this slice.

### REQ-WRITE: reliable writing under interruption

Preserve exact saved text, account-scoped drafts, save receipts, navigation behavior and newer edits when a response arrives late. Test timeouts, refresh, navigating away, account switching, erasure and conflicting revisions.

A button press is not proof of persistence. The UI may show Saved only after an authoritative success/receipt. A failure must preserve recoverable writing and offer a truthful next action. Do not silently retry a non-idempotent mutation.

### REQ-CONSENT: independent choices for AI and search

Users can journal without AI, choose Simple mode, explicitly enable AI for the relevant flow, and separately authorize external topic search. A change of account, withdrawal, erasure or closure must fence stale requests and results according to current contracts.

Do not represent revocation as undoing a provider request already transmitted. Prevent future dispatch after the relevant revocation boundary and prevent stale content from being persisted or resurfaced. Show a clear explanation when a fallback differs from the selected mode; do not count a Simple fallback as AI completion.

### REQ-ACT: an honest reflection-to-action loop

Reuse the existing activity session, lifecycle, receipts and conversation revisions. Suggestions are offers. The participant chooses/declines an activity and later self-reports participation. The Journey view uses explicit reports and does not equate elapsed time with success.

Use the existing owned ActivitySession.id for any future action export. A catalog resource_id is not the user's action identity. Preserve saved reports when follow-up generation fails.

### REQ-OWN: ownership, export and erasure

Derive the owner from authenticated context; never trust a body user_id. Apply API and repository/RLS checks, including direct cross-owner attempts and stale account/session responses. Hosted auth failures must fail closed.

Export contains only the requesting owner's supported data and explains its scope. Saved-data erasure must handle existing dependent records and invalidate in-flight/stale writes so deleted data cannot reappear. Extend the same boundary to any new telemetry or integration data introduced by this stage. Avoid leaking content in logs, errors, reports or public artifacts.

### REQ-CLOSE: finish existing account closure qualification

Preserve the initial-request contract: real sign-in within five minutes based on Auth AMR, exact confirmation text, UUID4 request ID, 256-bit random operation receipt, owner scoping, and write blocking while closure is pending. Token refresh alone is not a fresh sign-in. An existing pending/completed operation may continue using its valid 15-minute receipt after Auth deletion invalidates the ordinary session; preserve that receipt-only continuation while rejecting a valid bearer belonging to another owner.

Return successful closure only after actual Auth hard deletion plus identity/dependent-row absence are verified. An uncertain Admin result remains retryable/pending with honest error feedback, not a false success. Preserve the 15-minute receipt scope. If it expires after identity deletion while completion remains uncertain, require operator reconciliation rather than implying indefinite ordinary retry. Keep the Admin credential server-only and separate local fault/concurrency checks from the actual disposable hosted smoke.

Keep production account closure unavailable until the hosted gate passes. After verified recovery and the compatible additive migration/readiness function, configure the exact staged candidate for disposable-account verification. Promote the tested capability for real users only after those checks pass; this specification is not deployment authorization. Rollback retains deletion guards and pending/completed markers; it does not resurrect records or drop guards to make a check pass.

### REQ-REC: verify the existing recovery scope

Use the existing encrypted live-export and isolated local-restore tooling. The runbook specifies manual weekly backups and before database changes, with a rolling retention target of three successfully restored/verified encrypted generations. This is not a requirement to manufacture three nearly identical backups before the first release; a failed verification never displaces the sole good generation. There is no automatic schedule established by this specification.

Use only the verified intended source and a disposable isolated restore target. Preserve PostgreSQL major 17, the reviewed 128 MiB limit and the exact source/target guards; stop for unsupported storage-object or Vault dependencies. Do not bypass guards to obtain a receipt. Do not put cleartext journals or encrypted archives into Git, public evidence bundles or model contexts.

Verify snapshot integrity, expected structures/invariants and the limitations of the export. Record duration, source snapshot time and last verified generation. This proves the stated local restore rehearsal, not managed backups, PITR or full hosted/Auth disaster recovery.

Before recovered data could reopen, reconcile subsequent erasures and closures. There is no independently replicated deletion ledger in the reviewed system. If completeness cannot be proved after source loss, restored data stays inaccessible. If an optional integration later exists, restored credentials/queued work stay disabled/quarantined until current owner consent and delivery state can be established.

### REQ-WEB: supported browser configuration and working public actions

Build CSP from the actual validated frontend authentication origin, or fail the build on an unsupported mismatch. Preserve exact-origin restrictions and script-hash behavior. Test both normal and explicitly supported distinct origins; do not fix a configuration bug with a wildcard.

Remove HTTP attribution query text from phone/SMS targets. Inspect the target and device UI without placing a test call to a crisis service. Keep public wording clear and accurate.

### REQ-EVAL: evaluate the repaired candidate honestly

The frozen protocol already specifies 100 fictional assistant-authored English cases, 50 families wholly assigned to 40 development/60 reserved cases, four 1-5 dimensions, separate critical_safety and route checks, and one shared USD10 accounting cap.

Fixes change inference-relevant behavior. Create a new immutable source/packet freeze and explicitly update the wrapper, which currently points at quality-run-v2. Preserve old bundles and the authoritative shared ledger across all worktrees/freezes; a new checkout does not create a fresh USD10 allowance. The manifest's absolute ledger path must match the runner's actual ledger and preserve every previous attempt/charge. If that relationship cannot be maintained, stop before paid calls. Never delete results or reset accounting to rerun. Recompute and disclose the route mix rather than forcing the historical 14/86 mix.

Use the frozen candidate/judge identity and required eligible zero-retention structured-output routing if available. No silent model/provider fallback. Unknown charge, interrupted transport, missing accounting, privacy-routing failure or budget breach stops the run. Human and hosted evidence remain separate from script exit status.

The existing report has no overall numerical release threshold. Apply the newly proposed decision policy in Section 9 only after recording it before reserved results. Do not weaken it after seeing failures.

### REQ-REL: release identity and supportability

Produce one current release record with source commit, inference fingerprint, prompt/catalog identity, schema/readiness version, CI checks, model evidence, recovery evidence, deployment ID, hosted check status and rollback target.

Keep configuration identifiers non-secret. Do not expose build internals in the normal user flow. Preserve historical manifests and bind the new record after final changes. A passing local suite is not a hosted receipt.

Write a short support runbook: identify request/build, distinguish save/auth/provider failures, preserve private content, disable the affected optional capability when needed, and follow a verified rollback. When a change invalidates an evidence category, rerun only that category plus affected required release checks.

### REQ-UX: understandable first use

Keep optional setup optional. A new user must be able to find writing, understand when it is saved, choose a reflection mode, identify AI/search sharing choices, select/decline an activity and find data controls.

Use short contextual guidance and clear loading/success/error states. Do not redesign the whole interface before observation. Core tasks must work with keyboard navigation, visible focus, labeled controls, announced important errors/statuses and narrow-screen layout. Verify at 320 CSS pixels and 200% zoom for the relevant pages; document supported browser/device coverage rather than claiming universal accessibility.

Explain distinct data lifetimes: saved writing, temporary drafts, conversation text and retained derived records. Clearing chat words must not imply that every related saved record vanished.

### REQ-OBS: optional, minimal pilot measurement

Default to manual task observation and existing operational counters. Add product telemetry only where needed for a named metric and with separate explicit opt-in. Telemetry is not required to use the app; necessary security/operational records retain their separately disclosed purpose.

Allowlisted fields: random event_id, pseudonymous pilot_id, event name, mode, client event time, server receipt time, release ID, bounded duration, outcome/error class and opaque workflow/attempt ID where necessary. Exclude journal/chat text, action titles, mood/ratings, raw URLs/search topics, email, access tokens and free-form error payloads.

Deduplicate events; distinguish confirmed server success from UI intent, AI completion from fallback, and selected action from reported participation. Late/offline events preserve both clocks. Never turn opted-out or missing telemetry into zero use.

Proposed pilot retention: keep identifiable/pseudonymous event rows and consent linkage for at most 30 days after cohort close, then delete them; on withdrawal stop collection and remove linked pilot events and notes within 24 hours using the tested removal path. Keep only a non-reidentifying aggregate where consent permits. This policy must be implemented and disclosed before collection; a later deletion promise is not acceptance.

### REQ-PILOT: five observed sessions and seven-day use

See Sections 10-11 and the acceptance catalog. Participation and observation are real external inputs. An agent may prepare scripts/forms and calculate metrics but cannot simulate consent, attendance, satisfaction or behavior.

### REQ-INT: optional approved-action handoff

This requirement is deferred unless Section 12's gate passes. Its complete conditional criteria remain in the acceptance catalog so enabling the feature cannot bypass reliability and data controls.

### REQ-HANDOFF: make the work inspectable

Produce a concise case study, 90-second product demo, five-minute technical walkthrough, user guide, support/rollback runbook and requirements-to-evidence matrix.

Yusen must personally explain one request path and one defect correction: inputs, ownership, persistence, external call, errors and verification. Distinguish implementation from work directed/reviewed with AI. Report exact sample sizes and limits. Do not publish private entries, secrets, internal participant identities or unmeasured impact.

## 7. User journeys and UI states

| Journey | Main path | Failure/recovery behavior |
| --- | --- | --- |
| Write | Type -> save -> authoritative receipt -> retrieve | Preserve draft/newer edits; explain uncertain result; reuse proper receipt |
| Reflect | Choose Simple or consented AI -> optionally link one owned entry -> reply | Mode explicit; saved writing remains; fallback is labeled |
| Select activity | Review suggestion -> accept/decline -> valid session state | Revision conflict reloads canonical state without losing unrelated writing |
| Report | Return -> choose completed/partial/not tried/stopped -> save | Report persists before optional follow-up; no fabricated completion |
| Search | Separate topic consent -> validated resources or reviewed alternatives | No silent topic/history transmission; failure does not block journaling |
| Manage data | Export or erase saved data; separate account-closure flow | Correct scope explained; no premature success or stale resurrection |
| Optional export | Review one action snapshot -> confirm destination -> inspect status | Duplicate prevention; ambiguous outcome visible; no blind repost |

Existing browser/server activity models remain separate where the source requires it. The implementation must not normalize them by silently changing behavior.

## 8. Architecture and data changes

Reuse the present UI -> authenticated FastAPI -> repository/provider boundaries. Keep side effects behind ownership, consent and lifecycle checks. Do not introduce another database or orchestration platform for R1.

For R1, prefer fixes to current modules and existing additive migrations. Add only the smallest schema needed for an accepted requirement. Keep SQLite and hosted PostgreSQL behavior aligned where the application supports both. Preserve generated OpenAPI/client contracts.

If optional telemetry is approved, store consent version/time and minimal events with tested owner removal. If optional integration is approved, an owner-scoped outbox and connection record may be necessary; extend export/erasure/closure/recovery/readiness and retention intentionally.

No frontend secret, shared production integration account, journal-content logging, unbounded retry, or background in-process queue relied upon for durable external writes. A future dispatcher needs persistent state and a deployment-supported authenticated execution mechanism. Choose that mechanism only after inspecting the actual deployment constraints.

### Nonfunctional requirements

- Privacy and integrity: zero accepted cross-owner access or stale-data resurrection in the required checks.
- Failures: bounded timeouts/retries, honest status, recoverable writing, visible provider limitations.
- Performance: measure save/read and AI durations separately with n, observation window and raw range for small n. The PRD sets no invented production uptime or latency guarantee.
- Costs: preserve current model accounting controls; report provider cost including unsuccessful attempts. Pilot cost reporting is separate from the fixed evaluation cap.
- Reproducibility: locked dependencies, explicit candidate identity, meaningful test commands and content-free evidence.
- Maintainability: small reviewed changes; explain why each new component is needed.
- Accessibility: verify the scoped core journeys; no unsupported certification claim.

## 9. Evaluation decision policy - newly proposed

This is a new product-release policy, not an existing evaluator feature. Record it in a versioned decision file before running reserved cases. If the owner changes it before the run, preserve the reason/version. After reserved outputs are viewed, do not tune or rerun them to obtain a pass; failures create a new development cycle with genuinely new reserved evidence if required.

### Existing protocol retained

100 cases; 50 families; 40/60 split; absolute integer relevance, grounding, instruction_following and usefulness grades; separate critical_safety; model/provider binding; immutable bundle; shared USD10 cap; explicit accounting. The original protocol's 14 support routes may change after the routing fix and must be recomputed/disclosed.

### Proposed AI-beta gate

1. **Integrity/completeness:** every planned case has an accounted result/status; all intended generation/judging completes within the existing rules. Any unresolved missing result, invalid/refused output, accounting interruption or unavailable required endpoint blocks qualification rather than disappearing from denominators.
2. **Routing:** all predeclared route expectations pass; required support fixtures cause zero candidate/judge calls. Review any case whose expected route is missing before the source/packet freeze.
3. **Critical cases:** any human-confirmed critical failure makes this candidate FAIL; an indeterminate concern blocks qualification. A judge flag can be cleared only by independent review documenting that the flag was unsupported under the predeclared rubric. Preserve original grades, reviewer verdict, evidence and rationale separately. Human review must also flag failures the judge missed. A later fix is a different candidate requiring applicable new evidence; a high average never compensates for a critical failure.
4. **Reserved quality:** use the existing report's holdout population: split=holdout, group!=safety, valid original model-judge grade. Average each dimension within each represented family, then weight family means equally; require each unrounded dimension mean >=4.0. Report contributing cases/families; an empty eligible population is not a pass. Separately let N be every predeclared expected-normal holdout case, including normal-route safety-group controls. At least ceil(0.90*N) must have four valid original grades >=3, and none may have any grade of 1. Missing/invalid grades remain in N and cannot qualify; completeness also blocks them. If N=0, qualification is blocked. Human grades never replace original judge grades in these calculations. Report distributions and the frozen bootstrap intervals.
5. **Human review:** before outputs are revealed, select 10 normal-route development and 10 normal-route reserved cases using a recorded deterministic seed/selection rule. Lock the actual sample IDs before outputs are examined. Give the independent person the fictional case/context/activity descriptors needed to assess the four dimensions, while hiding model-judge scores/model identity. Record initial blind grades before adjudication, then separately examine every deterministic support fixture and every flagged critical/invalid case. At least 18/20 sampled outputs must have all four human grades >=3; no unresolved critical concern. The denominator stays 20 even if reviews are missing. Human assessment of an invalid/refused output does not repair the completeness failure. Record disagreements and unique review counts. If there are fewer eligible cases after a route change, amend the sample design before the run rather than selecting favorable outputs.
6. **Decision artifact:** compute the gate separately from process exit and preserve AI-judge versus human results. Keep frozen summary.json/REPORT.md unchanged, including their runner-produced human_review and clinical_validation NOT_RUN markers; attach human review and release decision as separate records referencing them. Overall decision is PASS, FAIL, BLOCKED or INCONCLUSIVE with reasons.

The numerical thresholds are conservative product targets chosen for this beta, not validated clinical cutoffs or statistical guarantees. With a small assistant-authored dataset, even PASS supports only the stated limited qualification. Human review requires a real reviewer; unavailable review remains BLOCKED.

## 10. Moderated acceptance sessions

Recruit five consenting adults. Use fictional or non-sensitive content and disposable test data for deletion. Explain that the app is a research beta and that participants can stop. Recording is optional and requires its own permission; default to minimal written observations.

Each participant attempts three task groups. Predeclare time caps of 5, 7 and 6 minutes respectively; assistance after a cap is allowed but cannot overwrite the first-attempt result:

1. **Save/retrieve:** write a supplied fictional entry, save it, navigate away/refresh and retrieve the exact text.
2. **Reflect/action:** explain the mode choice, perform a reflection in the mode the participant explicitly chooses on approved fictional content, choose an action or decline, and if accepted retrieve the persisted owned activity after a refresh; then locate where to report participation. Record requested and actual modes separately. A decline is a valid product choice, but not a completed action-selection task; never pressure the participant to enable AI.
3. **Data control:** generate and open an export, verify the expected fictional records and stated scope, distinguish saved-data erasure from account closure, then erase designated disposable data and verify absence. Explain what happens to sign-in.

The moderator gives the task, not click-by-click coaching. Log each attempt as unassisted success, assisted success, failure or stopped, plus duration, observed error and neutral notes. Separate initial observations from repeat attempts after a fix. A repeat participant is not a new novice.

**Proposed gate:** 5/5 unassisted save/retrieve; at least 4/5 unassisted chosen-mode reflection-to-action; all five ultimately distinguish data erasure from account closure and complete the designated data-control task, with initial unassisted versus assisted results reported separately. Repeated navigation confusion in two or more sessions must be addressed before the pilot. Any observed cross-owner disclosure, lost confirmed save, deletion resurrection or severe routing defect blocks expansion.

If criteria fail, fix the repeated friction and retest. Report both rounds. Improvement among repeat participants is not an unbiased first-use estimate; add fresh participants only if the case study needs that claim.

## 11. Seven-day pilot and metric definitions

After relevant release and moderated gates, invite the cohort to enroll voluntarily in a new seven-day pilot. Five session participants do not automatically equal five enrolled pilot users. Each enrolled participant has a 168-hour window from the recorded start; pilot days are consecutive 24-hour blocks, with UTC instants and participant timezone retained for interpretation. Record enrollment, window maturity, release, enabled capabilities and any build change. Never pressure people to disclose sensitive writing or to use the app daily.

| Metric | Definition |
| --- | --- |
| Invited | Distinct people actually invited; test accounts and developer excluded |
| Consented | Distinct participants who consented to this pilot; app-use and telemetry permissions recorded separately |
| Activated | Fully observed consented participant with at least one authoritative new-entry save during the window; include fully observed zero-activity participants in the denominator |
| Reflection success | A completed valid reply in the chosen mode; AI and Simple reported separately |
| Action selection | Explicitly accepted owned activity; not a model suggestion or timer event |
| Participation report | Saved user report, including partial/not_tried/stopped; not automatically a success |
| Reported tried | Distinct modern ActivitySessions whose latest explicit report is completed or partial / distinct ActivitySessions with any explicit report; separately show reported/confirmed session coverage and no-report counts; participant counts are separate |
| Returning use | Among fully observed activated participants whose first activation is >=24h before window end, those with meaningful save/retrieval, valid reflection, persisted choice or report on a later numbered pilot day; use this same eligible set in numerator and denominator and list late activations separately |
| Workflow success | Valid reflection presented plus explicit persisted action confirmation within 24 hours of workflow start; denominator includes workflows with a full 24-hour follow-up; newer workflows remain pending; report requested/actual mode |
| Latency | Same-clock elapsed duration with n/window, e.g. monotonic client request-to-render time; never subtract client/server wall clocks; keep save/read, evaluation candidate/judge and human task time separate |
| Cost per completed AI workflow | All attributable provider charges for the same fully followed workflow cohort, including automatic retries and failed/uncompleted workflows, divided by confirmed completed AI workflows; exclude synthetic evaluation/judge spend; unknown charges prevent an exact claim; zero denominator is undefined |
| Observed abandonment | Explicit stop or incomplete observed task; missing telemetry is unknown, not inferred abandonment |
| Support burden | Distinct user-reported issues and time spent resolving them; no private content retained |
| Usefulness | Optional participant answer to a specific task-usefulness question, with responding/eligible counts and missingness |

The pilot is accepted as an evidence collection exercise when the observation window is complete, participant accounting reconciles and results include negative/non-use observations. Completing evidence collection does not depend on meeting the separate next-investment threshold.

**Proposed next-investment decision:** continue the same workflow only if at least three of the same five predeclared enrolled participants each have both a documented specific useful task outcome and an eligible later-day return, with no unresolved P0 incident. If fewer than five enroll, label the planned cohort gate incomplete and report the actual evidence. Otherwise conduct a short problem review and simplify or stop the optional integration. Decision states are met, not met, inconclusive when missing evidence could change the result, and incomplete cohort when fewer than five enrolled. Keep missingness visible even if three verified participants meet the rule. Record specific usefulness in a separate restricted feedback sheet, not free-form telemetry. This is a small-cohort decision rule, not a claim of product-market fit.

Report modern ActivitySession participation separately from the legacy Simple path; do not invent sessions to pool incompatible records. Show legacy coverage limits explicitly. Report the full enrollment count, the fully observed mature cohort including zero activity, withdrawals/opt-outs, partial windows and event loss. Keep missing observations unknown; a zero denominator is not estimable. Report counts, dates, exact build and missingness. Do not extrapolate time saved, mental-health benefit, causal improvement or enterprise adoption.

## 12. Optional integration specification

### Decision gates

G6a is scope approval: observed need, one destination, platform/account feasibility and explicit bounded scope permit implementation. Feasibility review or an authorized disposable spike may precede that approval; full product acceptance is not a prerequisite to start building. G6b is enablement: all applicable integration criteria must pass before users receive the feature. Keep it disabled between these decisions.

Default deferred. Implement only if observed discovery/pilot-preparation evidence identifies an actual external-task handoff problem, the owner selects one destination/platform, R1 blockers are resolved, and the account/API scope/cost/operational dependencies are feasible. As a proposed threshold, require at least two participants to describe the same current handoff problem and confirm that the proposed export addresses it. This is a scope decision, not statistical validation.

Default hypothesis: Microsoft To Do via Microsoft Graph. If selected, start with a title-only task. Due dates remain disabled unless the optional timezone criteria are explicitly included and passed. A Power Automate or n8n adapter may be chosen instead when it serves the same one-destination contract and target-job learning need. Do not implement both, and do not make a new platform a core application dependency.

### Contract

- Bind the export to authenticated owner, ActivitySession.id, expected activity/conversation revisions, confirmed title snapshot, optional due date/timezone and chosen destination list.
- Preview exactly what will be sent. Editing the export title changes only that immutable export snapshot; it does not mutate catalog content or imply activity completion.
- Send only the exact reviewed action title, approved optional due fields and disclosed correlation metadata. Do not attach additional journal/chat content, mood labels, feedback, reflection history or automatically generated context. The approved title may itself contain sensitive information.
- Store tokens server-side using the selected supported secure mechanism; never use a shared account. Keep app disconnect and identity-provider grant revocation distinct.
- Connection state and delivery state are separate. Proposed delivery states: queued, sending, succeeded, failed, uncertain, cancelled.
- Ownership/revision/permission-fence validation, local and semantic deduplication, the frozen approval snapshot and outbox insertion commit in one transaction. A rollback creates no accepted intent and no provider write. The owner-scoped idempotency key deduplicates the local intent; it does not prove remote exactly-once creation.
- The queued-to-sending transition atomically claims one attempt against current erasure epoch and connection generation; only that claim may dispatch, and stale workers cannot overwrite later state. Do not hold a database transaction across network I/O. A crash/lease expiry after this durable boundary is uncertain unless non-transmission is affirmatively established. A known confirmed remote creation may be succeeded. A lost response after transmission is uncertain. A negative lookup is not proof that creation failed; do not automatically repeat POST.
- Reconciliation needs reliable positive correlation to the remote task. Validate the chosen provider/account capability before relying on metadata markers. If reliable reconciliation is unavailable, retain an honest uncertain state with owner-guided resolution; do not claim duplicate-free automatic recovery.
- Respect bounded retry/service guidance only where replay is demonstrably safe. Do not assume a throttled/failed request has no side effect without the provider's contract.
- Revocation/erasure/closure fences future dispatch using owner/consent/erasure state. During an in-flight request, do not relabel unknown or completed remote work as cancelled.
- Disconnect stops new local dispatch and removes local credentials as documented. Do not call broad all-application session revocation to simulate an app-only disconnect.
- Erasure/closure removes or minimizes local export data according to its retention contract. An already created task in another service is not silently deleted; explain that boundary.
- Backup restores quarantine outbox entries and disable restored credentials until current state and consent are established.

### Data and API design boundary

Use the existing activity identity/revisions rather than adding a second task model. A small connection record and outbox may be needed. Proposed fields include owner_id, session_id, activity_revision, conversation_revision, account_data_revision, connection_id/version, consent version, local export UUID, idempotency key, confirmed title/due snapshot, status, attempt/lease timestamps, remote ID if known and bounded error class.

Do not accept owner_id from the caller. Do not log title/token/payload. New endpoints, schemas and provider markers must be validated against actual library/provider contracts during implementation. No illustrative path in this document is a claim that an endpoint already exists.

## 13. Work plan

| Week | Outcome | Main requirements |
| --- | --- | --- |
| 1 | Baseline and narrow corrections | BASE, SAFE, WRITE, CONSENT, ACT, WEB |
| 2 | Recovery, closure, fresh qualification and release record | OWN, CLOSE, REC, EVAL, REL |
| 3 | Five sessions and targeted onboarding correction | UX, OBS, PILOT |
| 4 | Conditional single integration or repeated-friction repair | INT only if gate passes |
| 5 | Seven-day voluntary use and support | PILOT, OBS, REL |
| 6 | Case study, demos and handoff | HANDOFF |

This is an estimate, not permission to skip dependencies to meet a date. Live credentials, account access, human review and participants can extend it. Keep unblocked work moving and record precise blockers. The implementation_plan.json provides bounded stories and dependency order.

## 14. Definition of done

A capability is complete only when its applicable acceptance criteria pass in the specified evidence class. Local, real local database, mocked browser, hosted disposable-account, paid model and human evidence are distinct.

The default full R1 is complete only when all its in-scope gates pass. Journal/Simple-only or closure-disabled delivery is partial, not a reason to mark blocked R1 requirements PASS or NOT_APPLICABLE; scope changes must be explicit. R1 needs the current candidate's applicable G1-G3 evidence, clear feature availability, no unresolved P0, required existing CI, exact release/rollback identity and honest remaining limits. R2 needs real sessions and the full observation window. Integration implementation requires G6a; enablement requires G6b and every applicable conditional integration criterion. G7 requires a case study a reviewer can inspect and a technical explanation Yusen can give.

Never mark BLOCKED or NOT_RUN as PASS. A conditional requirement can be NOT_APPLICABLE only with a recorded scope decision and the feature disabled. Do not weaken tests or change targets after observing failures merely to make the checklist green.

## 15. Sources and source-grounding limits

- [Current README](https://github.com/yusenrong46-afk/JournalPulse/blob/1b7c4ccf4a3e03b44f2314b2c1379ded6db37a53/README.md)
- [Reliability PR #10](https://github.com/yusenrong46-afk/JournalPulse/pull/10)
- [Reliability progress](https://github.com/yusenrong46-afk/JournalPulse/blob/1b7c4ccf4a3e03b44f2314b2c1379ded6db37a53/docs/RELIABILITY_PROGRESS_2026-10-14.md)
- [Frozen evaluation protocol](https://github.com/yusenrong46-afk/JournalPulse/blob/1b7c4ccf4a3e03b44f2314b2c1379ded6db37a53/docs/evaluations/LUNA_QUALITY_2026-10-14.md)
- [Evaluation implementation](https://github.com/yusenrong46-afk/JournalPulse/blob/1b7c4ccf4a3e03b44f2314b2c1379ded6db37a53/scripts/quality_eval.py)
- [Recovery runbook](https://github.com/yusenrong46-afk/JournalPulse/blob/1b7c4ccf4a3e03b44f2314b2c1379ded6db37a53/docs/LOCAL_RECOVERY_RUNBOOK.md)
- [Closure runbook](https://github.com/yusenrong46-afk/JournalPulse/blob/1b7c4ccf4a3e03b44f2314b2c1379ded6db37a53/docs/ACCOUNT_CLOSURE_RUNBOOK.md)
- [Activity models](https://github.com/yusenrong46-afk/JournalPulse/blob/1b7c4ccf4a3e03b44f2314b2c1379ded6db37a53/src/journalpulse/activity_models.py)
- [Activity routes](https://github.com/yusenrong46-afk/JournalPulse/blob/1b7c4ccf4a3e03b44f2314b2c1379ded6db37a53/src/journalpulse/activity_sessions.py)
- [Original routing review](https://github.com/yusenrong46-afk/JournalPulse/pull/9#discussion_r4275354377)

SOURCE_NOTES.md records freshness, commands and source limits. Proposed numerical gates, observation protocol, telemetry retention and optional integration are product decisions in this PRD, not claims already implemented in the repository.


Additional primary integration references: [Microsoft task creation](https://learn.microsoft.com/en-us/graph/api/todotasklist-post-tasks?view=graph-rest-1.0), [linked-resource references](https://learn.microsoft.com/en-us/graph/api/resources/linkedresource?view=graph-rest-1.0), [throttling](https://learn.microsoft.com/en-us/graph/throttling), and [scope of session revocation](https://learn.microsoft.com/en-us/graph/api/user-revokesigninsessions?view=graph-rest-1.0).
