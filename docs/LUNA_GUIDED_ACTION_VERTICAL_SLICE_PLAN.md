# Luna guided action: complete vertical-slice upgrade plan

Status: implemented locally; the original candidate's walking-search delivery failure is fixed, but the fresh formal release evaluation is incomplete under evaluator rate limits. Prepared October 5, 2026. See `LUNA_SEARCH_CONTRACT_FIX.md` for current results and `LUNA_GUIDED_ACTION_RELEASE.md` for the preserved original result. The release targets and budget below remain unchanged.
Requested execution model: GPT-6.1 Sol with Ultra reasoning.

## 1. Outcome and scope

Deliver one complete experience in the existing JournalPulse app:

**Talk or use a linked journal → understand the user's intent → Luna proposes an
appropriate activity → user accepts or negotiates → activity opens inside chat →
user reports what happened → Luna adapts or finishes.**

Luna takes initiative in proposing activities. The user controls participation,
correction, direction, and stopping. The default voice is warm, concise, natural,
and practical. More direct encouragement can follow a user's request; a dedicated
coaching persona or Tony Robbins imitation is not required for this slice.

The reference end-to-end case is a brief guided meditation with a countdown and
automatic return to an inline check-in. Other existing activity categories use a
common card/session mechanism: timers when appropriate and explicit completion
controls when duration cannot establish completion.

### Included

- A frozen evaluation dataset, current-Luna baseline, bounded teacher-model
  conversations, and an honest before/after benchmark.
- One research-informed runtime `SKILL.md`, versioned and actually loaded into
  Luna's instructions.
- Grounded reflection, proactive recommendation timing, natural-language
  negotiation, user correction, direct requests, and stopping.
- Owner-bound activity sessions that keep the original conversation open.
- A meditation panel with instructions, countdown, pause/resume, finish early,
  interruption recovery, and an automatically displayed check-in.
- Context-sensitive selection from existing resources; inline Brave discovery
  and refinement when another resource is needed and search consent allows it.
- Linked-journal context with visible provenance and past/present distinctions.
- Inline completion and outcome reporting, followed by a bounded Luna response.
- Privacy/lifecycle tests, reviewable patches, release evidence, and a standalone
  HTML comparison report for the user's final evaluation.

### Later slices

- Automatic journal lookup by date, broader journal retrieval, and long-term
  pattern analysis. This slice uses the existing explicitly linked entry.
- Custom NLP training/distillation in Colab, learned recommendation policies,
  reinforcement learning, or off-policy evaluation of LLM recommendations.
- Wearables, passive sensing, push-notification infrastructure, speech, generated
  meditation audio, and a complete visual rebrand.
- A new mandatory emotion questionnaire or a specialized player for every format.

This consolidates item 1 and selected parts of items 2–4 in `NEXT_UPGRADE.md` into
one user-facing loop. Item 5 remains later. Keep the legacy paths functional.

## 2. Research basis and its practical limits

Use the accompanying reviews:

- [Reflection evidence](research/REFLECTION_EVIDENCE_2026-10-05.md).
- [Chat–action evidence and proposed flow](research/CHAT_ACTION_EVIDENCE_2026-10-05.md).

Guided discovery suggests exploring relevant interpretations, not increasing
question frequency indiscriminately. Motivational interviewing supports space
for ambivalence. Autonomy support motivates correctable, rejectable suggestions.
Behavioral activation motivates meaningful activity alongside short-term relief.
JITAI theory motivates adapting timing/content to both need and receptivity.

Breathing and movement studies justify plausible activity candidates, not a
guaranteed state change or a validated resource selected from the web. The
400-participant breathing active-control trial and null proximal findings in a
recent adaptive-intervention study must remain in the evidence narrative.

Measure **fit**, **participation**, **reported state change**, and **progress
toward the user's goal** separately. Neither more conversation nor immediate
positive mood is the sole objective. This upgrade is an engineering/product
hypothesis, not demonstrated treatment efficacy.

## 3. Starting point: preserve the audited workspace

Use `/workspace/JournalPulse`, including its existing tracked and untracked
changes. Do not treat `git HEAD` or a fresh `main` checkout as the current product.
The foundation audit and deployed preview include substantial uncommitted work.

Before implementation, record branch/HEAD, status, source hashes, current prompts,
schema versions, deployment identity, and tested environment. Create a separate
pre-upgrade source snapshot/manifest outside the checkout, without secrets,
generated build directories, or real user data. Preserve the existing foundation
audit artifacts and current deployment as evidence and a rollback reference.

Existing architecture requiring attention:

- `src/journalpulse/intelligence.py`: current structured Luna calls and provider
  refusal/deadline/schema controls.
- `reflection_prompts.py`: current shared reflection instructions.
- `conversations.py`: preference/revision/idempotency/safety lifecycle; existing
  acceptance closes the conversation and may clear messages.
- `resources.py`: catalog filtering and first-match baseline; current recommendations
  are not genuinely personalized by confirmed emotion.
- `discovery.py`, `discovery_models.py`, `discovery_prompts.py`: consented Brave
  search, snippet selection, refinement, exclusions, URL checks, and provenance.
- `journals.py`, `journal_models.py`: single-entry ownership and reflection.
- `persistence.py` and Supabase RPCs: trusted writes, RLS, signing, deletion,
  retention, concurrency, and receipts.
- `web/app/talk/page.tsx`: current chat → feelings → goal → offer → saved flow.
- `web/components/action-timer.tsx`: existing interval countdown; its finish hook
  currently runs inside a state updater and is not a safe new event authority.
- `web/app/check-in/page.tsx`: existing separate outcome flow.
- `evaluation.py`, `scripts/evaluate_luna.py`, `assets/evaluation/luna-v1.json`:
  existing offline preparation and structural review, not a live semantic benchmark.

Read `web/AGENTS.md` and relevant bundled Next.js documentation before frontend
edits. The current runtime is Python 3.12, Next 16.3.8, and Node 22.

## 4. Benchmark first

### Dataset v2: 60 scenario families/cases

| Group | Count | Development | Held out |
| --- | ---: | ---: | ---: |
| General conversations, recommendation, negotiation, activity, outcome | 40 | 24 | 16 |
| Safety and adversarial/privacy situations | 12 | 8 | 4 |
| Linked-journal grounding and lifecycle | 8 | 4 | 4 |
| Total | 60 | 36 | 24 |

Related variants belong to the same split. Do not reuse held-out results to tune
the candidate; if they inform a revision, label that evaluation exploratory and
create a new independent holdout before the next confirmatory comparison.

Use a mixture of human-authored fictional scenarios, manually reviewed synthetic
variants, and licensed public examples/adaptations. EmpatheticDialogues and
DailyDialog are candidate sources for natural language and emotion/dialogue-act
labels. Verify exact license, provenance, allowed uses, and attribution before
including text. Their labels do not supply our recommendation/safety judgments.
If reuse rights are unclear, use original fictional material; do not stall the
benchmark. Public material may be in model training and is not the strongest
generalization test. Do not use real journals or private conversation transcripts.

Each case records ID/family/split, origin/license, goal and known facts,
constraints, scripted turns or bounded branch rules, preference, fictional
journal/tool fixtures, activity events, allowed capabilities, required behavior,
forbidden behavior, applicable rubric items, severity, and maximum turns.
Expected behavior must allow multiple good responses; avoid exact prose matching.

Coverage must include positive experience, uncertainty, mixed feelings, correction,
wording requests, listening preference, refusal, topic change, stopping, enough vs
insufficient context for an activity, unavailable formats, negotiation, failed
search, no candidates, resource injection, declined activities, interruption,
timer expiration without participation, unchanged/worse/uncertain outcomes,
past journal versus present state, deleted sources, and unsupported access claims.

Safety cases include current urgent risk, historical/quoted/fictional risk,
harmful activity requests, concerning physical symptoms, unsupported beliefs,
journal/search instruction injection, and cross-account/source-access attempts.
Some are deterministic system tests, not attempts to make the model bypass an
intentionally locked support route. Existing conservative safety routing stays
authoritative; do not silently redesign it to make model scores look better.

### Controlled and adaptive tracks

- **Controlled:** identical conversation histories, fictional sources, search
  snippets, timer events, model settings, and applicable contracts for each version.
- **Adaptive:** eight designated cases (a subset of the 60) use a teacher simulated
  user with a fixed situation/goal/constraints and a four-turn maximum. Freeze the
  simulator instructions and seeds where supported. The simulator may negotiate
  within those facts, not invent facts to rescue or defeat Luna. Because trajectories
  diverge, compare scenario-level outcomes and burden, not identical turn wording.

Keep simulated outcomes scripted; they cannot measure real emotional benefit.
Use repeated runs on a small declared subset to inspect stochastic variation.

### Baseline must precede candidate prompt changes

Freeze exact current prompt modules and assembled requests, then observe current
Luna on every applicable shared-behavior case. Capture actual production-pipeline
behavior where safety/preferences/tools alter the response, plus controlled
provider tests for language. Preserve exact inputs, outputs, refusals, failures,
latency, tokens, and run provenance. An offline prepared request is not an observed
model response. Current support/guided outputs are not GPT-Luna generations.

New-only features such as an inline timer event are `unsupported_baseline`, not
language-quality failures. Keep comparable denominators explicit. Do not count
new UI capabilities as proof of improved conversational intelligence.

Only after the baseline is complete should the development cases guide prompt
revision. If external access blocks it, retain the frozen source/request bundles,
finish independent preparation, and report the missing observation honestly.

### Teacher evaluator and scoring

Use GPT-6.1 Sol as teacher/simulator/judge where actual access is verified. Ultra
is the coding agent's reasoning setting; it does not automatically identify the
evaluation API model or prove the judge used Ultra. Record the actual model and
settings. Do not label ordinary outputs as GPT-6.1 judgments without evidence.
This is LLM-as-judge evaluation, not distillation or model training.

Hide versions and randomize A/B order during judging. Prevent code diffs, version
names, and desired winner from entering the judge input. Require structured scores,
evidence excerpts, uncertainty, preference/tie, and reasons. Score five dimensions
1–5 where applicable: naturalness, useful understanding, grounding, recommendation
fit, and user control. Inspect repeated paraphrasing/questioning burden separately.
Do not reward verbosity, automatic agreement, or similarity to the teacher's style.

Keep safety/privacy and software gates as explicit pass/fail/uncertain results
outside the quality average. A judge cannot establish database ownership, timer
correctness, clinical safety, or a person's actual emotional change. Teacher
model-family bias remains a limitation; the user performs final subjective review.

Predeclare provisional engineering release targets before running the candidate:

- All defined critical safety/privacy and lifecycle checks pass, with no unresolved
  high-severity regression. Every observed severe failure blocks release.
- Applicable held-out primary quality improves by at least 0.3 points on the 1–5
  scale. Use useful-understanding for reflection cases and fit for action cases.
- Candidate wins at least 60% of non-tied eligible held-out pairs; report ties,
  failures, unsupported and unobserved cases with their actual denominators.
- Naturalness does not decline by more than 0.2 points overall; no severe loss of
  correction, refusal, or stop compliance is allowed.
- New functionality passes its separate acceptance tests.

These thresholds are proposed product gates, not scientific constants. Report
paired differences, uncertainty and sample size; passing a small benchmark does
not prove broad reliability or clinical efficacy. If targets fail, report the
failure and revise only with development evidence, not selective deletion of cases.

### Bounded live-run budget

Proposed ceiling for the implementation run: 600 Luna provider attempts,
180 teacher/simulator/judge attempts, six live Brave calls, and USD 25 total
estimated model/search spend. All retries/repairs count. Fixed-fixture search is
not a live Brave call. These are upper bounds, not a request to consume them.

Preflight actual model access, retention/schema capability, advertised pricing,
required credits, and estimated input/output tokens. Tighten concurrency and case
turns to remain within the budget. If pricing cannot be verified, do not present
the dollar limit as enforceable; use the call/token limits and surface the gap
before spending. Cache by complete request hash and resume completed runs.
No automatic credit purchases, external key creation, or unbounded retry loops.
If a limit is reached, retain results and mark incomplete rather than fabricate a
pass. Any larger run requires an explicit budget change.

## 5. Runtime skill and conversation decisions

Store the canonical skill under a packaged source path, for example
`src/journalpulse/skills/guided_action/SKILL.md`, with version/name/description.
Include research references, conditional rules, examples, and limitations.
Load it explicitly from trusted package resources; ensure it is included in the
built wheel and Vercel artifact. A Markdown file left in excluded `docs/` has no
runtime effect. Record skill version/content hash with model-run provenance.

Trusted core instructions retain safety, consent, output contracts, and tool
limits. Skill instructions are static trusted content. User journal text, web
snippets, and user messages never become system instructions.

Replace conflicting normal-mode instructions coherently: the current prompt
requires a latest explicit activity request and puts resources in a separate flow.
Simply appending a proactive-action skill would leave contradictory instructions.
Version the assembled prompt and verify every runtime path uses the intended
composition. Keep the standalone journal-reflection no-action contract and explicit
Just talk restriction unless the user changes direction in an ongoing conversation.

At each turn Luna chooses an appropriate conversational move: answer/reflect,
clarify, propose an activity, negotiate/revise, discuss an outcome, or pause.
Use existing strict JSON with bounded, validated extensions; avoid an unrestricted
agent loop. The server validates all resource references and action transitions.
Limit orchestration to one bounded recommendation/search operation per turn.

Goals and emotions inferred by the model remain tentative. The user's correction
overrides them. Do not invent numerical emotion confidence, sensor readings, or
neuroscience mechanisms. Ask only an essential missing question. A listening or
stop request suppresses proposals. Positive experiences need no intervention.

Do not require feelings and goal screens when conversation already provides
enough information. Keep optional controls for clarification/correction and the
guided/no-AI mode. Provider failure remains a visible retryable failure, not a
canned response represented as a successful model turn.

## 6. Activity sessions and data contracts

**Do not reuse the current `/accept` close-and-clear operation for an activity
inside the ongoing conversation.** Introduce a separate additive activity-session
contract; preserve existing acceptance and old clients.

Suggested session states: `offered`, `active`, `paused`, `awaiting_report`,
`completed`, `stopped`, `declined`. A terminal session does not automatically close
the conversation. Allow one nonterminal session per owner/conversation at a time.
Define duplicate-offer and superseded-offer behavior explicitly.

Capture minimal owner-bound information: conversation/session ID, selected and
recommended resource references, expected conversation/session revisions,
goal/constraints needed for the activity, trusted recommendation provenance,
duration and authoritative timestamps/remaining duration, bounded event receipts,
participant report, outcome, and follow-up status. Do not duplicate raw journal
passages into activity rows or indefinitely store unretained conversation excerpts.

Support start/pause/resume/finish-early/expiry/report commands with client request
IDs. The server enforces ownership, state transitions, revision checks, signing,
and idempotency. SQLite and Supabase implement the same behavior. Treat these as
proposed contracts; finalize endpoint names and schemas before parallel coding.

An expiry event requests an inline check-in; it never asserts participation,
helpfulness, or a mood change. Participant reports distinguish completed,
partially tried, not tried, and stopped. Outcome fields distinguish activity fit,
reported state change (toward target/same/away/unsure), goal progress, optional
same-scale before/after rating, helpfulness, effort, and bounded free text.

Reuse existing outcome/export concepts where their semantics fit; extend them
additively where they cannot represent partial/unknown participation. Never force
unknown or partially tried into a misleading existing `completed=true` record.
The existing standalone check-in must continue reading old records correctly.

Keep the current 20-user-message limit for ordinary chat. Session controls and
outcome submission must still work when that limit is reached. Permit at most
one separately budgeted final outcome response for an already-started session at
the limit; then offer a clear conversation ending instead of another activity.
Automatic events cannot create an unlimited generation bypass. Test the boundary,
including refusal, interruption, retry, and report storage when generation is
unavailable. Existing idle-retention expiry remains authoritative; a timer cannot
resurrect an expired or closed conversation.

Expose trusted resource/prompt/provider/selection provenance. LLM recommendation
and user negotiation are not randomized research-policy decisions: mark them
ineligible for OPE and do not invent selection propensities/confidence. Keep
research policy/memory flags off.

## 7. Timer, automatic check-in, and feedback behavior

Compute time from an elapsed/deadline model, not accumulated interval decrements.
Intervals repaint the display only. Specify authoritative server start/expiry,
pause duration, refresh recovery, wall-clock correction, and offline behavior.
Use injected clocks/fake time for lifecycle tests, not long blocking sleeps.

Start only when the user presses Start. The panel shows duration, clear instructions,
optional existing reviewed audio, countdown, pause/resume, and finish-early. Audio
does not autoplay and no countdown updates are announced every second to screen
readers. Use keyboard-accessible controls and a single gentle finish announcement.

On expiration, display one deterministic inline participation check immediately.
Commit its event/receipt through the session API; refresh/two tabs/retries must not
create duplicate questions or model calls. Do not call a paid model just to render
the initial completion check. A serverless function must not stay open for the
meditation duration. Background expiry is recognized on the next client sync;
there is no push notification promise while the browser is closed.

After the participant reports, invoke one bounded model follow-up using validated
outcome context. Make report storage and follow-up generation independently
retryable with receipts/revision checks: a model timeout cannot erase an outcome
or cause another outcome on retry. Show saved-report/pending-follow-up status.

Luna discusses what changed without assuming improvement. It can propose a revised
option when welcomed, return to the concern, or finish. Do not start another
activity automatically after every check-in. Closing/deleting a chat, changing
owner/preferences, source deletion, and safety support invalidate obsolete effects.

## 8. Resource recommendation and inline Brave

Luna selects only from server-approved candidates. Use confirmed constraints,
available time/format/location, stated goal, recent rejection, and report feedback.
Choose one primary recommendation; a short reason should name the relevant fit,
not claim a cure or validate a guessed psychological pattern.

Use the existing reviewed catalog first when appropriate. Under existing search
consent, Luna may propose a bounded general activity query when candidates do not
fit or the user requests an alternative. Users may edit the topic; do not add a
mandatory approval dialog for every query once consent and intent are established.
Do not send private journal passages, raw conversation history, names, or identifiers
to Brave. Validate topic scope; model instructions alone are not a privacy boundary.

Reuse existing Brave selection/refinement, URL validation, exclusions, limits,
privacy routing, and provenance. Preserve the original goal and rejected links
when refining. Show that selection uses snippets, not a full-page clinical review.
Handle missing key, provider refusal, timeout, empty/unsuitable results, and stale
search replies. Offer a fitting catalog option or continue talking with clear
status instead of fabricating a retrieved resource.

Opening a link does not save it or prove completion. Save only on an explicit
Save action. Existing resource formats receive suitable completion controls:
timed meditation/movement, and explicit Done/Not yet for video, reading, or social
activities. External links retain a return-to-chat path; embedding depends on the
source's capabilities and restrictions.

## 9. Linked journals in the new loop

Preserve journal writing independent of AI consent and preserve its original text.
Standalone reflection remains possible without accepting an activity.

A linked-entry conversation uses only the explicitly selected owner's entry.
Display the entry/date that informed the response and allow continued chat without
that context. Explain past events without assuming the same state is current.
The user can correct the older account. No unsupported recurring-pattern or
all-journal-memory claims.

Re-read/validate source ownership and existence before model disclosure and before
committing a source-dependent reply. Deleted source text must not re-enter future
prompts, sessions, exports, or cached pending responses. Raw source data stays in
an untrusted message. Journal/search injection cannot invoke access to other users,
reconfigure the skill, or reveal credentials.

Do not interpret historical safety content as proof of current intent solely
because it is in a journal. Preserve the existing router's actual conservative
behavior and report limitations; clinically nuanced historical-risk routing is
not silently added in this slice. Immediate current-risk support overrides the
normal action loop.

## 10. Privacy, migrations, compatibility, and failure handling

Add new migrations after all applied files; never edit/replay existing migrations.
Use versioned RPCs/readiness where necessary. New tables require owner-scoped RLS,
server-signed writes for system facts, explicit export/deletion integration,
retention behavior, bounded receipts, and source/conversation cascades.

Plan compatibility with the shared existing Supabase project. Preview and old
production must both remain functional after additive migration. Test old/new
contracts locally; do not reset or use the hosted database as a scratch database.
Stage migration SQL, impact, rollback limitations, and validation evidence before
any external execution. Destructive data changes or security-boundary changes
require explicit authorization. Use the existing preview/project authorization
within its scope; this plan is not production-deployment authorization.

Active sessions must not retain private text past current conversation retention.
Define whether an unfinished session marks activity for expiration, then include
it in the existing retention job/readiness diagnostics. Source deletion and account
data deletion invalidate pending generations. Browser keys/session caches remain
account scoped with revision checks and delayed-result rejection.

No secrets in chat, logs, fixtures, patches, or the HTML report. TLS verification,
provider zero-data-retention requirements, quotas, body bounds, and native-refusal
handling remain enabled. Evaluate with fictional disposable accounts; clean up
only records created by the run. Existing real user data is never benchmark input.

## 11. Delivery sequence and evidence at each step

These are internal work units of one vertical slice, not independent feature
releases. Integrate and evaluate the whole loop before updating the stable preview.

| Unit | Deliverable | Completion evidence |
| --- | --- | --- |
| 0 | Pre-upgrade snapshot, environment/model preflight, frozen scope/contracts | Source/prompt hashes, status, capabilities and budget |
| 1 | Benchmark rubric, dataset v2, development/holdout manifests | Valid schemas, family split, source/license review, failure criteria |
| 2 | Current-Luna controlled and adaptive baseline | Actual transcripts, provenance, unsupported/unobserved markers |
| 3 | Canonical packaged skill and bounded model decision contract | Skill hash/version; load verified in build; development comparisons |
| 4 | Additive activity session persistence/RPC/lifecycle | Ownership, transitions, receipts, races, retention/export/deletion checks |
| 5 | Context-sensitive catalog and inline Brave negotiation | Candidate validation, query privacy, exclusions, refusal/failure tests |
| 6 | In-chat meditation/session UI and outcome follow-up | Timer/refresh/two-tabs/interruption/mobile/keyboard real-stack tests |
| 7 | Linked-journal integration and end-to-end hardening | Grounding and source deletion/ownership/privacy regression evidence |
| 8 | Frozen-candidate benchmark, blind judging, HTML report | Paired results, gates, limitations, source/model manifests |
| 9 | Candidate preview release, smoke checks, stable preview update | Ready checks, disposable scenario, compatible rollback target |

Freeze cross-layer DTOs and resource IDs before parallel implementation. Build
small modules around session lifecycle, prompt loading, resource selection, and
report rendering rather than enlarging the current talk component indefinitely.
Preserve existing API compatibility. Regenerate OpenAPI/types through existing
commands. Comment ownership, state transitions, timing, retention, uncertainty,
and idempotency decisions where they are enforced; avoid comments that merely
repeat obvious statements.

At each work unit save a narrow patch against the preceding snapshot, a short
before/after behavior note, files changed, meaningful validation, and remaining
issues. Also produce a combined upgrade patch against the full pre-upgrade
workspace. Do not reset inherited changes or pretend a diff against `HEAD` is the
upgrade-only diff. Do not commit/push/merge merely to create these artifacts.

## 12. Tests and completion criteria

Required checks: Ruff/mypy; backend tests and relevant branch coverage; OpenAPI
consistency; catalog validation; frontend lint/typecheck/unit/build; scratch
PostgreSQL migration/RLS/signing/concurrency tests; mobile/desktop browser tests;
real-stack integration; bounded live model/search evidence. Run relevant targeted
checks while changing a component, then the existing required suites for final
integration. Do not expand into unrelated audits or repeatedly rerun passing suites
without a change or unresolved concern. Coverage is execution evidence, not a
conversation-quality or safety score.

High-value regressions include old acceptance still closing atomically, new start
keeping chat open, duplicate start/expiry/report/follow-up receipts, two tabs,
lost responses, paused timer recovery, duration changes, background throttling,
owner A→B→A, deleted chat/source, stale offer/search reply, support/preference
change during generation, missing consent, malicious snippets, export/deletion
of new rows, and old-client compatibility after migration.

Release is complete when the frozen benchmark and hard checks meet the declared
gates, artifacts are inspectable, and the candidate/stable preview smoke checks
pass. If a gate fails, report it as failed; do not call the slice done by averaging
it away. The user's later review is the subjective usefulness check, not a reason
to omit the automated evaluation or mislabel teacher scores as human scores.

## 13. Final HTML report

Generate `artifacts/luna-guided-action/comparison.html` with its data manifest.
It must be a standalone offline document with embedded CSS/JS and fictional data,
no remote dependencies or credential/authentication requirements. Escape all user,
journal, search, and model text; hostile strings must render as text, not script.

Include per-case side-by-side complete transcripts, scripted user facts/events,
expected/forbidden behavior, blind A/B toggle, score/evidence excerpts, differences,
teacher uncertainty, failures and unsupported/unobserved states. Show aggregate
paired results and actual denominators; filter by group, severity, outcome, and
split. Provide model/settings, prompt/skill/code hashes, dates, call/token/spend
counts, controlled-vs-live provenance, and known limitations.

Allow the user to record Agree/Disagree/Unsure and an optional note locally, with
downloadable review JSON. Do not claim local annotations were uploaded or used to
train anything. Include screenshots or short captured demonstrations for the timer
and journal UI where useful, alongside the language benchmark. Clearly distinguish
teacher-model judgments from user review, deterministic tests, and actual live
retrieval. Keep the report out of the product's normal user flow.

## 14. Ultra execution handoff

Use the prompt below with GPT-6.1 Sol Ultra in this same workspace:

```text
Implement the full Luna guided-action vertical slice described in
docs/LUNA_GUIDED_ACTION_VERTICAL_SLICE_PLAN.md. Read that plan, its research
references, applicable AGENTS.md files, and the current source before editing.

Use the existing audited workspace and preserve all inherited changes. Work
through the plan's internal units as one integrated user experience. Snapshot
the full current workspace and create upgrade-only patches and before/after
notes for each unit. Comment non-obvious code so I can understand and troubleshoot
the system. Avoid unnecessary abstraction and unrelated feature work.

You may delegate explicitly scoped work to subagents. First freeze the benchmark,
capture current Luna's actual baseline, and agree cross-layer contracts. Then use
parallel agents for backend session/persistence, frontend activity UI, resource
search/journal boundaries, and independent test review. The primary agent owns
integration and contract changes. Give each agent distinct files; coordinate shared
files before editing. A benchmark custodian controls the held-out cases; do not
feed their results to prompt authors while tuning. No agent may deploy, migrate
the hosted database, change secrets, or reset shared data independently.

Use GPT-6.1 Sol teacher evaluation where access is verified, with the declared
bounded call/spend limits. Keep exact model/prompt provenance. Use controlled
fixtures for fair before/after comparisons and bounded adaptive conversations for
robustness. Only judge observed outputs; keep failures, unsupported and unobserved
cases explicit. Preserve safety, privacy, ownership, consent, refusal, retention,
quotas, and stale-result protections.

Build the complete chat→propose→negotiate→activity→report→adapt loop, including
the meditation countdown and automatic single check-in, inline consented Brave
alternatives, and the existing explicitly linked journal. Keep automatic journal
date lookup and model training outside this slice. A timer expiring cannot prove
the user participated or felt better. Never close/clear the ongoing chat merely
to start its activity.

Run meaningful component and final integration checks, the candidate benchmark,
and independent review. Produce the standalone before/after HTML report described
in the plan, reviewable patches, migration/release evidence, and a concise account
of changes and remaining limitations. Complete independent work autonomously;
ask me only for a genuinely necessary missing prerequisite or approval. Do not
purchase credits, expose secrets, fabricate evaluation, or deploy to production.

For the preview release, prepare the candidate and reviewable migration impact
first, use the existing preview and Supabase authorization within its scope, verify
old/new compatibility and candidate health, then update the existing stable preview
only after the gates pass. Do not reset/replay hosted migrations or change unrelated
production settings. If an external action needs additional approval, leave the
concrete candidate/artifacts ready and explain the precise blocked action.
```

Ultra is a reasoning configuration, not a guarantee of multitasking. Delegation
must be invoked explicitly, and coordinated parallel work follows the baseline
and contract prerequisites. The final product is one evaluated vertical slice.
