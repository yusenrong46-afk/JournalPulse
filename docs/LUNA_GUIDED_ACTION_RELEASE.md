# Guided-action candidate: release evidence

Latest status: the evaluated `.3` version was deployed for explicitly authorized
preview testing on October 5, 2026. See [preview release evidence](LUNA_PREVIEW_RELEASE_2026-10-05.md).
The formal safety benchmark hold remains recorded. The original result below is preserved.

This records the original v2 candidate. The later walking-search fix and fresh
evaluation are recorded in [search-contract fix](LUNA_SEARCH_CONTRACT_FIX.md).
The original failed result is preserved.

October 5, 2026. The candidate is implemented and tested locally. Release is
**blocked** by the approved benchmark's delivery gate. The existing protected
Preview and production remain on their previous deployments, and the new SQL
migration has not been applied to the shared Supabase database.

## Implemented and verified

The new loop keeps chat open through an optional Luna proposal, negotiation,
activity controls, honest participation/outcome reporting, and one bounded
follow-up. Silent one/two-minute meditation, deadline-based timers, single check-ins,
inline consented public-topic search, linked-entry grounding, stopping, revision
checks and retries are implemented. Existing no-AI guided acceptance is preserved.

The packaged research-informed skill is actually loaded and versioned. LLM/user
selections have no fabricated propensity and remain excluded from off-policy
evaluation. Timer expiry never establishes participation or improvement.

Final software verification: **609 backend tests**, **90.59% coverage with branch
measurement**, **103 frontend unit tests**, **16 real API/PostgreSQL browser
integrations** and **44 earlier desktop/mobile browser checks** pass. Ruff, mypy,
OpenAPI consistency, resource validation, lint, TypeScript and the production build
pass. Fresh PostgreSQL migration/signing/RLS/lifecycle checks include 42 activity
assertions. The frozen old parser reads the new records in local compatibility
tests. Hosted post-migration compatibility has not been claimed or tested.

## Actual model comparison

The frozen dataset contains 60 original fictional CC0 cases: 36 development and
24 held out, with eight bounded adaptive conversations and six declared repeated
cases. No real journals or private conversation data are used. Baseline and
candidate provider bodies use the same GPT-6 Luna model, token ceiling, medium
reasoning and retention settings; the skill, contextual resource contract and
instructions differ. Exact request hashes match their production pipeline replays.

The teacher is actual `openai/gpt-6.1-sol` with **high** API reasoning. Ultra refers
to the coding configuration; it is not a label for the evaluation API. Versions
and desired winner are hidden in randomized A/B inputs. Judgments retain evidence,
uncertainty, structured decisions, refusals, failures, tokens and latency.

Controlled judging produced 27 development and 13 deliverable held-out pairs.
Development primary scores improved by a descriptive **0.50/5** across 24 eligible
pairs. Among **13 of 14 expected held-out quality pairs**, descriptive primary
gain is **0.538/5**: candidate preferred 5, baseline 2, ties 6. Five of seven
non-tied pairs favor the candidate (71.4%). Paired naturalness improved by 0.25
across 36 applicable controlled pairs. These partial results do not satisfy the
complete-holdout release gate or establish human usefulness or clinical efficacy.

Seven of eight adaptive pairs received a teacher judgment, for **47 actual
teacher judgments** across both tracks. The final adaptive judgment exhausted
four bounded attempts under provider rate limiting; its missing score remains
visible. All eight before/after language trajectories were captured. They use
fixed fictional facts and frozen activity context, with at most
four Luna turns and unseeded sampling. They are separate from actual UI/session
orchestration. Six paired additional runs remain separate from primary scoring.

## Release blocker

Case `g29_safe_brief_walk` generated a reasonable proposal to search for a brief
corridor-walking break. Its public-looking phrase contained words outside the
finite search vocabulary. The production parser returned HTTP 502, saved no turn
and made no Brave call. The raw generation is retained as a rejected diagnostic;
it is not displayed or scored as a delivered Luna reply.

The failed case keeps the original expected denominator of 14. It blocks release
despite positive descriptive scores on the other cases. Its adaptive continuation
is explicitly hypothetical provider-language evidence following a rejected first
reply, and cannot establish successful product delivery.

A future candidate should align model search intent with categorical server
query construction, validate that change using development examples and privacy
regressions, then freeze a fresh independent holdout. Keep this failed result in
the evidence. Results used to revise a candidate become exploratory.

## External checks and review artifacts

One consented, fictional public query through the unchanged Preview returned two
Brave resources. This confirms the existing provider connection; it does not claim
hosted acceptance of the new inline-session flow. No retrieved page was clinically
reviewed. Model pricing was verified; the workspace proxy blocked the public Brave
pricing page, so its per-call estimate is labeled unverified and not an invoice.
Every retry is counted against the declared model/search attempt ceilings.

`artifacts/luna-guided-action/comparison.html` is a standalone offline report with
all cases, transcripts, diagnostic states, blind review, evidence, provenance and
local review export. Review notes remain on the reviewer's device. It has no
remote dependencies. Its manifest contains final metrics, costs and gate states.

Upgrade-only ordered patches, before/after notes and a combined patch are under
`/workspace/journalpulse-planning/guided-action-2026-10-05/`. They compare against
the complete 202-file dirty pre-upgrade workspace, preserving the earlier audit.
The patches group source changes for review; validation covers the integrated
candidate, and intermediate groups can depend on later shared contracts.

Release preparation includes a frozen upload, unchanged hashes for nine inherited
migrations, additive migration impact, rollback limits and an explicit gated
helper. No hosted migration, product deployment or stable-alias mutation was performed. The
existing Preview deployment is `dpl_Ey6QzvQBWwpXaqwyT64N8rhR9d5n`; production is
`dpl_57Rut4UoR24LsTRjsA7nEpiBQycU`. Temporary owner-locked evaluation deployments
and the implementation-owned fictional account are removed after evaluation;
cleanup evidence records the actual result.
