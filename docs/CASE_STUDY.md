<!-- agent_draft_unreviewed: written by the coding agent from repository files; Thomas to check the framing and every claim before sharing. -->

# Case study: JournalPulse

Written as one turn of the customer loop: who it is for, what they need, what was built,
how it was measured, what was learned, and what changed. Every claim points to a file in this
repository; numbers come from generated files, not from memory.

## 1. Who it is for

English-speaking adults who want to write privately and, sometimes, talk a day through and pick
one small next step ([README](../README.md)). It is a research beta run by one person, and it is
explicitly not therapy or crisis care: the 9-8-8 line is on the sign-in screen and on every demo
view ([`web/components/demo-banner.tsx`](../web/components/demo-banner.tsx)).

**What we know about them, honestly.** There has been no user study. Evidence about needs comes
from the author's own manual evaluations of the preview
([CURRENT_PRODUCT_MANUAL_EVALUATION.md](CURRENT_PRODUCT_MANUAL_EVALUATION.md),
[LUNA_UI_AUDIT_2026-10-05.md](LUNA_UI_AUDIT_2026-10-05.md)) and from fictional benchmark scenarios.

## 2. What they need (the constraints that shaped the design)

* **Their words stay theirs.** Saving an entry never involves AI; sharing an entry with a chat is an
  explicit choice; chat text is cleared 24 hours after a chat goes idle unless they keep it
  ([WALKTHROUGH Q7](WALKTHROUGH.md#q7-how-long-is-chat-text-kept)).
* **Nothing unsafe reaches a model.** Risk language gets a fixed support reply before any model call
  ([WALKTHROUGH Q2](WALKTHROUGH.md#q2-where-is-the-consent-and-safety-gate-and-what-are-its-limits)).
* **It works without AI.** Simple mode is scripted and local
  ([WALKTHROUGH Q9](WALKTHROUGH.md#q9-how-does-simple-mode-work-with-no-ai)).
* **Nothing is half-saved.** Retries, late replies and old tabs cannot corrupt or resurrect data
  ([WALKTHROUGH Q1](WALKTHROUGH.md#q1-what-happens-when-a-late-or-duplicate-ai-reply-arrives-after-a-newer-save),
  [Q5](WALKTHROUGH.md#q5-what-does-delete-all-my-data-remove)).

## 3. What was built

A FastAPI service ([`src/journalpulse/`](../src/journalpulse/)) over Supabase PostgreSQL with
row-level security and signed writes ([`supabase/migrations/`](../supabase/migrations/)), and a
Next.js static web app ([`web/`](../web/)) served by the same API. Luna, the AI companion, runs
behind the safety, consent and rate-limit gates; Simple Luna runs without a model.

## 4. How it was measured

From [`docs/evaluations/portfolio/results.json`](evaluations/portfolio/results.json) (`make eval`,
offline, no model calls):

* **Engineering:** 1,436 backend tests with 92.07% branch-aware coverage, and 238 frontend unit tests.
* **Safety routing:** 29 of 34 development cases pass, 3 miss support, 2 need review, 0 false
  alarms. The set was used during development, so it is labelled NOT_QUALIFIED.
* **Luna's activity suggestions:** on 20 held-out pairs from a 60-scenario benchmark graded by a
  model, the new prompt won 8, lost 5 and tied 7. The mean gain was +0.35, but the approximate 95%
  interval (-0.13 to 0.83) includes zero
  ([LUNA_FULL_RELEASE_BENCHMARK_2026-10-05.md](LUNA_FULL_RELEASE_BENCHMARK_2026-10-05.md)).
* **Operations:** one command builds and checks a whole environment
  ([`outputs/onboarding/first-run/REPORT.md`](../outputs/onboarding/first-run/REPORT.md)), and a
  database-outage drill was run ([INCIDENT_DRILL.md](INCIDENT_DRILL.md)).

## 5. What was learned

* **A good average is not a release.** The benchmark beat its targets, but one critical case (a
  chest-pain warning) came out "uncertain", so the release gate held. The gate passes only when
  every check passes ([`src/journalpulse/guided_evaluation.py`](../src/journalpulse/guided_evaluation.py),
  `build_comparison`).
* **Phrase lists miss real phrasing.** "I can't see any way out" is still missed
  ([SENTINEL_RELEASE_2026-10-06.md](SENTINEL_RELEASE_2026-10-06.md), remaining risks). The safety
  numbers are reported with that label, not hidden.
* **Nobody can try it without an account.** The hosted app requires sign-in, so a reviewer or a
  prospective participant following a link cannot see it work. That is why this upgrade added the
  demo sandbox ([DEPLOY_DEMO.md](DEPLOY_DEMO.md)).

## 6. What changed

* A **demo sandbox**: anonymous identities, three fictional entries, AI and search refused in code,
  erased after 24 hours ([`src/journalpulse/demo.py`](../src/journalpulse/demo.py),
  [`supabase/migrations/202611180001_demo_sandbox.sql`](../supabase/migrations/202611180001_demo_sandbox.sql)).
* **Measured, repeatable claims**: the README's numbers are generated and checked by
  `make check-docs`, and a drop below the committed baseline fails CI
  ([`scripts/portfolio_eval.py`](../scripts/portfolio_eval.py)).
* **Operated, not just deployed**: build identity, an hourly synthetic check with an SLO, and a
  database-outage runbook ([SLO.md](SLO.md), [OPERATIONS.md](OPERATIONS.md)).

## 7. Next turn of the loop

Put the demo in front of five to ten real people (the demo is ready, the hosted switch is an owner
step in [DEPLOY_DEMO.md](DEPLOY_DEMO.md)), qualify a held-out safety set before claiming safety
performance, and rerun the benchmark against the current runtime prompt, which the 2026-10-05 run
does not cover ([WALKTHROUGH Q10](WALKTHROUGH.md#q10-how-do-you-know-which-prompt-produced-a-reply)).
