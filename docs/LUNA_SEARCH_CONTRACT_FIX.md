# Luna search-contract fix

October 5, 2026. The walking-search delivery bug is fixed locally. Fresh software
and provider checks verify the fix. The formal release evaluation remains
incomplete because of evaluator rate limits; the Preview has not been updated.
The failed original v2 result is preserved.

Subsequent local audit fixes advance the current packaged skill and prompt to
`guided-action-2026-10-05.3`: outcome requests now include the actual selected
activity, saved goal, and configured duration as bounded untrusted data. The audit
also corrected bare unsafe-feeling routing, stale inferred feelings, and catalog
format constraints. These changes have local synthetic regression evidence;
the live model observations and comparisons below describe `.2` and do not
validate the current `.3` assembly. No new paid evaluation or product deployment
was performed for this audit.

## Before and after

Previously the provider schema allowed any search phrase. The application then
required every word to belong to a small public vocabulary. Luna's reasonable
phrase, `brief corridor walking movement break`, was rejected, producing HTTP
502 and no saved reply.

Luna now chooses one of twelve explicit activity categories. Both the provider
schema and the Python parser enforce the same categories. The server builds the
public query from that category, the goal and validated constraints. For example,
`walking` plus a six-minute limit becomes
`gentle movement gentle walking 6 minute`.

The strict private-word filter remains unchanged for edited queries and search
refinements. Names, contact details, copied journal text and arbitrary model
phrases are still rejected before search. A proposal still requires the person's
explicit consent before the application contacts Brave.

The evaluated packaged skill and prompt were version `guided-action-2026-10-05.2`.

## Verification

- 617 backend tests pass, with 90.67% coverage with branch measurement.
- All 16 real API/PostgreSQL browser integration tests pass.
- Ruff, mypy and OpenAPI consistency pass.
- The actual built wheel contains the revised skill and the twelve categories.
- All 54 candidate model requests match their actual production-pipeline replay:
  49 accepted completions and five correctly handled native refusals. Six
  deterministic fixtures also pass; there are no delivery failures.
- The original walking case was rerun with actual GPT-6 Luna. It selected
  `walking`, returned HTTP 200 through the real parser, and saved its reply.
  This rerun is a regression check, not independent held-out quality evidence.
- One consented fictional public query on the existing Preview returned three
  Brave resources. This verifies the provider connection and compiled query;
  hosted acceptance of the new inline-session flow is a separate release check.

## Evaluation and release

The original failed comparison remains intact. Dataset v3 retains the 36
development cases and freezes 24 fresh fictional held-out families before the
baseline capture and code change. Both versions use the same Luna model,
retention policy and token/reasoning settings. The adaptive track has a
predeclared two-assistant-turn cap within the schema's four-turn ceiling.

The fresh run captured all 54 requested Luna observations and six deterministic
fixtures. Formal GPT-6.1 Sol judging produced 16 held-out judgments, including
15 of the 20 expected quality pairs. Among those 15 pairs the candidate was
preferred eight times, with seven ties, and descriptive primary gain was 1.067
points on the five-point scale. These are **partial** results: they do not pass
the complete-holdout gate. Formal adaptive trajectories are observed for seven
baseline and five candidate scenarios out of eight; none has a fresh paired
teacher judgment. Four development safety cases remain unresolved in the formal
report, distinct from passed deterministic software checks.

Judgments resume from matching request hashes and preserve every attempt. Calls
were spaced after responses, with bounded retries. Provider metadata showed a
degraded OpenAI flex endpoint; preferring Azure still returned rate limits. Case
`r15_uncertain_effect` exhausted four rate-limited attempts. Further calls were
paused between requests; no in-flight response was discarded.

The user requested browser/agent capabilities instead of approving a larger
budget. Chromium read the provider's official limits, error and routing documents
using Playwright's configured CA validation; certificate checks were not disabled.
The documents recommend honoring `Retry-After`, bounded backoff and allowing other
eligible providers. The retained gateway errors omit detailed rate-limit headers,
so the precise account-versus-upstream quota cannot be established from this run.
Browser access cannot remove that quota or supply missing independent scores.

Supplemental agent verification reads all 60 captured case pairs and adds four
post-hoc follow-ups with identical wording for both versions. All eight extra
replies are actual GPT-6 Luna generations and pass structured parsing. They keep
the two-minute/no-breath-focus constraints, permit eyes open and stopping, shrink
a reading task, and distinguish past journal fear from current relief. These
unblinded checks use no teacher calls and do not replace the formal benchmark.
They are language checks, separate from real UI/session orchestration. The older
continuation gateways omit actual transport body hashes; their request provenance
is labeled reconstruction rather than exact replay.

The original global ceilings remain unchanged: 219 of 600 Luna attempts, 171 of
180 teacher attempts, and two of six Brave calls. Conservative estimated
reservation is $22.40 of $25. Retained unique response envelopes report about
$1.887 in model usage; this is not an invoice and excludes model usage from the
shared-search endpoint. Brave pricing remains unverified. No credits were bought
and no retry or spend limit was increased.

The standalone [formal comparison](../artifacts/luna-guided-action-search-fix/comparison.html)
and [supplemental conversations](../artifacts/luna-guided-action-search-fix/browser-agent-checks.html)
pass Chromium checks. Filters, blind labels, refusal states and review export work;
the supplemental report also fits a mobile viewport. Both have no external requests
or JavaScript errors. [Agent review notes](../artifacts/luna-guided-action-search-fix/AGENT_REVIEW.md)
record minor summary-pronoun, legacy-label and phrasing issues without tuning the
candidate against these held-out responses.

The existing Preview, production and shared Supabase schema remain unchanged
until the revised candidate passes the applicable release gates. The three
temporary owner-locked evaluation deployments and the fictional account were
deleted after capture; retained product aliases are unchanged.
