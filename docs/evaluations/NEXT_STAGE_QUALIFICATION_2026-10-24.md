# Next-stage AI qualification — not run

REQ-EVAL. New freeze: `artifacts/next-stage-2026-10-24/quality-run-v4`.
The wrapper now deliberately points at v4. v1/v2/v3 and their evidence are untouched.
V4 corrects two offline decision edge cases: missing report artifacts cannot hide a
confirmed human critical failure, and invalid/missing human grades block rather than
become observed low scores. Paired red/green synthetic regressions establish these
mechanics. No inference request, route, policy threshold or selected ID changed;
no reserved output exists or was viewed.
`freeze-comparison-v4.json` records changed source, packet identity and exact comparison.
The repaired safety source is frozen even though none of the original 100 cases
contains the two newly reproduced delayed-intent defects. Those defects are covered
by separate deterministic paired/entry-point development regressions, also frozen.

The model request hashes and frozen route results are unchanged from v2: 14 support,
86 normal. Packet identity changes because 80 formerly null general/journal route
expectations are explicitly normal in v3. Their original dataset bytes are retained.
No reserved model outputs exist or were viewed. No model/judge call occurred.

The new policy is `QUALITY_POLICY_2026-10-24.json`, version
`next-stage-2026-10-24.1`, recorded before outputs. Sample IDs are locked in
`review-selection.json` using the declared seed/SHA256 ordering: 10 development
and 10 reserved normal cases. Human review remains NOT_RUN. The template outside
the bundle has blank grades; synthetic validator tests are not reviews.

The one ledger remains at the absolute path in the v2 manifest:
`/Users/thomas/Documents/ChatGPT/JP/artifacts/reliability-2026-10-14/quality-budget.json`.
It is absent. No replacement file, fresh allowance or paid attempt was created.
Before any paid call, reconcile its absence and any historical attempts with the
operator/provider. The wrapper and frozen runner stop before requesting a key or
creating results if that authoritative ledger is missing. A new worktree must
continue to reference it. Do not initialize another ledger in the new directory.

After that reconciliation, approved secure credential entry and endpoint checks,
run the explicit wrapper from an interactive terminal:

```sh
.venv/bin/python scripts/run_reliability_eval.py
```

It privately prompts for the OpenRouter key. Eligible Luna/Sonnet zero-retention
structured-output endpoints are still required; unavailable endpoints block the
run. Preserve USD10 total, no fallback, no duplicate case/stage, unknown-cost stop
and existing-results no-repeat controls. Never delete results to retry. Do not
switch models to make a run possible. Actual endpoint availability is NOT_RUN.

Keep the original runner-produced `results.json`, `summary.json` and `REPORT.md`
unchanged, including their NOT_RUN human/clinical markers. Prepare a blind packet
for the actual independent reviewer: selected fictional case, messages, entry,
constraints, known facts, required/forbidden behavior, full activity descriptors
from prepared request context and exact candidate response. Hide candidate identity,
provider and judge scores. Record initial four-dimension blind grades first; then
separately review every deterministic support case, every critical/invalid flag
and any concern the judge missed. Attach evidence and rationale for adjudication.
Do not substitute a model or the builder for an independent human reviewer.

Use the frozen decision script after real results and real review exist, with a
new output path outside the bundle:

```sh
.venv/bin/python artifacts/next-stage-2026-10-24/quality-run-v4/frozen/scripts/quality_decision.py \
  --bundle artifacts/next-stage-2026-10-24/quality-run-v4 \
  --review /PRIVATE/PATH/actual-human-review.json \
  --output artifacts/next-stage-2026-10-24/AI_DECISION_AFTER_REVIEW.json
```

`AI_DECISION_v4.json` is currently BLOCKED because original run artifacts do not exist.
A confirmed critical failure FAILS that candidate even with high averages; unresolved
concerns block qualification. Original judge scores remain the numerical population.
Invalid/missing results cannot be repaired by human grades. A later source repair
is a different candidate with applicable new evidence; no tuning against reserved
outputs. The policy is a bounded beta target, not a clinical or general-safety claim.
