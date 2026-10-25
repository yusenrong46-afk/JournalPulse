# Current R1 qualification procedure — 25 October 2026

Current bundle: `artifacts/r1-follow-up-2026-10-25/quality-run-v5`.
The wrapper explicitly selects v5. V1–v4 remain unchanged and unrun. There are no
reserved model outputs to inspect or tune against.

Policy `next-stage-2026-10-25.2` was recorded before outputs. It fixes an integrity
gap: a human record with the same input-packet digest but another source or another
set of model responses could previously qualify. Two new red regressions reproduce
that false PASS. The corrected decision requires the exact frozen manifest hash,
prepared-packet digest and original results.json hash. Numerical thresholds, rubric,
models, endpoint restrictions, selection seed and the USD10 total cap are unchanged.

The blind-review exporter is now frozen alongside the evaluator/decision tool. It
provides the locked20 cases, exact responses and full fictional source/activity
context without judge scores or model/provider metadata. Missing/invalid outputs
stay in the sample denominator. Support/flagged/invalid cases are exported separately
for adjudication after initial blind grades. A response that self-identifies is not
rewritten: record any resulting blinding limitation. Files are created privately,
refuse overwrites and cannot be written inside the original bundle.

## Ledger and preflight

The canonical ledger remains:
`/Users/thomas/Documents/ChatGPT/JP/artifacts/reliability-2026-10-14/quality-budget.json`.
Its first initialization followed the user's confirmation that no runner had ever
started, plus local history/worktree/artifact reconciliation. Current recorded spend
isUSD0 and attempts0; capUSD10. This is the existing protocol allowance, not a fresh
allowance and not the provider account balance. The live ledger/lock are ignored by
Git and must never be copied/reset per checkout. Every frozen version binds the same
absolute path. The ledger receipt is under the October24 follow-up artifacts.

The October25 unauthenticated endpoint preflight found eligible Luna/Azure and
Sonnet/Amazon Bedrock metadata. This proves metadata availability at that instant,
not model generation, provider retention practice or future endpoint availability.
The frozen run rechecks eligibility and locks actual routing/prices. Cost projections
are planning scenarios only: approximatelyUSD7.07 throughUSD11.20 in conservative
reservations for the declared response-byte scenarios. A long-output scenario can
exceed the cap; the runner must stop before its next reservation rather than alter
the cap, omit cases or switch providers. No generation/accounting request occurred.

## Actual execution and review

Only after the private credential and preceding release prerequisites are available:

```sh
.venv/bin/python scripts/run_reliability_eval.py
```

The runner prompts privately for the provider credential. No new key, fallback,
paid-model result or accounting receipt has been manufactured. Do not repeat a
bundle that already has results or a case/stage already in the shared ledger.
Unknown charges, missing generation IDs, privacy-routing/identity mismatch or a
pricing breach remain stop conditions. Preserve all original model artifacts.

After the run terminates and original results/summary/report exist, export a fresh
private review directory with the frozen helper (substitute a private new path):

```sh
.venv/bin/python artifacts/r1-follow-up-2026-10-25/quality-run-v5/frozen/scripts/prepare_quality_review.py \
  --bundle artifacts/r1-follow-up-2026-10-25/quality-run-v5 \
  --output /PRIVATE/NEW_REVIEW_DIRECTORY
```

Give the independent reviewer `BLIND_CASES.json` and the blank human template first.
They record initial scores without seeing judge grades. Only afterward supply
`ADJUDICATION_AFTER_BLIND.json`, including all deterministic support and flagged or
invalid cases. Preserve initial grades, actual reviewer independence, evidence and
rationale. Templates and synthetic tests are not human review. Human assessment
cannot repair missing/invalid model results; unresolved concerns block and a
confirmed critical failure fails the candidate.

Run the frozen decision with the completed real review and a new output path outside
the bundle. It binds the exact source and original result bytes and preserves all
original scores/reports, including their runner-produced NOT_RUN human/clinical
markers. Keep the actual human record and release decision as separate artifacts.

Current AI decision remains BLOCKED: there are no real candidate/judge outputs or
independent human review. Default scope remains R1 plus preparation for R2. Actual
participant sessions, seven-day observation and R3 ownership demos are later gates;
they are not silently substituted for R1 model/operational qualification.
