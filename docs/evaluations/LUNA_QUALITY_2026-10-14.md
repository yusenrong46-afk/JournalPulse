# Frozen Luna quality evaluation — protocol and current status

Status: **NOT RUN — secure OpenRouter credential input is pending.** No model
quality score exists for this run. No paid generation has been recorded by this
runner. Historical 61.5% non-tied wins must not be presented as today's result.

The immutable bundle is `artifacts/reliability-2026-10-14/quality-run-v2`. Its manifest
records SHA256 for exact source, prompt, catalog, dataset, runner, dependencies and
prepared requests. Version 2 was frozen after local account-closure implementation and a pre-run
latency-logging repair. Version 1 remains untouched and unrun; the scenario/request
packet hash is identical. Later application changes are not automatically qualified
by these results. Before
release, compare the inference-relevant code and prepared request hashes. Preserve
both identities and report any difference rather than relabelling the benchmark.

Design: 100 original fictional English cases, paired into 50 scenario families.
Forty development cases and 60 reserved cases keep each family wholly within one
split. The cases are assistant-authored, not an independent human test set. There
is no live journal data. No tuning or repeated run against reserved cases is allowed.

Candidate: `openai/gpt-6-luna`. Judge: `anthropic/claude-sonnet-4.6`, a separate
model family. Both require currently available zero-retention structured-output
endpoints. The runner locks routing to the eligible endpoint, disables provider
fallbacks, and stops if either requirement is unavailable. This is **AI-judged
evidence**, with human review marked NOT RUN unless actually performed.

The frozen rubric grades relevance, grounding in supplied facts,
instruction-following/consent, and practical usefulness from 1 to 5. Critical
safety failures are reported separately. The 14 deterministic support routes do
not call the candidate; the other 86 routes can call the candidate and judge.
Errors, refusals, invalid structured outputs, missing results and route mismatches
remain visible in denominators. Family-level bootstrap intervals represent sample
uncertainty, not clinical benefit or population-level accuracy. This absolute
quality evaluation is not a new measured gain over the historical baseline.

Run through the wrapper in your own interactive terminal:

```sh
/Users/thomas/Documents/ChatGPT/JP/.venv/bin/python /Users/thomas/Documents/ChatGPT/JP/scripts/run_reliability_eval.py
```

Paste the key only at its hidden prompt. The shared ledger
`artifacts/reliability-2026-10-14/quality-budget.json` enforces US$10 total including
failed attempts. Before every request it reserves a conservative maximum using
published endpoint prices, UTF-8 input bounds and output caps. Actual reported
cost and generation ID must be reconciled before another call. An unknown charge,
transport interruption or pricing breach stops continuation; do not delete the
ledger, change its path, or retry to manufacture a favorable result.

The runner retains fictional requests/responses, latency, costs, validation errors,
and a generated report. Report negative/inconclusive results faithfully. Offline
control tests demonstrate cap/freeze/error mechanics only; they do not establish
candidate or judge quality.
