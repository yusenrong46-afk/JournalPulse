# Full Luna benchmark: prepared run

Completed October 5, 2026 within the authorized $7 budget. See [the results](LUNA_FULL_RELEASE_BENCHMARK_2026-10-05.md). Quality gates pass; release remains incomplete because one critical safety grade is uncertain.

The current `.3` prompt and runtime are frozen. Compare them against the preserved
pre-upgrade baseline on **60 scenarios**: 36 development cases and 24 new fictional
final-test cases. Keep the prompt unchanged throughout testing. New cases were
authored after the candidate freeze, but by the coding assistant; this does not
establish independent annotation or exclusion from model training.

Both request bundles are prepared. All 36 existing baseline development records
are eligible for reuse after exact request and fixture matching. Old candidate
responses and teacher scores will not count as current `.3` evidence.

The run includes eight longer scenario comparisons, with at most two assistant
turns per version, six declared repeated cases, actual GPT-6.1 Sol judgments with
randomized A/B labels, replay through the real application parser and lifecycle,
and one bounded live search check if available. Software gates and language
judgments remain separate. Produce an HTML before/after report with missing,
failed, refused and unsupported results retained explicitly.

## Approved budget and calls

The user authorized **$7** on October 5, 2026: “I will authorize 7 dollars.”
Treat this as the total benchmark accounting ceiling, including approximately
$1.8952 of retained historical provider response costs and a $0.35 allowance for
unreconciled historical charges. This leaves approximately **$4.7548** for new
accounted spending. Historical records are not a verified invoice; the allowance
cannot establish the actual amount of missing past charges. No credit purchase
or unrelated usage is authorized.

Reserve each new request before sending it using serialized UTF-8 request bytes
as a conservative token bound, current input/cache-write rates, maximum output,
and a 15% contingency. Replace a reservation with provider-reported cost plus
contingency when usage is present; retain the full reservation when usage is
missing. Stop before another reservation would exceed $7. This replaces the
unapproved $40 proposal and flat per-attempt planning reservations for this run;
preserve the old ledger as historical evidence.

| Item | Bound |
| --- | --- |
| Controlled teacher judgments | Up to 54 |
| Simulated-user continuations | Up to 16 |
| Adaptive pair judgments | Up to 8 |
| Additional teacher attempts including retries | At most 120 |
| Additional Luna attempts including repeats/retries | At most 150 |
| Additional live Brave calls | At most 1 |
| Global attempt bounds | Teacher 291, Luna 600, Brave 6 |
| Approved benchmark accounting ceiling | **$7 total** |

Count all attempts, including failures. Use complete-request caching, sequential
teacher requests, and bounded retries for transient failures. Completion does
not imply passing: quality, safety and user-control thresholds remain unchanged.
Do not tune the prompt or cases during this run.

This run does not include a product deployment or shared database migration.
Remove temporary private evaluation resources afterwards. Preserve raw responses,
request identities, approval, reservations, costs and failures in
`/workspace/journalpulse-planning/luna-release-benchmark-2026-10-05/`.
