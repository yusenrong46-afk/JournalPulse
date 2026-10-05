# Luna guided-action fix review

The walking-search bug is fixed locally. The formal release benchmark is still
incomplete under evaluator rate limits, so the existing Preview and shared
Supabase remain unchanged. All original failed results are preserved.

Open `comparison.html` for all 60 formal before/after cases, actual delivery
states, blind version labels, teacher evidence and local review-note export.
There are 16 fresh heldout judgments, including 15/20 expected quality pairs;
unobserved judgments and incomplete adaptive trajectories stay visible.

Open `browser-agent-checks.html` for four supplemental paired follow-ups and eight
actual GPT-6 Luna replies. These post-hoc Codex-written questions use the same
wording in each pair, zero teacher calls, and no substitute formal scores.
`AGENT_REVIEW.md` explains findings and limitations.

Both reports work offline with no external dependencies. Their final exact bytes
passed Chromium checks; review notes remain on the reader’s device. Local UI/API
session tests use named Luna/Brave test doubles, distinct from real provider
language captures and real Brave connectivity.

`UNIT_NOTES.md` explains before/after source changes. `search-contract-fix.patch`
and `combined-upgrade.patch` preserve the inherited dirty audited workspace.
The `evidence/` directory contains passed software checks, partial formal scores,
unchanged budgets, cleanup, retained deployment identities and release gates.
The release helper rejects the incomplete candidate before external mutation.

Original budgets stayed unchanged: 219/600 Luna attempts, 171/180 teacher attempts,
2/6 Brave calls and a conservative $22.40/$25 reservation. Reported model usage is
not an invoice; Brave pricing remains unverified. Three temporary evaluation
deployments and the fictional account were deleted after capture.
