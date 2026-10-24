# A defect and request path Yusen can explain

REQ-HANDOFF preparation. This written guide is not evidence Yusen has given the
walkthrough; the actual ownership demonstration remains NOT_RUN.

The defect was a denial whose meaning changed with a time qualifier. The exact
review fixtures were `I won't kill myself until tonight` and
`I will not harm myself before I say goodbye`. Both routed NORMAL at baseline.
The router marked the denial's span safe before considering time, so even
“tonight” could not recover the risk match. Some denied verbs also lacked a separate
affirmative pattern. Fixing only the negated span would therefore still miss them.

In `src/journalpulse/safety.py`, normalization and existing clause/history handling
stay in place. Only an immediate `until`, `before` or `yet` after a matched denial
marks it as a postponement. Genuine denials retain suppression. A postponement is
checked with the same historical/present rules, including when its verb lacks a
separate affirmative match. This is a bounded English rule, not semantic or
clinical understanding. “Not suicidal yet” deliberately takes the conservative
support route; the eight original review-only English cases remain unresolved.

Trace a fictional `POST /v1/conversations/{id}/messages` request:

1. FastAPI resolves the authenticated owner and request/data revision. Owner and
   conversation/source lifecycle checks precede use of private context.
2. `conversations.py` assesses the current message and, only when explicitly linked,
   the owned source entry. An unlinked chat cannot read the journal automatically.
3. The repaired delayed-intent fixture enters deterministic support. The persisted
   chat has support mode, no action readiness or unsafe ordinary activity offer.
   Generation is bypassed. Follow-up/support behavior stays sticky for the chat.
4. Persistence commits under revision/erasure/lifecycle fences, so stale generated
   work cannot reopen a closed/deleted/replaced chat. The browser receives canonical
   state and removes inapplicable controls.

Run `pytest tests/test_temporal_negation.py tests/test_english_safety_entrypoints.py`.
Show baseline `routing-before.json`, final `routing-after.json` and the exact review
source record. The paired denials/history controls remain normal. The actual API
entry-point tests capture zero model/search calls for support; standalone support
also bypasses generation, and stopped participation remains user-reported.

For the contrasting save path, the journal page's coordinator in
`web/lib/journal-save.ts` holds the exact submitted words, owner, receipt and erasure
boundary outside the page. The save endpoint persists without AI. Only an
authoritative receipt allows Saved; a lost response reuses that receipt. Newer
writing survives an older response. Account-switch/erasure epochs discard stale
work. `tests/test_journals_api.py` and the real-stack audit-repairs browser tests
establish complementary API/persistence and navigation evidence.

In a real walkthrough, have Yusen locate the boundary, explain a denial/control,
trace one rejected stale request and make a small reviewed change with its focused
regression. Record the actual candidate and result separately. No personal demo,
paid-model qualification, hosted release or user-benefit claim is implied here.
