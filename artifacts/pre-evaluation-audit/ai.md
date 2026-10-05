# AI, safety, and resource audit — October 5, 2026

Scope: current `/workspace/JournalPulse` checkout before user evaluation. The inherited
dirty tree and original evaluation evidence were preserved. Review and reproductions
used local synthetic fixtures only. No model/search charges, external mutations,
deployments, resets, commits, or held-out-driven prompt tuning occurred.

## Confirmed findings

### P1: bare unsafe-feeling statement bypassed safety — fixed locally

`src/journalpulse/safety.py:10` placed a required space before the optional suffix
in `feel safe (?:right now|tonight|today|alone)?`. Clause normalization strips the
space, so `I don't feel safe.` incorrectly entered normal chat. A local real API
fixture returned normal mode, allowed an action, and invoked its synthetic model
once; adding `right now` correctly bypassed the model.

With root authorization, the resource/safety child moved the space inside the
optional group and added narrow unit/API regressions. The six new cases failed
before the fix. Afterwards the focused safety/support suite passed **25 tests**
(11 deselected), and Ruff passed for the three touched files. Bare straight and
curly apostrophes now produce support mode, readiness false, and zero model calls.
Existing negation and explicitly documented detection-gap tests still pass.

Changed files: `src/journalpulse/safety.py`, `tests/test_research_beta_safety.py`,
`tests/test_conversations_api.py`. These changes preserve the prior dirty-tree
edits. Full before/after evidence is in [resource-safety.md](resource-safety.md).

### P1: inline search browser payload violated the nested API contract

`web/components/activity-session-discovery.tsx:62-70` spread the five activity
constraint fields into the request root. `src/journalpulse/inline_discovery.py:38-45`
accepts a nested `constraints` object and forbids extra fields. Even a new chat
contains all five default fields, so consenting to the inline search produced
HTTP 422 before any search call.

A local TestClient reproduction of the exact browser payload produced five
`extra_forbidden` errors and zero stub search calls. Nesting the same values
produced HTTP 200 and an offer. The frontend reviewer was notified and owns any
product correction. This is separate from the already verified `.2` model search
category fix. Evidence: [resource-safety.md](resource-safety.md).

### P2: outcome generation omitted which activity the person selected — fixed locally

`src/journalpulse/activity_chat.py:200-228` supplies the participant report but
never includes `session.resource`, `session.goal`, or actual session duration in
the follow-up request. `chat_activity_context()` includes a generic approved
candidate pool and latest status; it does not identify the selected candidate or
an external search result.

Reproduction: two valid completed ActivitySession objects with the same report,
one for a two-minute meditation/settle goal and the other for a fictional external
communication workbook/connect goal, produce **identical provider request
objects** through `generate_activity_follow_up()` and `build_guided_request()`.
The workbook title is absent from its request. The repository fixture supplies
the actual current session, so the request correctly reports `activity_state` as
completed; selection identity is what is missing.

This deterministically prevents the model from grounding its feedback in which
search result the person selected, the goal saved with that activity, or an
adjusted timer duration. It does not establish that every generated follow-up
will be wrong. Include a bounded, explicitly untrusted session descriptor in the
user data and verify changing the selected activity changes the request without
moving its text into system instructions. Product edits were held for root scope
review initially; the authorized fix is recorded below.

### P2: empty current feeling suggestions preserved an outdated label — fixed locally

`src/journalpulse/conversations.py:587` assigns
`list(completion.feelings) or conversation.feelings`. An empty list is valid in
both provider and local schemas, but it cannot clear an earlier suggestion.

A two-turn local real API reproduction uses the actual
`OpenRouterConversationClient` guided parser with `httpx.MockTransport`. Turn one
returns `feelings=["anxious"]`; after a correction asking to clear that label,
turn two returns a valid `feelings=[]`. Both responses are HTTP 200, but the saved
conversation still has `["anxious"]`. `web/app/talk/page.tsx:582-583` and 611 use
these suggestions to preselect the feelings controls.

Current model suggestions should be able to replace the previous suggestions
with an empty list. Explicit confirmed feelings have a separate field. Product
edits were held for root scope review initially; the authorized fix is recorded below.

### P2: a website container was treated as no-video/no-audio evidence — fixed locally

`src/journalpulse/activity_resources.py:195` derives these two capabilities from
`resource_type != "video"`. The catalog's `move_nhs_fitness_studio` is a website
titled **NHS Fitness Studio Exercise Videos**, explicitly summarized as guided
workout videos, but resolves to `no_audio=True, no_video=True` and passes both
hard constraints.

This later catalog item is outside the ordinary first-sixteen unexcluded model
pool, but can enter after exclusions. The helper contract defect is confirmed;
an actual model recommendation was not generated. Use reviewed capability
metadata, treating unknown format as unable to satisfy a hard constraint, rather
than inferring accessibility from the website container type. Evidence and
repro: [resource-safety.md](resource-safety.md).

## Other input mismatches passed to the frontend reviewer

The inline feedback placeholder `shorter, seated, no video` contains punctuation
rejected by the restricted public-query validator. Its textarea accepts 600
characters while the API accepts 160. These are additional deterministic input
contract inconsistencies currently masked by the request-shape failure.

## Verified behavior and bounded limitations

- The initial reviewed packaged skill and prompt were `.2`. The authorized
  context correction advances both to `guided-action-2026-10-05.3`; the actual
  provider request still uses the shared builder and categorical search schema.
- The real parser checks IDs against the server candidates, enforces activity
  constraints and available-state limits, rejects unknown fields/coercions, and
  handles native refusals and truncation before saving a reply. Rejected values
  are not included in diagnostic logs/headers.
- Supplied journals, candidate descriptors, and outcome notes remain user data,
  with fixed trusted instructions. Selected journals are rechecked for owner and
  source identity; no broader journal collection is supplied. This confirms
  message construction and server boundaries, not universal resistance of a
  model to adversarial text.
- Search categories compile to restricted public queries. No private-query
  disclosure or signed-resource receipt bypass was reproduced in this bounded
  review. Search snippets deliberately cannot establish a hard duration or
  accessibility constraint; that conservative behavior is already documented.
- Safety remains a narrow English phrase router. The initially observed direct
  future-intent gap (`I will kill myself tonight.`) was subsequently repaired
  after root review, as recorded below. Indirect wording can still be missed,
  and quoted/historical risk can be overdetected; this is not a clinical classifier.
- Following linguistic corrections, preserving constraints unless changed,
  avoiding invented memories, and obeying stopping in generated prose still rely
  partly on the model. No new paid language-quality claims are made.
- The formal `.2` release evaluation remains incomplete under evaluator 429s.
  These checks neither replace missing judgments nor authorize deployment.

## Local evidence

[ai_repros.py](ai_repros.py) is a rerunnable synthetic reproduction for the
outcome-context and stale-feelings findings. [ai_repros_before.json](ai_repros_before.json)
preserves its output before any fixes. It performs zero external calls.

Run from `/workspace/JournalPulse`:

```bash
.venv/bin/python /workspace/journalpulse-planning/pre-evaluation-audit-2026-10-05/ai_repros.py
.venv/bin/pytest -q tests/test_guided_action_prompt.py tests/test_activity_chat.py tests/test_openrouter_conversation_contract.py tests/test_journal_chat.py tests/test_guided_current_intent.py tests/test_activity_stop.py
```

The targeted suite completed with **107 passed**, one existing Starlette
deprecation warning, in 2.68 seconds. The separately authorized safety fix has
its own before/after and focused test results above. No full-suite result is
claimed by this sub-audit.

## Authorized corrections and post-fix evidence

Root subsequently authorized all three P2 corrections. Four new regression cases
failed before their implementation (two resource constraints, actual outcome
context, and clearing inferred feelings).

- `guided_action.py` now defines a strict bounded `ReportedActivityContext`, and
  `activity_chat.py` supplies the selected resource ID/title/kind/format/provenance,
  saved goal, configured duration and instructions in user data. URLs, owner IDs,
  signing receipts and selection tokens are excluded. The regression verifies
  a one-minute configured timer is distinguished from its two-minute catalog
  default and an external selected resource changes the request. Role-looking
  selected-resource text remains absent from system messages.
- `conversations.py` replaces inferred feeling suggestions with the current
  model list, including an empty list. Confirmed self-report fields are separate.
- `activity_resources.py` requires explicit reviewed `no_audio`/`no_video` flags;
  both default false rather than being inferred from a website/game type. The
  NHS video website is now excluded by either hard preference. Built-in known
  capabilities continue to pass their existing tests.

The new prompt context and its instructions are versioned
`guided-action-2026-10-05.3`, packaged-skill SHA256
`abb73f69089a8591e5f30f1ac62e4557cd0a62f45d408d8fdf718f1bd4808270`.
`docs/LUNA_GUIDED_ACTION_IMPLEMENTATION.md` describes the current version and
`docs/LUNA_SEARCH_CONTRACT_FIX.md` explicitly qualifies its retained `.2` live
evidence. No held-out observation or original report was changed, and no paid
evaluation was added. `.2` language-quality observations are not a validation of
the changed `.3` assembly.

[ai_repros_after.json](ai_repros_after.json) preserves the fixed outcomes: the two
selected-activity requests differ, the external selected title is present, and
the second stored feelings list is empty after the actual guided parser runs.

Post-fix focused command:

```bash
.venv/bin/pytest -q tests/test_activity_resources.py tests/test_activity_chat.py tests/test_action_readiness_current_turn.py tests/test_guided_action_prompt.py
```

Result: **53 passed**, one existing Starlette deprecation warning, in 1.69 seconds.
Ruff passed on all eight touched AI/resource code and test files; mypy passed on
`activity_chat.py`, `activity_resources.py`, `guided_action.py`, and `conversations.py`.
The main audit owns the integrated full-suite verification and release decision.

## Additional direct-intent safety correction

Root review correctly identified `I will kill myself tonight.` as a material
explicit-risk routing defect, rather than merely a vague-language limitation.
The authorized bounded correction adds direct first-person `I will`/`I'll`
statements about killing oneself or ending one's life (including normalized curly
apostrophes). It also recognizes clear `do not`/`don't`/`never want to` denials for
these same phrases. Denial matching remains limited to the matched phrase;
separate affirmative statements in the same sentence still route to support.

`tests/test_safety_direct_intent.py` froze 15 synthetic cases before correction:
four direct statements, two mixed-denial statements, five denials, three positive
real-API fixtures, and one denial API fixture. The pre-fix run had **13 failed,
2 passed**. After the minimal `safety.py` change, the focused safety suite had
**39 passed, 12 deselected**; Ruff and mypy passed. The three positive API fixtures
assert support mode, no ordinary action readiness, a safety-router response, and
**zero model calls**. The clear-denial API fixture remains normal with one local
model-double call. No network/model requests or benchmark changes occurred.

Preserved logs: [direct-intent-before.txt](direct-intent-before.txt) and
[direct-intent-after.txt](direct-intent-after.txt). Remaining limitations include
indirect language, phrasing outside the explicit pattern vocabulary, and
quoted/historical interpretation. No broad safety-detection claim is made.
