# Next upgrade: purposeful reflection, activities, and journal access

Recorded 2026-10-05. These are agreed product directions, not implemented
capabilities. Review today's evaluation findings before finalizing tickets and
building the next vertical slices. Existing preview behavior is unchanged.

The active first-slice specification is now
[Luna guided action](LUNA_GUIDED_ACTION_VERTICAL_SLICE_PLAN.md): a benchmark-first,
research-informed chat–recommendation–activity–feedback loop, including an in-chat
meditation timer, inline resource discovery, and the existing linked-journal flow.
It consolidates purposeful reflection and selected parts of items 2–4 below;
automatic journal lookup by date remains a later slice. The specification includes
the GPT-6.1 Sol Ultra execution handoff and the before/after HTML report requirements.

## Upgrade inventory: five main items

| Item | Intended outcome |
| --- | --- |
| 1. Purposeful reflection, first small slice | Luna explores one grounded distinction or uncertainty instead of repeatedly paraphrasing. |
| 2. Optional emotion suggestions | Users can confirm, correct, or add suggested emotions and optionally rate intensity. |
| 3. Context-sensitive activity recommendations and feedback | Choices consider confirmed feelings, the clarified concern, goals, and constraints; users choose/rank options and report helpfulness after use. |
| 4. Discovery inside actions | Each action category combines saved activities with Find something else, refinement, and explicit saving of a chosen result. |
| 5. Journal lookup by date | Users ask Luna about their own dated writing without selecting an entry manually. |

Ranking and post-use feedback are included in item 3. Search refinement and saving
discovered activities are included in item 4. Purposeful reflection is the small
first delivery of the existing proactive-reflection direction, not an additional
duplicate feature. Inventory order does not authorize building all five at once;
finalize later delivery order after evaluating item 1 and outstanding current UX.
Custom NLP/model training remains the user's separate Colab research track.

## Item 1: purposeful reflection, first small slice

Agreed starting point after discussing the research: improve the usefulness of
Luna's reflection with a narrow, versioned prompt change. Briefly acknowledge the
event, then explore one relevant distinction or information gap when useful. Use
the person's stated facts; frame possible interpretations tentatively and leave
room for alternatives. Accept corrections. Direct requests get direct answers,
and stop requests end questioning. Just talk remains a controlling preference.

Example, based on the fictional cancellation case:

> You rearranged your day, then the cancellation disrupted it. What bothered you
> most: losing that time, having little notice, or something else?

This wording is our design example, not a quotation or a prescribed clinical
technique. More questioning is not automatically better, and one question per
turn is a usability constraint rather than an experimentally established optimum.
The flexible CBT-informed structure below remains a later design candidate; this
first slice does not require a complete CBT protocol or custom-trained model.

Compare the current prompt and revised prompt on five frozen fictional scenarios:
ordinary positive reflection, uncertain feelings, the cancellation correction,
an explicit direct-answer request, and a stop request. Record actual outputs and
prompt/model provenance. Review useful insight, grounding, correction acceptance,
unsupported assumptions, and questioning burden. Use controlled regressions for
contracts and separately bounded real-model/human review for language quality.
Do not call a valid schema an insight-quality pass.

## Item 2: optional emotion suggestions

Suggest a short set of possible emotions with plain-language reasons, allow the
user to confirm/remove/add alternatives, and optionally rate intensity. Keep
model suggestions distinct from confirmed self-report. Only confirmed emotions
should be treated as user-reported state; neither inferred labels nor intensity
are a diagnosis or a measured model-confidence value.

## Item 3: context-sensitive activity recommendations and feedback

Recommend activities using the confirmed feelings, the user's goal, constraints,
and preferences rather than the current first-item baseline. Show why an option
might fit and let the user choose, rank alternatives, or request something else.
Separate preference ranking before trying an activity from helpfulness feedback
after trying it. Do not imply that collected feedback already powers a trained
personalization model. The user owns custom NLP/model research in Colab; keep
future model integration separate from the product/prompt work.

Acceptance should compare shallow paraphrasing with useful grounded exploration,
check correction/stop/direct-answer behavior, and verify that emotion suggestions
are editable and optional. Activity ranking must demonstrably respond to
confirmed feelings and constraints without assuming one emotion has one correct
activity. Include mixed/uncertain feelings and rejection of Luna's labels.

## Item 4: find more activities where actions already live

Each of the five action categories should offer its saved/curated activities plus
"Find something else with Luna." Confirm the current category names during
implementation rather than introduce a second category taxonomy.

Use the existing Brave search and Luna selection/refinement backend. Display
recommendations within the action experience, with a short reason and source
link. Users can ask for something shorter, a different format, or better suited
to their needs. Refinement keeps the original goal and avoids rejected links.
Provide an explicit way to save a chosen discovered activity; opening a result
must not silently save it. Preserve the existing curated collection. The
standalone Discover page may remain for independent browsing.

Show an editable general search topic and obtain the required search/AI consent.
Do not send journal passages or conversation history to Brave. Continue to label
snippet-based evidence accurately; selection does not verify full-page claims.

## Item 5: ask Luna for journals by date

Users should be able to ask, for example, "Analyze my journal from October 5"
without selecting an entry in the journal screen first.

Give Luna a narrow, read-only journal lookup tool. The model proposes date
parameters; the server validates them, binds the request to the authenticated
user, and retrieves only entries in the requested date scope. The model receives
no database credentials and cannot select another user's identity or execute SQL.

Allow users to enable journal access for Luna. With that access and AI consent
enabled, the explicit dated request authorizes lookup without another per-entry
selection dialog. Keep the existing journal-selection route available as a
fallback. This slice does not introduce background reading of every journal or
automatic long-term memory.

Resolve dates using the user's configured timezone and saved timestamps. Ask a
short clarification when the date/year is ambiguous. If multiple entries match,
identify the matched entries and analyze the requested day's entries within
bounded entry/text limits; explain any limit rather than silently omit writing.
If none match, say so without guessing or substituting another date.

Show which dates and entries informed each response. Separate quotations and
recorded events from tentative interpretations; accept corrections. Journal
content remains untrusted data, never executable instructions. Preserve current
safety routing, provider-refusal handling, consent, usage limits, and provenance.
Source deletion during generation must invalidate the pending result, and later
turns must not keep supplying deleted source text to Luna.

## Evidence and later reflection structure

The user requested credible psychology/cognitive science sources and verified
quotations as a basis for this slice. See `PSYCHOLOGY_RESEARCH_SUMMARIES.md` for
browser-verified quotations, methods/results, and exact reading scope. The full
Braun et al. manuscript and relevant NICE sections were read; Cochrane is limited
to its summary/abstract and four other sources to indexed abstracts. These support
design hypotheses, not a validated Luna intervention. Keep those limits explicit
and complete the remaining full-text and direct conversational-agent review before
making stronger efficacy claims.

Requested during current-product evaluation: Luna should do more than acknowledge
and paraphrase. Explore a flexible CBT-informed reflection structure:
situation -> interpretation -> feelings/body response -> need or desired outcome
-> optional next step. This is a design candidate to evaluate, not a claim of
providing CBT treatment or a mandatory questionnaire for every conversation.

Luna can identify a useful missing distinction, offer a tentative hypothesis, or
ask one focused question. Use what the person already said rather than asking
every framework question in order. Direct-answer and stop requests override
question generation. Just talk can include exploratory questions, while action
invitations remain optional and respect the current preference. Do not frame an
ordinary emotion or reasonable concern as a cognitive distortion by default.

## Acceptance evidence to include in the implementation plan

- Activities: saved options and live discovery work in the same flow; refinement
  respects goal/format feedback; saving is explicit; private text stays out of
  search requests and URLs.
- Journal access: exact dates, relative dates, timezone boundaries, missing dates,
  ambiguous dates, multiple entries, and bounded results behave as described.
- Ownership and lifecycle: another user's entries remain inaccessible even with
  forged model tool arguments; disabled access makes no journal tool disclosure;
  source deletion and stale responses cannot restore retrieved writing.
- AI evaluation: feedback cites actual retrieved writing, distinguishes inference
  from fact, resists embedded instructions, and respects correction/stop requests.
- Use controlled provider responses for software tests and separately report
  bounded live-provider and human evaluations. A valid schema alone does not
  establish useful analysis.
