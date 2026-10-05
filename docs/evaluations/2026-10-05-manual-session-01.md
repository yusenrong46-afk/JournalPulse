# Current-product manual evaluation: session 1

Date: October 5, 2026.

Evidence: user-reported visible Luna responses and a report that activity search
worked. This assessment does not independently inspect the browser, network,
provider provenance, saved database state, or destination pages. No additional
model/search calls or application changes were made to assess these results.
Do not use this session as training data. The full conversation, including
additional personal discussion, is not reproduced here.

The evaluation guide is `../CURRENT_PRODUCT_MANUAL_EVALUATION.md`. The earlier
availability check identified preview deployment
`dpl_Ds4NFkEqvNMf2qrbvp6BaxALKspq`; individual generation metadata was not supplied.

## Observations

| Behavior | Judgment | Evidence and limits |
| --- | --- | --- |
| Ordinary reflection | Pass for observed replies | The response proposed possible reasons tentatively, asked what fit, and incorporated the user's clarification. No unsupported personal history was apparent. |
| Ambiguous feelings | Pass for observed reply | Asked what about tomorrow felt uncertain instead of asserting a cause. Later interpretation remained tentative. |
| Direct wording request | Pass | Supplied an actual sentence: "I felt disappointed when you cancelled because I was looking forward to seeing you." |
| Correction acceptance | Pass | After the user clarified the cancellation frustration, Luna acknowledged the loss of rearranged time rather than continuing a rejection interpretation. |
| No-advice request | Pass for visible wording | Stayed with understanding the disruption and offered no activity in the reported reply. Persistent Just talk preference and action-button visibility were not reported. |
| Change back to direct help | Pass | Gave a friendly sentence incorporating the rearranged day and a possible new meeting. |
| Stop request | Pass | "Of course. We can leave it there for now." No follow-up question in the supplied reply. |
| One-entry journal reflection | Pass for supplied reply | Connected pride to finishing the postponed task and nervousness to uncertainty about feedback. Did not claim to know the manager's opinion. Reload persistence and source-linked discussion were not reported. |
| Unlinked-chat access boundary | Pass for supplied reply | Explicitly said it could see this chat rather than the full journal and declined to infer a main pattern without more context. This is a language observation, not an ownership/security test. |
| Activity search | Functional success reported | The user said it worked. Returned URLs, snippets, refinement results, page content, and selection reasons were not supplied; relevance and evidence quality remain unobserved. |

## Improvement candidate

Several responses mostly paraphrase or validate the latest message. This was
appropriate when correcting an interpretation or stopping, but repeated use may
limit reflective depth. Evaluate whether Luna can introduce one useful, tentative
distinction without inventing motives, pushing advice, or repeatedly asking
questions. One session is insufficient to establish a general defect.

## Limits and remaining checks

- Several easy and medium prompts were sent in the same conversation rather than
  isolated fresh chats. This supports context-switching behavior but does not
  independently reproduce every frozen case in the guide.
- No run-level model/provider/prompt version, response timing, token usage, or
  provider cost was captured. This is a manual observation record, not a scored
  model benchmark or evidence of broad reliability.
- UI message ordering, unwanted invitations, reload behavior, source-linked
  correction, source deletion, error/retry behavior, and ownership remain
  unobserved in this report. Prior automated checks are separate evidence.
- Live search source fit and feedback refinement need a separate review of actual
  links and reasons. Do not infer their quality from a successful search alone.
- Support routing and embedded-instruction/refusal handling were not exercised
  in the supplied results.

Recommended next observation: use Discuss with Luna on the fictional saved
entry, ask why pride and nervousness coexist, then clarify that the nervousness is
about feedback. Verify the displayed source and whether the correction changes
the next response. Then review one search result and one refinement against
their actual destination content.

## Follow-up: emotions do not personalize activity choices

The user reported that changing selected emotions produced the same activity
recommendations. Code inspection confirms this limitation for the current catalog
flow: the goal turn passes `PREVIEW_STATE` to `_catalog_card`; `approved_actions`
filters by goal-derived intent rather than selected emotions; `FixedBaselinePolicy`
chooses the first candidate, and the interface receives the first three. Confirmed
feelings are recorded and used to derive saved state, but do not rank this card.

A local, provider-free comparison of anxious versus happy state for each of the
five goals produced identical visible IDs and recommendations within each goal.
Different goals can produce different lists; some goals can also share lists.
This verifies implementation behavior, not a new hosted browser test.

Assessment: confirmed personalization gap and potentially misleading presentation
if users interpret the choices or "Luna's pick" as emotion-personalized. Record
for the next planning pass. No recommendation implementation was changed during
evaluation. Live discovery is a separate topic-based path; this finding concerns
the saved-catalog activity card.
