# Connected Luna manual evaluation — October 4, 2026

Recorded from the user's pasted preview conversation. These findings describe
one fictional test conversation, not general model quality.

## Conversation findings

- **Grounding:** Luna followed the cancelled-plan details and treated the
  friend's intentions as uncertain.
- **Respecting reflection:** when asked to stop suggesting action and keep
  reflecting, Luna followed that request.
- **Accepting correction:** when “feeling unimportant” was corrected to
  frustration about rearranging the day, Luna shifted to the time disruption.
- **Practical requests need improvement:** a request for wording was initially
  answered with another confirmation question. A further explicit request
  produced a useful message. Track this transition for prompt refinement.

## JP-UI-001 — submitted message appears late in the chat

**Status:** reported by the user; source inspected; no fix or new deployment.

**Reported behavior:** after sending a question, a new Luna bubble appears below
the previous reply. After a delay, the submitted question appears between them.
It is not yet established whether that first bubble contains reply text or is
the typing indicator.

**Confirmed source behavior:** `web/app/talk/page.tsx` enables the Luna typing
indicator when sending starts, but appends the submitted user message and actual
assistant reply together only after the HTTP response succeeds. Thus the pending
question is absent from the transcript while Luna's indicator is already visible.
This is a likely explanation of the report; actual reply reordering has not been
reproduced independently.

**Expected behavior:** show the submitted question immediately in its own place,
follow it with Luna's typing indicator, then place the reply underneath. Clearly
distinguish pending/failed submission from confirmed saved messages.

**Verification for a future fix:** delay the API response with a test double and
assert the question is visible before the reply arrives. Check final ordering,
failed-send recovery, retry without duplicate bubbles, and late responses after
switching or closing a chat. No paid model call is needed for these UI checks.

## JP-UI-002 — journal workflow needs a clearer interface

**Status:** saved for a future redesign; no implementation or deployment change.

**User feedback:** “The UI is a bit confusing.” This was reported while testing
writing a journal entry, asking for a reflection, and continuing into chat. The
specific confusing controls have not yet been identified.

**Areas to review:** make saving, requesting a reflection, and discussing the
selected entry easy to distinguish. Explain what is saved and what is temporary,
including why the current reflection disappears on reload. Make the next step
and the journal entry used by chat clear without requiring guidance in this chat.
These are proposed review areas, not confirmed individual defects.

**Expected outcome:** a person can understand the save → reflect → discuss flow
and what happens to their writing at each step. Preserve explicit AI consent and
the clear separation between saved writing and temporary generated replies.
Any change to reflection retention needs an intentional product decision.

**Future verification:** walk through the redesigned flow with a first-time user
on mobile and desktop; check that they can save, reopen, reflect, and continue in
chat without assistance, and can explain which content remains after reload.

## Completed stopping and later checks

The user asked to stop without another question; Luna acknowledged and ended the
exchange. Later journal-connected turns kept the selected source, treated a
possible motive as tentative, and refused to infer the friend's intentions.
The small-step invitation remained before the user chose Just talk, which was
expected AUTO behavior. The user did not report a completed Just talk/reload
check in that manual sequence.

## Audit upgrade follow-up

JP-UI-001 and JP-UI-002 now have local implementation and regression evidence in
`/workspace/journalpulse-planning/audit-upgrade/`. This historical report preserves
the original observations; current deployed status and verification belong in
`AUDIT_UPGRADE_RELEASE.md`. The new versioned fictional evaluation protocol is
`LUNA_EVALUATION_V1.md`.
