# Test Connected Luna one slice at a time

The three features are implemented together, but acceptance proceeds in this
order. The combined deployment and its current provider status are recorded in
`CONNECTED_LUNA_PREVIEW.md`. Use fictional writing and a disposable account when
testing deletion.

## Before a live preview

1. Confirm the current release source and migration history. All three migrations
   `202610040001` through `202610040003` are already applied to the existing
   Supabase project; never edit or replay them. The audit upgrade adds only
   `202610040004_audit_integrity.sql`, whose applied status and checksum are
   recorded in `AUDIT_UPGRADE_RELEASE.md`. Scratch reset scripts are local only.
2. Configure preview-only Luna credentials/model and
   `JOURNALPULSE_LLM_ENABLED=true`, with `JOURNALPULSE_LLM_ZDR=true`.
   Confirm the configured model is actually available to the provider account.
   Use environment settings for credentials, never chat or source files.
3. Enable discovery only when ready for slice 3, using
   `JOURNALPULSE_SEARCH_ENABLED=true` and `JOURNALPULSE_SEARCH_API_KEY` for Brave.
   Keep a small, explicit live test budget. A discovery request makes at most one
   search and two model calls; a failed request can still incur provider charges.
4. Deploy a new protected preview with these exact changes and check `/ready`.
   Confirm its own Supabase redirect URL and public API settings. Keep the
   working Phase A preview available for comparison.

These are general release prerequisites. The current preview's completed steps
and provider verification are recorded in `CONNECTED_LUNA_PREVIEW.md`.
Local deterministic tests need neither provider credentials nor a paid budget.

## Slice 1: write, save, reflect

Open **Journal** and save: “My friend cancelled our plan. I felt disappointed,
but I said it was fine.” Save without asking Luna anything. Reload, reopen the
entry, and check that its wording and line breaks are preserved. Approve sending
this one entry to Luna, then choose **Reflect on this entry**.

Expect a reply grounded in the writing, tentative interpretations, and at most
one useful question. Reload: the saved entry remains, while this temporary
reflection is not stored. Try without consent and with AI disabled; writing
should still work and AI unavailability should be clearly reported. Do not
delete this entry until slice 2 is tested.

## Slice 2: discuss the selected entry

Choose **Discuss with Luna**, then **Use this entry in a new AI chat**. Check that
the screen identifies the selected entry. Ask: “What could I explore about why
I said it was fine?” Correct Luna if it assumes a motive, and check its response.
Reload and confirm that the same source remains selected. The chat text retention
setting still determines which temporary messages can be recovered.

Starting this discussion while an unrelated chat is open must require a clear
choice before replacing it. Try accessing a different account's entry: the
server must refuse access. Finally delete the fictional source entry and check
that its linked chat cannot continue. Unrelated entries and chats should remain.

## Slice 3: find resources, then refine

Open **Discover** and enter a general topic such as “Understanding overthinking.”
Approve sending that topic to Brave and Luna, then search. Inspect each source
link, the reason it was chosen, and the evidence label. This implementation uses
search snippets; Luna has not read the full pages or verified their claims.

Ask for “Something shorter and practical,” then choose **Find different sources**.
Check that the original goal is preserved, results change, and rejected sources
do not return. No journal content should appear automatically in the query.
Try an unavailable provider: the topic and feedback should remain so you can
retry. Click the sources yourself to assess whether they support Luna's reasons.

## What counts as acceptance

For each slice, record the exact input, prompt version, model, result, and any
problem. Judge grounding, respect for corrections, usefulness, and source fit.
Record latency and provider cost in live tests. A plausible reply or a passing
software test alone does not prove reflection quality or source reliability.

Automated evidence and its limits are in `CONNECTED_LUNA_EVIDENCE.md`. The
external review folder preserves the starting source, focused patches, test
logs, and mobile screenshots. These patches share contracts and are review
views of one implementation, not three independent deployments.
