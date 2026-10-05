# Luna guided action: implementation and review guide

Updated October 5, 2026. The slice is implemented in this workspace. The original
candidate failed its delivery gate when a walking-search phrase was rejected.
That contract mismatch is now fixed; software checks pass, but the fresh formal
evaluation is incomplete under evaluator rate limiting. Browser and supplemental
agent checks are recorded separately. The existing Preview remains unchanged and the new migration has not
been applied to the shared database. See the [search-contract fix](LUNA_SEARCH_CONTRACT_FIX.md),
original [release evidence](LUNA_GUIDED_ACTION_RELEASE.md) and comparison report.

The subsequent local audit advances the current runtime to `.3`. Outcome requests
include the actual selected activity, saved goal, and configured duration as
bounded user data; an empty current feeling list clears prior inferred labels;
unknown catalog audio/video capabilities no longer satisfy hard preferences;
and bare unsafe-feeling statements reach the existing safety route. Local
synthetic regressions cover these corrections. The retained live model comparison
describes `.2`; it does not validate the changed `.3` prompt/context assembly.

## What changed

Private AI chat can now lead into an activity without ending the conversation.
Luna can reflect, ask one useful clarification, or propose an optional activity
when the stated goal and circumstances provide enough information. The user can
negotiate in ordinary language, decline, choose **Just talk**, or stop.

An approved proposal appears inside the chat. **Start activity** starts it;
receiving a suggestion does not. The silent meditation has reviewed local
instructions and real one- and two-minute options. Other app activities include
comfortable movement, reflection, a small task, and a connection step. Catalog
links use suitable external/manual controls instead of pretending a web page is
an in-app timed exercise. The existing guided/no-AI and legacy acceptance paths
remain available.

The server owns the activity state and timer deadline. The browser repaints the
remaining time and recovers state after refresh or returning to the tab. Pause,
resume, finish early, and stop have explicit commands. Expiry presents one
**Did you try it?** check-in; it does not infer participation or emotional change.

Reports distinguish completed, partly tried, not tried, and stopped. Optional
fields separately record activity fit, reported state change, and goal progress.
The report is saved before one bounded Luna follow-up. A failed follow-up can be
retried without erasing the report or creating another outcome. Luna should accept
unchanged, worse, or uncertain results and offer another activity only when welcome.

## Resources, search, and journals

The server supplies a bounded pool of approved resource IDs. It checks the chosen
resource against the latest duration and format constraints, including user
corrections. Luna cannot invent a resource URL or turn an arbitrary ID into an activity.

**Find another resource** provides inline Brave search with explicit search
consent. Queries and refinements use bounded general activity words; chat history,
journal text, names, and identifiers are not forwarded to Brave. This narrow
vocabulary can reject uncommon wording. Results expose snippet-only evidence:
full pages were not read, and unknown duration or accessibility cannot satisfy a
confirmed hard constraint. Empty or unavailable search stays an explicit result.
Opening a link does not save it. **Save this activity** uses a signed, short-lived
offer tied to the owner, conversation, and current revision.

A journal-linked chat uses only the explicitly selected owner's entry, with
visible source/date context. Older writing does not establish the person's current
state; their correction takes priority. Standalone journal reflection remains
reflection only. Deleting a source invalidates source-dependent pending work.
To continue without that entry, start a new unlinked chat; this slice does not
silently detach its history or search all journals. Automatic lookup by date,
long-term pattern retrieval, and custom NLP training remain later work.

## Instructions, research, and limits

The canonical runtime file is
[`src/journalpulse/skills/guided_action/SKILL.md`](../src/journalpulse/skills/guided_action/SKILL.md).
It is included in the Python wheel and loaded through trusted package resources.
Model-run metadata records its version and content hash; a Markdown file in
excluded documentation alone would have no runtime effect.

- Current version: `guided-action-2026-10-05.3`.
- Current SHA256: `abb73f69089a8591e5f30f1ac62e4557cd0a62f45d408d8fdf718f1bd4808270`.
- Research reviews: [reflection evidence](research/REFLECTION_EVIDENCE_2026-10-05.md)
  and [chat–action evidence](research/CHAT_ACTION_EVIDENCE_2026-10-05.md).

Guided discovery motivates relevant questions rather than endless questioning.
Motivational interviewing and autonomy support motivate correctable, voluntary
suggestions. Adaptive-intervention research motivates checking both need and
receptivity. Activity research motivates plausible options and honest feedback,
with active-control and null findings retained in the evidence. None proves
clinical efficacy of Luna, this skill, a selected link, or a particular activity.

The existing conservative safety router remains authoritative and can bypass
normal Luna generation. Journal content, snippets, and participant notes remain
untrusted data rather than system instructions. Owner checks, consent, retention,
revision checks, and idempotent receipts apply to the new loop. Ordinary chat
retains its 20-user-message limit; a started activity can still receive a saved
check-in and at most one final successful follow-up at that limit. LLM and user
selections are excluded from off-policy evaluation and have no invented selection
probability. Research memory and adaptive-policy flags remain off.

## Easiest manual review

These steps apply to the updated preview, deployed October 5, 2026 with its activity
migration and Luna configured. See [release evidence](LUNA_PREVIEW_RELEASE_2026-10-05.md).
Use a disposable test
account and fictional writing. Begin a new private AI chat
with AI consent enabled. These are behavioral checks, not exact reply templates.

1. Say: **“My head is noisy after work. I have two minutes for a quiet pause,
   no audio, and want to feel more settled.”** Look for a fitting silent activity
   in the same chat, without a required emotion/goal questionnaire.
2. Before starting, say: **“Can we make that one minute instead?”** Check that
   the new recommendation and duration really change; there is a one-minute
   meditation resource rather than invented timing.
3. Press **Start activity**. Pause, resume, and refresh. Confirm the timer retains
   its state, the original chat remains open, and no audio starts automatically.
   Finish early or let it end. Confirm one inline participation check appears.
4. Choose **Not tried** and save the check-in. Luna should not congratulate you
   for completing it or assume you feel calmer. The saved-report status should
   remain visible even if generation fails.
5. In a separate run, actually try the activity and report honestly. Choose
   **About the same** for state change if nothing changed. Luna should accept that
   result without turning it into improvement or automatically starting another
   activity.
6. Say: **“That's enough for now. Please don't ask another question.”** Check for
   a brief ending without another question, recommendation, or automatic search.
   Separately check **Just talk** suppresses offers; **Find a small step** restores
   the activity preference when you want it.
7. In a fresh chat with no restrictive duration/audio requirements, open
   **Find another resource**, permit the general search, and try **“quiet
   meditation”**. Open a link without saving it, then explicitly save a returned
   activity. Try general refinement such as **“shorter text”**. Missing or unsuitable
   results should be explained rather than fabricated.
8. Save a fictional journal entry, explicitly use it in an AI chat, and correct
   its past mood: **“That was yesterday; I feel calm now.”** Check that Luna uses
   the selected entry without treating it as the present state or claiming access
   to all journals. Start a new unlinked chat and verify the entry is not presented
   as its source.

## Where the code lives

| File or module | Simple purpose |
| --- | --- |
| `src/journalpulse/guided_action.py` | Loads the trusted skill; defines the bounded decision and context schemas. |
| `src/journalpulse/intelligence.py` | Builds the actual model request, validates its response, and records model/skill provenance. |
| `src/journalpulse/activity_resources.py` | Defines approved activities, checks constraints, validates public search words, and signs/verifies discovered offers. |
| `src/journalpulse/activity_chat.py` | Converts a validated Luna choice into a chat card and builds the outcome follow-up. |
| `src/journalpulse/conversations.py` | Keeps normal chat, preferences, linked sources, safety, and revisions coordinated. |
| `src/journalpulse/activity_models.py` | Defines sessions, commands, reports, and selection provenance. |
| `src/journalpulse/activity_lifecycle.py` | Applies timer/state transitions without assuming the user participated. |
| `src/journalpulse/activity_sessions.py` | Exposes owner-bound activity APIs and separates saving a report from generating its reply. |
| `src/journalpulse/inline_discovery.py` | Connects consented, validated general searches to signed activity offers. |
| `src/journalpulse/persistence.py` | Stores activities and receipts; enforces ownership, revisions, retention, export, and deletion. |
| `supabase/migrations/202610050001_guided_activity_sessions.sql` | Adds hosted activity storage and trusted RPCs while preserving older contracts. |
| `web/components/activity-session-workspace.tsx` | Coordinates the activity inside the current chat and rejects stale results. |
| `web/components/activity-session-panel.tsx` | Shows instructions, timer controls, and the honest participation check-in. |
| `web/components/activity-session-discovery.tsx` | Shows search consent, general refinement, source links, and explicit saving. |
| `web/lib/activity-session.ts` | Shares browser activity contracts, API calls, and display-clock calculations. |
| `web/app/talk/page.tsx` | Places the loop alongside messages and existing journal/preference controls. |

Code comments explain the non-obvious boundaries where they are enforced:
server-owned timing, untrusted prompt data, signed ownership, retries, stale
responses, retention, and the difference between timer completion and a report.

## Verification recorded so far

The original implementation run recorded **609 backend tests passing**, with
**90.59% coverage with branch measurement**, **103 frontend unit tests**, and **16 real-stack
integration tests**. An earlier run recorded **44 desktop/mobile browser
regression tests**; that earlier run is not substituted for final candidate
browser verification. The actual built wheel was checked for the packaged skill
and its loader/hash. Coverage measures executed code, not empathy, clinical safety,
or language quality.

The original before/after benchmark produced a standalone report. There are 40
controlled teacher judgments and seven of eight adaptive judgments; the final
adaptive judgment exhausted its bounded retries under provider rate limiting.
One held-out walking-search proposal was rejected by the application's public
topic validator, so the release gates failed. The report retains that failure and
the missing judgment. The existing Preview and shared database remain unchanged.
See [release evidence](LUNA_GUIDED_ACTION_RELEASE.md) for the exact results and
remaining work. Software-test counts alone do not establish a benchmark win.

The subsequent search-contract fix passes **617 backend tests** with **90.67%
coverage with branch measurement**, all **16 real-stack browser integrations**,
static/OpenAPI checks and wheel validation. Its fresh 60-case report has 16
held-out judgments, including 15 of 20 expected quality pairs; evaluator rate
limits leave the release gate incomplete. Four separate post-hoc paired
follow-ups have eight actual Luna replies and no teacher scores. See the
[search-contract fix](LUNA_SEARCH_CONTRACT_FIX.md) for current results, costs,
browser evidence and review artifacts. The original failed report is preserved.
