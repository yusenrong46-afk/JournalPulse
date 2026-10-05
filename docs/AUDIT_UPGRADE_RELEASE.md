# JournalPulse audit upgrade — 2026-10-04

The later [foundation audit](FOUNDATION_AUDIT_2026-10-05.md) records local repairs,
before/after measurements, and review diffs. It has not changed this hosted release.

Followup: [NLP_REFLECTION_FIX.md](NLP_REFLECTION_FIX.md) identifies the provider's
content-filter signal and corrects its handling. The stable preview now serves
that followup deployment. The release facts below remain the historical audit
checkpoint; the filtered case still produces no reflection.

This release starts from the source recorded behind the existing preview. All 123
previously uploaded file hashes matched the starting workspace and the live
deployment's recorded manifest. That is provenance reconciliation, not a download
of remote source files. Existing uncommitted work was preserved before editing.

The review directory is `/workspace/journalpulse-planning/audit-upgrade/`. It holds
the recoverable baseline, inherited tracked diff, four independent audit reports,
fresh review, regression logs, final numbered diffs and source-only archive.
Numbered diffs are review views of a compatible release, not separate deployments.

## Changes and manual evaluation

| Area | Problem and resulting behavior | Review view | Try in the preview |
| --- | --- | --- | --- |
| A: message delivery | The typing indicator appeared before submitted text. The question now appears immediately, followed by typing and the actual reply. Failed delivery stays visible. Exact retries preserve identity; edited requests reconcile the earlier receipt; returned messages appear once. Conversation switching suppresses stale responses and keeps drafts with their chat. | `01-reliability-review.patch` | Send a fictional question and check its position while waiting. Try a failed send, retry, edit and switch chats. A delivered pair should appear once. |
| B: journal clarity | Saved writing, temporary reflection and chat handoff now have separate explanations. Reflection consent resets on selection/reload. A failed reflection preserves the saved entry and says so. Chat shows the selected source, uses only its original writing after consent, and never silently attaches a temporary reflection. | `02-journal-review.patch` | Save a fictional entry, reflect after consenting, then reload. Writing remains; the temporary reflection and consent disappear. Discuss the entry and check its source label. |
| C: direct answers | Luna's prompt asks it to answer explicit wording/practical requests in the same reply. Strict output checks reject malformed output instead of fabricating a successful response. Sanitized rejection logs identify stages and field/type categories without logging text. | `03-luna-evaluation-review.patch` | Ask for one short message to a friend, correct an interpretation, then ask Luna to stop. Assess usefulness and grounding, not exact wording. |
| D: invitation timing | Message count and previous readiness no longer imply a fresh invitation. Current intent controls readiness; Just talk is saved and suppresses invitations after reload. Explicit current acceptance can enable an action. | `01-reliability-review.patch` and `03-luna-evaluation-review.patch` | Reflect without asking for an activity. Select Just talk, reload, and check it persists. Explicitly ask for a small step, then refuse it. |
| E: connected discovery | Chat and saved action cards now link to Discover. Only a fixed general goal enters its URL; the topic is editable and initially unapproved. Consent explains topic, feedback and derived search terms. Saved catalog links and live Brave results are labelled separately. | `04-discovery-review.patch` | Open Find resources, inspect/edit the topic, approve search, then request shorter text sources. Return to chat. Do not put private details in the topic or feedback. |

Other confirmed repairs bind request IDs to their intended payload/conversation,
protect acceptance ownership, retain recovery identities on transient reads, fix
IME Enter handling, and make authenticated cancellation stop before fetching and
during retry backoff/body reading. Additive migration 004 enforces signed usage
policy and database replay/acceptance checks. Legacy writers remain compatible.
Private journal/chat text is not hashed into persisted retry metadata.

## Executed software checks

| Check | Result |
| --- | --- |
| Final backend suite after diagnostic followup | 303 passed; 90.86% coverage; one warning |
| Frontend unit suite | 64 passed across 11 files |
| Ruff, mypy, frontend lint and TypeScript | Passed; mypy checked 22 source files |
| Generated OpenAPI contract and catalog | Current; 35 resources, 3 support items, zero catalog errors |
| Actual scratch PostgreSQL | 100 foundation, 17 journal and 19 integrity assertions passed |
| Real local PostgreSQL → PostgREST → API → browser integration | 13 passed, including commit-then-drop-both-responses recovery with one stored pair |
| Mobile/desktop browser and accessibility suite | 42 passed |
| Static export and fresh independent review | Passed; no remaining review blockers before hosted evaluation |
| Offline fictional evaluation preparation | 14 frozen cases; zero provider calls |

These counts describe different checks and must not be summed into a model-quality
score. Local integration uses explicit auth/provider doubles and system Chromium.
Earlier failing tests and fixture/locator corrections are retained in
`INTEGRATION_DIAGNOSIS.md`; assertion strength and application limits were preserved.
The diagnostic followup additionally has a recorded red-to-green test run.

## Hosted release evidence

The stable testing address remains
https://journalpulse-preview-yusenrong46-9212s-projects.vercel.app.
The alias now points to READY deployment `dpl_FPiUu75TNLc4oTuSwNc13vRwTuza`,
immutable host `journalpulse-c25xrun70-yusenrong46-9212s-projects.vercel.app`.
Its uploaded manifest SHA-256 is
`f6e79cc65e279e0f7da0ac0888d34854b48479e840b9bb19f0871db4163b7754`.
All 132 uploaded application file hashes still match the reviewed workspace.
Documentation/test-only final changes are excluded from Vercel's source upload.
The stable alias's readiness and journal page passed; an unauthenticated request
still receives the protection challenge. Production routing remains
`dpl_57Rut4UoR24LsTRjsA7nEpiBQycU`. Alias and source verification are in
`release/deployment.json`, `uploaded-files-sha256.json` and `alias-verified.json`.

Hosted verification passed ordinary reflection, exact writing/reload, separate
consent, account isolation, pending submitted text, selected journal context,
Just talk persistence, deletion of linked context, and discovery handoff/return.
Actual Brave search selected three snippet-based sources; refinement returned two
different URLs while keeping the original approved goal. No full pages were read.
Disposable accounts were removed after every attempt, including failed runs.

Actual usage was **15 model requests and two Brave requests**. Twelve model calls
returned accepted GPT-6 Luna/Azure output, valid schemas and no fallback: five
frozen chat inputs, three actual-history variants, one ordinary journal diagnostic,
and three discovery calls. All eight visible chat replies met their supplied
assistant-review criteria; readiness flags matched the expected application state.
A correction reply had a minor wording concern recorded in
`assistant-rubric-review.md`. Human acceptance and general quality remain pending.
Exact dollar cost was not measured. The first successful-study helper summary
retains a historical “six frozen inputs” label; actual observation metadata and
the review correctly distinguish the ordinary journal probe from that fixture.

**Unresolved: embedded-instruction journal reliability.** Three attempts on the
same fictional adversarial entry returned HTTP 502; zero accepted reflections.
The first showed a format/schema error. Later generic journal errors leave their
exact stage unknown. Saved writing remained intact and invalid output was not
shown. Raw provider output was unavailable, so instruction-following/grounding
criteria cannot be scored. Safe logger tests pass, but captured runtime events
did not yield the exact rejection diagnostic; historical dashboard retrieval was
unavailable through the scoped API. This is a failure to investigate before any
production promotion, not a prompt-injection pass. The provider-study budget was
expanded from 12 to 14, then 15 for explicit diagnostic attempts; no further paid
requests are part of this release.

Migration `202610040004_audit_integrity.sql` was applied once in a bounded
transaction to the existing Supabase project. Its SHA-256 is
`ac08f65784ef09628c09108818ebf2cfd6df19977c496534bc93510cd83d0535`.
All four 2026-10-04 migrations are now applied and immutable. Never replay them or
use the hosted project with scratch verification/reset helpers. No existing user
records were rewritten by 004; disposable hosted QA users own their fictional data.
No production promotion, repository push, merge or new hosted project occurred.

## Evaluation boundaries and prioritized followup

The fictional evaluation set is hand-authored evaluation data, not training data.
Application observations include provenance and visible replies, not raw provider
JSON. Actual-history variants are labelled separately from frozen inputs. Rubric
review by an assistant is distinct from human acceptance and clinical usefulness.
Search selection uses Brave snippets; full pages, source claims and suitability
have not been verified. Empty results can be an honest outcome.

1. Diagnose the repeated embedded-instruction reflection failure with sanitized
   rejection-stage evidence, then verify a bounded repair without relaxing output
   checks. Keep it visible as a preproduction gate. Complete human acceptance of
   the five changed flows, especially journal clarity
   and whether direct replies feel useful rather than repetitive.
2. Verify privacy-sheet/menu keyboard focus, Escape behavior and focus restoration;
   automated axe checks do not establish those interactions.
3. Broaden fictional multi-turn and adversarial evaluation, including irrelevant
   results, snippet injection, long conversations and source relevance.
4. Test actual email login/refresh and service-worker upgrade/offline behavior;
   hosted QA blocks service workers and uses real password-issued test sessions.
5. Verify managed backup/restore and profile long-running lock-map growth, HTTP
   client lifecycle, and export consistency during concurrent writes. Legacy
   message receipts lack the new input metadata and have narrower replay checks.

These are recorded limits or investigation tasks, not newly established production
defects. This audit does not establish that every hidden bug has been found.

Reusable cloud startup instructions were saved to the existing configuration draft;
install script, secure bindings and network settings were preserved. Publishing
that draft is separate from this completed Vercel preview update.
