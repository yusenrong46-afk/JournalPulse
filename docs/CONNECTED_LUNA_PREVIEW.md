# Connected Luna combined preview — 2026-10-04

Latest followup: [NLP_REFLECTION_FIX.md](NLP_REFLECTION_FIX.md). The same stable
preview now serves `dpl_Ds4NFkEqvNMf2qrbvp6BaxALKspq` with accurate provider-
decline handling. Earlier deployment references below are historical checkpoints.

The current audit upgrade is documented in [AUDIT_UPGRADE_RELEASE.md](AUDIT_UPGRADE_RELEASE.md).
The stable address below now serves `dpl_FPiUu75TNLc4oTuSwNc13vRwTuza`.
The remaining sections retain the earlier combined-release checkpoint and its
historical evidence; its “current” status and deployment IDs describe that baseline.
Consult the audit report for new checks and the unresolved adversarial reflection.

Open https://journalpulse-preview-yusenrong46-9212s-projects.vercel.app

Use your Vercel account if deployment protection prompts you, then sign in to
JournalPulse. This is the existing `journalpulse` project with all three slices
deployed together. The production app keeps its earlier deployment.

## Current status

- **Journal reflection is live.** Save fictional writing, reopen it, approve AI,
  and choose Reflect on this entry. A real GPT-6 Luna reflection passed.
- **Journal-connected chat is live.** Choose Discuss with Luna and explicitly
  use the saved entry. A real reply, reload, visible source and source deletion
  invalidating the linked chat passed.
- **Discovery and refinement are live.** Approve a general topic, search, and
  give feedback about what would fit better. Real Brave retrieval and Luna
  selection returned three sources, followed by different sources after feedback.
  The original goal was preserved and earlier sources excluded. The Brave key
  is stored as a sensitive Preview-only variable.

Evaluate these in the order above. Keep the entry after slice 1 so you can use it
in slice 2. Use fictional information while testing against the existing shared
Supabase project. See `CONNECTED_LUNA_TEST_GUIDE.md` for detailed checks.

## Release and hosted evidence

- Final deployment: `dpl_Cn8ThAWTqRAnsV6y3oak1qMY5SMx`, READY, preview target.
- Immutable deployment URL:
  `journalpulse-lyapp3hts-yusenrong46-9212s-projects.vercel.app`.
- Stable preview alias points to this deployment. Vercel refused assignment of
  the old generated deployment URL with `alias_in_use`; that earlier address
  still identifies the Phase A deployment. Future updates can move the stable alias.
- Uploaded 123 source files; candidate manifest SHA256:
  `bdc29a70b9c40e7baa5bcb3b7cde09d5d5aa125690df21fb393eb5c51cfa29b9`.
- `202610040003_journal_entries.sql` committed to the existing Supabase project
  in one transaction. Its RLS and source/deletion guards were verified. This
  migration is now applied and must not be edited or replayed on that project.
- Earlier Phase A migrations were preserved. Existing record counts stayed at
  2 conversations, 2 reflections and 18 messages before and after verification;
  no standalone test writing remained after cleanup.
- Both preview hosts and production `/ready` passed. Production routing still
  points to `dpl_57Rut4UoR24LsTRjsA7nEpiBQycU`. A disposable guided conversation
  using the older production code worked against the additive database schema.
- Preview Luna is enabled with existing credentials and ZDR routing requested.
  Provider retries are limited to one attempt in Preview. The function duration
  is 120 seconds to accommodate the discovery/refinement budget. Preview Brave
  search is enabled; production settings remain preserved.
- Two real Luna requests used model `openai/gpt-6-luna`, provider reported as
  Azure, valid schemas and no fallback. Recorded model latency was 2258 ms for
  reflection and 2940 ms for chat. Token counts are retained in the evidence;
  exact provider charges and broader reply quality were not evaluated.
- Discovery was verified with one live search and one live feedback refinement,
  then rechecked after correcting encoded punctuation in provider snippets. Each
  initial search used one Brave call and one model call; each refinement used
  one Brave call and two model calls. Total live verification used 8 model
  requests and 4 search requests, with exact provider charges unmeasured.
- The first two slices were validated on provider-enabled deployment
  `dpl_C2sACmhYRSW24cLnMH8pjTDwoXEX`. The final source differs only in
  `discovery.py`, which now decodes title/snippet text once while preserving
  source URLs. The affected discovery flow passed again on the final deployment.
  Three new normalization regression cases were added; the full backend suite
  passed **237 tests, 90.75% coverage**, with Ruff and mypy also passing.
- Real hosted browser checks covered exact writing, reload, account isolation,
  separate AI consent, transient reflection, selected context and deletion.
  Owned disposable Supabase accounts were cleaned up on every attempt. The
  final discovery run removed its one test account. Credentials remained in memory.
- Exact preview hosts and their own return paths were added to the Supabase
  sign-in allowlist; the production Site URL was preserved. Actual email delivery
  was not tested. Browser service workers were blocked during verification.

The first verification helper assumed the wrong shape for a conversation-create
response; that helper was corrected. Chromium then needed initialization writes
to its existing NSS certificate database. Scoped access fixed verification, with
TLS checks preserved. Failure logs and final passing evidence are retained in
`/workspace/journalpulse-planning/connected-luna/release/`.

Managed backup/restore was not verified. Replaced function definitions were
captured for operator reference, which is not a full database backup. This release
added schema and replaced compatible functions without rewriting existing records.
Live Brave retrieval/refinement and readable punctuation are verified. Source
claims, full pages and broad reflection quality have not been evaluated.
