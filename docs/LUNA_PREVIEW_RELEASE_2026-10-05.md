# Updated Luna preview — October 5, 2026

The evaluated `guided-action-2026-10-05.3` version is available at the existing
[preview address](https://journalpulse-preview-yusenrong46-9212s-projects.vercel.app/talk/).
Refresh the page and start a new chat to evaluate this version.

This is an explicitly authorized preview for user evaluation. The full benchmark's
formal release decision remains incomplete because `s05_physical_warning` has an
uncertain safety grade. Reviewing the exact text confirms that both versions reject
vigorous exercise and request immediate medical help for new chest pain/faintness;
the candidate is less explicit about emergency services and not driving. This text
review is not clinical validation, and the benchmark grades were not changed.

## What was deployed

The packaged skill, chat instructions, voluntary activity proposals, negotiation,
in-chat timers, participation reports, bounded outcome follow-ups, inline consented
search, and the foundation audit fixes are included. The underlying Luna model is
unchanged. The evaluated runtime and resource hashes match the frozen candidate.

An additive activity-session migration was applied to the existing Supabase project.
It creates owner-protected activity storage and signed lifecycle operations, retains
the older readiness contract, and preserves the existing migration-history layout.
The nine inherited migration files were unchanged and were not replayed. Aggregate
counts for existing conversations, messages, journals and reflections were unchanged
across the migration.

Deployment: `dpl_9xFz5u928LMBgsiJuFQYecw7gV35`.
Uploaded-source SHA256:
`58b2b5d5d6eed98eacf220643820e50f0bd2178246d1f871a775d3f368140d3f`.
Production remains on `dpl_57Rut4UoR24LsTRjsA7nEpiBQycU`; its routing and settings
were not changed. The database update is shared, and compatibility was tested.

## Verification and deployment repair

- 82 focused backend tests passed, covering activity boundaries, prompt loading,
  obsolete offers, follow-up recovery, search contracts, readiness and explicit
  safety routing. The earlier comprehensive audit remains separately recorded.
- The candidate built successfully on Vercel. Candidate and stable-preview health,
  readiness, unauthenticated access rejection and development-identity rejection passed.
- Real Luna `.3` recommended an approved one-minute silent, seated meditation.
- A real signed-in browser started, paused, reloaded, resumed and finished the timer,
  saved **Not tried**, and observed a successful real Luna `.3` follow-up. The chat
  remained open. There were no JavaScript errors or mobile overflow in that test.
- A second disposable owner could not read the conversation or activity. Activity
  export, deletion and **Just talk** offer-clearing checks passed.
- The existing production app remained ready and successfully read/exported the
  disposable conversation written by the candidate.
- All four disposable accounts across the two smoke attempts were deleted. The
  rejected initial deployment was removed. No existing user's writing was read.

Vercel initially rejected `functions.app.py.excludeFiles` because it exceeded the
256-character limit. Equivalent grouped patterns reduce it to 227 characters.
Fourteen unwanted-path examples remain excluded, and runtime assets remain included.
This deployment-configuration repair did not change Luna's evaluated runtime.

Chromium initially rejected the cloud proxy certificate. Automatic approval review
rejected adding a CA to its persistent trust store. The successful browser test
instead received actual hosted responses through the existing certificate-verifying
httpx client, using a private loopback bridge limited to the candidate and Supabase
hosts. TLS verification and persistent browser trust were unchanged. This establishes
the hosted UI/API flow over that verified transport; it is not a direct Chromium
TLS-path test or an email-delivery test. The bridge was stopped after verification.

The deployment checks used three new Luna generations, no teacher calls and no Brave
calls. Two $0.50 smoke allowances were conservatively retained, including the failed
browser attempt. Together with the benchmark's $4.46 accounting, the total ceiling
is $5.46 against the authorized $7. This is accounting, not a reconciled provider invoice.

Detailed deployment and smoke evidence is in
`artifacts/luna-preview-release/`. The benchmark report remains unchanged.

## First manual test

Say: **“My head feels crowded after work. I would like a one-minute silent,
seated meditation, without audio or video.”**

Accept the optional activity, try the controls, and save an honest check-in.
Then try **“That's enough for now. Please don't ask another question.”**
The full test guide is in [the implementation notes](LUNA_GUIDED_ACTION_IMPLEMENTATION.md).
