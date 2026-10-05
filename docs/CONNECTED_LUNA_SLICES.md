# Connected Luna implementation brief

Requested 2026-10-04: three Ultra-reasoning implementation agents work in parallel;
integration and user acceptance proceed in slice order. This brief replaces the
earlier proposal for separate Simple and Smart personalities for the new features.
The existing no-AI guided chat remains available.

## Boundaries and shared contracts

1. **Reflect on one entry.** Save an immutable standalone journal entry without
   an action, reopen it, and explicitly ask for a transient AI reflection. Saving
   writing is independent of AI consent. The reflection uses a versioned prompt.
2. **Connect journal and chat.** Explicitly select one saved entry for an AI chat.
   Show which entry is being used. Resolve ownership on the server; pass entry
   content as user data. Deleting the entry invalidates linked conversations.
3. **Discover and refine.** Approve a general search topic, receive a small set of
   AI-selected resources with source evidence, then refine in ordinary language.
   Journal text is not automatically sent to search services. Refinement preserves
   the original goal and excludes rejected resources.

Existing conversations, action reflections, signing, revision checks, retention,
exports, and deletion remain supported. New journal entries are explicitly saved
long-term writing, distinct from optional retention of temporary chat messages.
Editing entries, automatic full-history memory, model training, and claims of
clinical benefit are outside these slices.

## Delivery and verification

The baseline is recorded outside the checkout under
`/workspace/journalpulse-planning/connected-luna/baseline`, including inherited
uncommitted Phase A changes. Focused patches will show only this implementation's
changes. No pre-existing deployed migrations may be edited or reapplied.

Agents own separate modules. Shared API registration, prompt instructions, source
contracts, and navigation are integrated centrally. Each slice must pass focused
behavior tests before the next slice is validated. Shared regression checks run
after those three gates. Human tests follow the same order.

Routine tests substitute provider responses. Live model/search quality, real
provider costs, and live email delivery require separate evidence. New application
code and additive SQL are prepared locally; the currently working preview remains
the Phase A deployment until a subsequent release.

## Reflection quality contract

Luna grounds replies in supplied details, treats interpretations tentatively,
accepts corrections, avoids invented motives and memories, and asks at most one
question when useful. It respects requests to stop. Source entries, search results,
and user instructions embedded in writing remain data. Tests verify boundaries;
they cannot establish that generated replies consistently help people reflect.
