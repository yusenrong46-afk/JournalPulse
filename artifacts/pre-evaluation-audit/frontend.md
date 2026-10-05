# Frontend audit and targeted repairs — 2026-10-05

Scope: Talk, journal, discovery, inline activity, legacy check-in, Journey, Me,
authentication, pending message ordering, account switches, timers, consent,
recovery, keyboard behavior and mobile accessibility. Work used the inherited
dirty checkout. No paid calls, external writes, deployment, commit or reset.
Read `web/AGENTS.md` and bundled Next 16.3.8 guides before frontend edits.

## Confirmed defects and repairs

### P1 — Inline search posted the wrong request shape

- Location: `web/components/activity-session-discovery.tsx:62`;
  API contract: `src/journalpulse/inline_discovery.py:39`.
- Reproduction: open inline activity search from a real AI conversation, grant
  consent and search. The UI spread `activity_constraints` into the request root.
  API responses serialize the constraint defaults too, so even unconstrained
  chats sent forbidden top-level fields. Strict validation returned HTTP 422
  before any discovery call.
- Independent API proof is in `resource-safety.md`: original UI-shaped request
  returns 422 and zero provider calls; nested request returns 200 with an offer.
- Fixed: send `{constraints: activityChat.activity_constraints}`. The new
  `activity-discovery.test.tsx` regression was run before the fix and failed with
  all five constraint fields at the root; it passes afterward.
- Added real-stack browser coverage in `guided-action.spec.ts:153`: consented
  search, exact API shape, refinement excluding prior links, signed offer save,
  activity start and persisted source identity. Root owns execution of this test.

### P2 — Inline refinement advertised invalid syntax and permitted oversized input

- Location: `web/components/activity-session-discovery.tsx:111`.
- Reproduction: use the former example `shorter, seated, no video`; the API's
  strict general-topic vocabulary rejects commas. The textarea also allowed 600
  characters while `InlineDiscoveryRequest.feedback` allows 160.
- Fixed: example is `shorter seated no video`, maximum length 160. The new
  real-stack case submits the actual placeholder example through the real API.

### P2 — Check-in retries silently accepted older answers

- Location: `web/app/check-in/page.tsx:69`.
- Reproduction (executed with a commit/lost-response test double): choose
  helpfulness 1, submit, lose the response after server commit, choose 5, retry.
  The old UI reused the first request UUID with a new payload, ignored the
  replayed outcome and displayed “Thank you!” while the stored rating remained 1.
- Server evidence: SQLite `save_outcome` returns the original record for the same
  owner/decision/UUID; PostgreSQL `save_outcome_record` has matching replay
  semantics and permits one outcome per decision. A fresh UUID is not an update.
- Fixed: preserve the complete submitted JSON, including elapsed time, across
  retries; disable answer controls while saving and during ambiguous recovery;
  explain that submitted answers are held unchanged and offer “Retry saving
  check-in.” A definitive 400/422 rejection leaves answers editable and starts a
  fresh receipt when corrected. Existing route keys still reset forms/receipts
  when decision identity changes.
- Added tests for ambiguous-save exact replay/locked inputs and editable
  validation failure. The pre-fix scratch reproduction ran successfully at
  18:23 UTC, demonstrating the unwanted behavior; it was outside the checkout.

### P2 — Chat input exceeded the API limit and validation errors were unreadable

- Locations: `web/app/talk/page.tsx:1140`, `web/lib/api.ts:80`;
  server `ConversationTurnRequest.text` maximum is 2,000 characters.
- The former composer had no input maximum. FastAPI's validation array was
  passed directly to the `Error` constructor, producing `[object Object]` rather
  than an actionable reason.
- Fixed: composer maximum is 2,000. Validation formatting shows up to three
  bounded messages with friendly field labels where known. It never serializes
  the submitted `input` or validation `ctx`; unexpected objects receive a clear
  fallback. String details and formatted output are bounded to 600 characters.
- Added API tests for a private-input validation response, malformed objects and
  bounds. Browser paste/typing test confirms a 2,001-character paste is bounded
  and the actual submitted message is 2,000, on desktop and mobile.

### P1/P2 shared backend finding — An obsolete unstarted session hid a new offer

- UI implication: `activity-session-workspace.tsx:273` correctly treats only
  completed/stopped/declined sessions as terminal; a stale `offered` session
  therefore suppresses a newer Luna recommendation indefinitely.
- Backend reviewer reproduced a two-minute offer surviving a new one-minute
  constraint and still starting the old 120-second resource. Backend reviewer
  owns SQLite/PostgreSQL invalidation repair and tests.
- Added real-stack UI case in `guided-action.spec.ts:207`: save a signed search
  offer without starting, ask Luna for a different quiet activity, verify the old
  session becomes stopped/unstarted, then start the new recommendation. No
  additional frontend state workaround was introduced.

## Verification completed by frontend reviewer

- Baseline unit suite: **103/103**, 19 files.
- After repairs: **108/108**, 20 files, `npm run test:unit`.
- TypeScript: **pass**, `npm run typecheck`.
- ESLint over all touched frontend modules/tests: **pass**.
- Automated accessibility: **18/18** (nine routes × desktop/mobile), Chromium,
  all API calls mocked. Existing checks assert no serious/critical axe findings;
  this does not establish full WCAG conformance or a populated-state visual review.
- New composer browser regression: **2/2**, desktop/mobile, mocked API.
- Existing tests cover immediate pending-user-message ordering before Luna's
  indicator, no duplicate receipt recovery, delayed responses after route/account
  switches, original-draft preservation, explicit source selection, unsupported
  AI fallback, consent boundaries, modal/menu keyboard focus, authoritative
  activity clocks and paused timer recovery. All remain passing.
- Two real-stack integration cases added but not launched here; root runs the
  serialized shared PostgreSQL/PostgREST/API/browser suite after all repairs.
- The reviewer-owned Next dev server was stopped after browser checks.

## Product gaps and evaluation caveats (not silently redesigned)

1. **AI activity outcomes do not populate Home/Journey.**
   `web/app/page.tsx:56` and `web/app/journey/page.tsx:32` read legacy
   reflections/outcomes only. New inline reports live in `activity_sessions`;
   the existing activity integration test explicitly expects legacy `outcomes`
   to remain empty. A user can complete an AI activity and still see an empty
   garden. The vertical-slice plan promises inline reporting and preserved
   legacy flows, but does not explicitly promise a merged Journey view. Treat
   this as product continuity/scope work, not evidence of lost data. Evaluate
   the current inline loop through chat; design an activity history/read model
   before promising all activity reports appear in the garden.

2. **Me-page descriptions were stale; corrected before evaluation.**
   `web/app/me/page.tsx:117` now describes checks for certain clear crisis phrases
   and acknowledges they can miss distress. The resource explanation now
   distinguishes built-in/reviewed resources from consented Brave snippet-only
   results, and AI choices from Simple mode's fixed rule. Root authorized this
   follow-up copy correction; no new UX or copy-only tests were added. ESLint
   passed for the touched page.

3. **Journal reflection-to-chat remains a deliberate context boundary with UX
   cost.** `web/app/journal/page.tsx:132` separates a temporary one-off reflection
   from the original saved entry and tells users the reflection does not travel
   into chat. Discuss requires a separate source-choice screen. The inherited
   clarity work is present and its consent/source tests pass, but the original
   reported confusion needs first-user mobile/desktop observation. Do not infer
   that source isolation tests prove people understand these two modes or that
   temporary reflection retention should change automatically.

4. **Multiple discovery entry points serve different purposes.** Talk offers a
   standalone “Find resources” link, an offer's standalone “Search for other
   resources” link and an inline “Find another resource” control. Standalone
   discovery has no save-to-current-activity handoff; inline discovery does.
   This is a possible navigation/expectation problem for manual evaluation,
   separate from the repaired request contract.

## Files changed by this reviewer

- `web/components/activity-session-discovery.tsx`
- `web/app/talk/page.tsx` (composer maximum only)
- `web/app/check-in/page.tsx`
- `web/app/me/page.tsx` (follow-up product-copy correction)
- `web/lib/api.ts`
- `web/tests/unit/activity-discovery.test.tsx` (new)
- `web/tests/unit/api.test.ts`
- `web/tests/unit/check-in-workspace.test.tsx`
- `web/tests/e2e/core-flow.spec.ts`
- `web/tests/integration/guided-action.spec.ts`

Other inherited changes were preserved.
