# Architecture

## Request flow

```text
Next.js PWA ─► request guard ─► auth ─► safety gate ─┬─ support ─► fixed message + human resources
                                                      │
                                                      └─ normal ─┬─ AI allowed ─► Luna on OpenRouter
                                                                 └─ no AI ──────► scripted Luna
                                                                        │
                                            chosen goal ─► reviewed catalog ─► fixed baseline pick
                                                                        │
                                                   person's choice ─► saved reflection ─► check-in
```

Chat follows this loop. Standalone journals and discovery are described in
[CONNECTED_LUNA_ARCHITECTURE.md](CONNECTED_LUNA_ARCHITECTURE.md). The server owns the parts that must be trusted:
the safety check, the model call, the catalog, the policy, and what gets saved. The browser owns the
gentle prompts in between (the feelings and goal buttons) and sends only their result.

## Chat stages

| Stage | Where it lives | What happens |
|---|---|---|
| Welcome | Browser | Greeting and mood faces. The first tap or message creates the conversation; a mood tap sends its `mood_score` with that message |
| Chat | Server | Each message runs the safety check, then Luna replies. The reply can mark the chat `ready_for_action` and suggest up to three `feelings` from a fixed list |
| Feelings | Browser | Luna's suggestions are preselected the first time; after that, the person's own confirmed set is |
| Goal | Browser → server | The chosen goal is sent as an ordinary message with `goal` and `confirmed_feelings`. The server stores what the person confirmed and builds the card from the catalog without a model call |
| Offer | Server data, browser view | Three catalog actions; the fixed baseline's pick is marked "Luna's pick" |
| Saved | Server | `POST /accept` writes a `ReflectionRecord` in one transaction with the chat's close |
| Check-in | Server | `POST /v1/outcomes` records whether it was tried, how much it helped, and optional feelings and a note |
| Support | Server | Any message that trips the safety check moves the chat into support mode for good. The model is never called again in that chat, and every later reply is the support message recorded on entry |

The goal message goes through the safety gate like any other message, so typing a crisis message into a
goal turn still routes to support.

### Smart and Simple Luna

A conversation records its `mode` when it starts:

- `ai`: the person allowed AI and a model is configured. Replies come from
  `OpenRouterConversationClient` (`JOURNALPULSE_CHAT_MODEL`, default `openai/gpt-6-luna`) with a strict
  JSON schema, zero-data-retention routing, and bounded retries. There is no canned reply when the
  provider fails; nothing is saved for that turn.
- `guided`: no AI. `journalpulse.guided` returns fixed questions and guesses feelings by keyword.
  Current-turn intent and the saved interaction preference govern activity invitations.
  It never copies the person's words into the summary.

When a record is saved, its `model_run` names the model that actually held the conversation, not the
local goal step.

## Browser account boundaries

Authenticated workspaces remount when their owner changes. Preferences, resume IDs,
and reminders use account-specific browser keys; only actual boolean values enable
AI consent or retention. Legacy shared-device consent is not assigned to an account.
API requests validate the resolved token's owner and capture an account revision,
so switching away and back cannot revive an obsolete request or retry.
Backend RLS and ownership checks remain authoritative. Client cancellation does
not undo a transaction already committed by the server.

## Conversation lifecycle

A model reply can take tens of seconds. While it runs, the same chat can be closed, deleted, accepted,
swept, or answered by another server instance. Correctness therefore lives in the database, not in
process memory.

Every conversation row has a `revision`. The API reads the chat (and its revision), calls the model with
no transaction open, then commits with `jp_commit_turn`, which locks the row and stores the turn only if:

- the row still exists and belongs to the caller,
- it is still `open`, and
- its revision still equals the one the turn started from.

Otherwise nothing is written and the API answers 409 (closed or changed) or 404 (deleted). A retry with
the same `client_message_id` returns the stored turn instead. Close, accept, the retention sweep, and
every commit increase the revision, so a reply computed before any of them can never reopen a chat,
overwrite newer state, or restore text that was already cleared. The in-process lock that answers "a
reply is already in progress" is kept only as a fast path; correctness does not depend on it.

`jp_accept_conversation` is one transaction: check the revision, write the reflection bundle
(reflection, observation, decision, model run, safety event), link it, close the chat, and clear its
text unless the person chose to keep it. A unique index allows one reflection per conversation. The same
request ID returns the saved record; a different one gets 409. Any failure rolls the whole step back.

`jp_delete_conversation` removes the chat, its messages, and its linked reflection in one transaction.

### Persistent conversation choice

`interaction_preference` is separate from AI/guided `mode`: `auto` preserves the existing flow,
`listen` means Just talk, and `act` follows an explicit Find a small step request. Older JSON records
without this field default to `auto`.

`POST /v1/conversations/{id}/preference` takes `client_request_id`, `expected_revision`, and
`preference` (`listen` or `act`). The repository checks ownership, receipts, lifecycle, and revision
inside one transaction. It clears the card, sets readiness for the new choice, advances the revision,
and stores a small command receipt. It does not call a model or save a reflection. Receipts last for
the conversation's lifetime, contain no journal text, and are exported and deleted with it.

A duplicate command returns **current** state; replaying Listen cannot undo a later Act choice.
Reusing its ID with another preference or revision is a conflict. This route does not acquire the
model-generation lock: its revision change invalidates a reply still being computed. Browser response
handling also rejects older revisions, other conversation IDs, and attempts to undo close/support.

While listening, model offer flags and message counts cannot restore readiness or an ordinary card.
Guided replies stay conversational. AI turns include a trusted system instruction derived only from
the validated preference (`CONVERSATION_PROMPT_VERSION=2026-10-04.2`). Live prompt adherence remains
a language-quality question; these controls do not prove that every generated sentence avoids advice.
Support routing takes precedence and its resource cards remain usable.

Act clears any old card and reopens feelings/goal selection. The next goal builds a fresh card.
The browser sends `expected_revision` when accepting. The API rejects an obsolete revision; after an
explicit preference change, acceptance without a client revision is rejected too. Legacy `auto`
chats retain acceptance without a client revision, so that older contract has no guarantee about an
obsolete displayed card. Preference-aware chats require the updated client and API; see the runbook.

## Trusted provenance

Row-level security proves who owns a row, not who wrote it. Signed-in users therefore keep SELECT and
DELETE on their own rows but have no INSERT or UPDATE on any JournalPulse table. Every write goes through
a `security definer` function that checks `auth.uid()`.

| Written by | Examples | How it is protected |
|---|---|---|
| The person | Mood face, confirmed feelings, goal, helpfulness, note, close, delete | Owner check in the function |
| The server | Policy decision and propensity, model and provider, safety-router result, action card, OPE eligibility | Owner check plus an HMAC-SHA256 signature over the exact payload text |

The signing key is shared only by the API (`JOURNALPULSE_WRITE_SIGNING_KEY`) and the database
(`private.server_secrets`, in a schema PostgREST does not expose). Each signed payload names its purpose
and owner and carries an issue time, so it cannot be replayed into another function, for another person,
or much later. A person holding their own session token can still read, export, and delete everything
they own, but cannot create records that JournalPulse later treats as system evidence.

Within a stored record, what the person reported (`self_report_input`: feelings and mood face) is kept
apart from what the server derived from it (`state`, with `derivation: "feeling-buttons-v1"`). A derived
state has no `confidence`: that field is reserved for a model's own estimate.

## Retention

| Text | Exists | Cleared |
|---|---|---|
| Messages in an open chat | So a reload can restore the chat | When the chat is accepted, closed, deleted, or idle for 24 hours |
| Messages in a chat the person chose to keep | Until they delete it | Never automatically |
| Summary, confirmed feelings, goal, and choice on a reflection | Until the person deletes it | On deletion |

The 24-hour limit is enforced by `jp_purge_expired_conversations`, scheduled every 15 minutes with
`pg_cron`. It closes idle chats, clears text that should not survive, and repairs any closed,
non-retained chat that still holds text. Each chat request also runs `jp_close_my_stale_conversations`
for the signed-in person. If the scheduled job is missing or failing, `/ready` reports
`retention_job: not_scheduled` and the request-time sweep is the only remaining path, so a person who
never returns keeps their open-chat text until the job runs again.

## Rate limiting

AI-backed requests consume signed `jp_consume_rate_limit_v2`, a per-person sliding window in Postgres guarded by an
advisory lock, so every server instance sees the same count. A refusal returns 429 with `Retry-After`
(seconds until the oldest counted request leaves the window). If the counter cannot be reached, the
request fails closed with 503 and no model call is made. A small in-memory limiter still rejects bursts
early on each instance. Usage counters hold no content, survive journal deletion (so deletion cannot
reset the limit), and expire after a day.

## Data model

Supabase Postgres, with row-level security on every table (`auth.uid() = user_id`):

| Table | Holds |
|---|---|
| `conversations`, `conversation_messages` | Open chats and their messages as JSON records, plus the `revision` |
| `conversation_preference_requests` | Owner-scoped command receipts: ID, preference, expected revision, timestamp; no text |
| `reflections` | One saved check-in: derived state, the person's report, target goal, summary, safety result, decision, and the source conversation |
| `policy_decisions` | The action offered, the person's choice, propensity, and policy version |
| `outcomes` | The later check-in for a decision |
| `affective_observations`, `model_runs`, `safety_events` | Provenance for each saved reflection |
| `rate_limit_events` | Timestamps for the shared generation limit; not readable by users |
| `profiles`, `consents`, `intervention_catalog`, `episodic_memories` | Account and research scaffolding |

Local development and tests use `SQLiteRepository`, which enforces the same revision checks,
single-accept rule, idempotency, retention, and rate limit inside serialized transactions.
SQLite connections close after commit/rollback. HTTP operations close transports
they create while preserving caller-owned injected transports. An in-process set
tracks only active conversation turns; idle rate-limit identities are periodically
removed. See [FOUNDATION_AUDIT_2026-10-05.md](FOUNDATION_AUDIT_2026-10-05.md) for
the regressions and measured local optimizations.

## Policy

`FixedBaselinePolicy` is transparent and does not learn. It chooses from at most three safety-filtered
catalog actions and logs a non-zero propensity. If the person picks a different option, the decision is
recorded as a user override and excluded from off-policy evaluation. Adaptive policies and episodic memory
exist only behind flags that stay off until the gates in the [research track](RESEARCH_TRACK.md) pass.

## Safety gate

`journalpulse.safety` matches a short list of explicit risk phrases, with negation that applies only
to the matched phrase it contains. It routes clear risk language to human support before any model call, and errs
toward support when negation is phrased in a way it does not recognise. It is not a classifier and does
not understand meaning: indirect language such as "ending it all" is not detected. The tests in
`tests/test_research_beta_safety.py` pin both the intended behaviour and these known gaps.

## API

| Route | Purpose |
|---|---|
| `GET /health` | The process answers |
| `GET /ready` | Configuration, catalog, model configuration (never probed with a paid call), database reachability, schema version, shared signing key, and retention job |
| `GET /v1/system/status` | Whether AI is configured and where data is stored |
| `POST /v1/conversations` | Start a chat (`llm_consent`, `retain_text`, `locale`) |
| `GET /v1/conversations/{id}` | Restore a chat, including the person's confirmed feelings and mood |
| `POST /v1/conversations/{id}/preference` | Persist an explicit listening/action choice and invalidate older work |
| `POST /v1/conversations/{id}/messages` | Send a message, optionally with `mood_score`, or `goal` and `confirmed_feelings` |
| `POST /v1/conversations/{id}/accept` | Save the chosen action; the state is derived from what the person confirmed |
| `POST /v1/conversations/{id}/close`, `DELETE /v1/conversations/{id}` | End or delete a chat |
| `GET /v1/reflections`, `DELETE /v1/reflections/{id}` | Past check-ins |
| `GET /v1/outcomes`, `POST /v1/outcomes` | Check-in results |
| `GET /v1/resources` | The reviewed catalog |
| `GET /v1/insights` | Descriptive totals |
| `GET /v1/export`, `DELETE /v1/account/data` | Download everything (paged through every table) or delete all journal data |
| `POST /v1/reflections/analyze`, `POST /v1/actions/preview`, `POST /v1/reflections` | The original guided-reflection API. The app no longer calls them; they remain for compatibility and tests |

`web/openapi.json` and `web/lib/generated-api.ts` are generated from the FastAPI app, and CI fails when
they drift.

## Module boundaries

- `journalpulse.safety`: deterministic support routing and exploration shutdown.
- `journalpulse.conversations`: chat routes, turn logic, goal cards, and accept.
- `journalpulse.guided`: scripted Luna for chats without AI.
- `journalpulse.self_report`: turns confirmed feelings and the mood face into the derived state.
- `journalpulse.signing`: HMAC signatures for provenance writes and the readiness probe.
- `journalpulse.intelligence`: the OpenRouter clients, prompt, schema, and deterministic fallback.
- `journalpulse.resources`: catalog validation and selection.
- `journalpulse.policy`: the policy contract and fixed baseline.
- `journalpulse.persistence`: the SQLite adapter and the Supabase adapter, with typed lifecycle errors.
- `journalpulse.api`: app assembly, readiness, rate limiting, reflections, outcomes, export, deletion,
  and static-site serving.
- `journalpulse.middleware`: request-size limits, request IDs, security headers, redacted logs, and the
  per-instance burst limiter.
- `web/`: the PWA. The service worker caches only static assets and an offline page, never API
  responses.

## Hosting

Production runs on Vercel. The build exports the Next.js app into `web-dist/`, and `app.py` exposes the
FastAPI app as a single function that serves both the site and the API from one origin. The Render
blueprint and `Dockerfile.api` package the same thing as a container. Lifecycle, rate limiting, and
retention live in Postgres, so they hold across any number of instances.

## Privacy

AI is opt-in per person, and every model request uses zero-data-retention routing. Logs record model,
provider, latency, token counts, schema validity, and fallback reason, never message text or keys.
Deleting journal data removes every journal row (reflections, provenance, outcomes, chats, messages,
memories, consents, profiles) but leaves the Supabase sign-in identity; deleting the identity is a
separate privileged task that is not built.
