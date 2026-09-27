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

Everything a person does happens in one conversation. The server owns the parts that must be trusted:
the safety check, the model call, the catalog, the policy, and what gets saved. The browser owns the
gentle prompts in between (the feelings and goal buttons) and sends only their result.

## Chat stages

| Stage | Where it lives | What happens |
|---|---|---|
| Welcome | Browser | Greeting and mood faces. The first tap or message creates the conversation |
| Chat | Server | Each message runs the safety check, then Luna replies. The reply can mark the chat `ready_for_action` and suggest up to three `feelings` from a fixed list |
| Feelings | Browser | Luna's suggested feelings are preselected; the person edits them |
| Goal | Browser → server | The chosen goal is sent as an ordinary message with a `goal` field. The server builds the card from the catalog without a model call |
| Offer | Server data, browser view | Three catalog actions; the fixed baseline's pick is marked "Luna's pick" |
| Saved | Server | `POST /accept` writes a normal `ReflectionRecord` using the person's own feelings, the goal, and the chosen action |
| Check-in | Server | `POST /v1/outcomes` records whether it was tried, how much it helped, and optional feelings and a note |
| Support | Server | Any message that trips the safety check moves the chat into support mode for good. The model is never called again in that chat |

The goal message goes through the safety gate like any other message, so typing a crisis message into a
goal turn still routes to support.

### Smart and Simple Luna

A conversation records its `mode` when it starts:

- `ai`: the person allowed AI and a model is configured. Replies come from
  `OpenRouterConversationClient` (`JOURNALPULSE_CHAT_MODEL`, default `openai/gpt-6-luna`) with a strict
  JSON schema, zero-data-retention routing, and bounded retries. There is no canned reply when the
  provider fails; nothing is saved for that turn.
- `guided`: no AI. `journalpulse.guided` returns fixed questions, guesses feelings by keyword, and marks
  the chat ready after the third message. It never copies the person's words into the summary.

When a record is saved, its `model_run` names the model that actually held the conversation, not the
local goal step.

## Data model

Supabase Postgres, with row-level security on every table (`auth.uid() = user_id`):

| Table | Holds |
|---|---|
| `conversations`, `conversation_messages` | Open chats and their messages, stored as JSON records |
| `reflections` | One saved check-in: state, target goal, summary, safety result, decision |
| `policy_decisions` | The action offered, the person's choice, propensity, and policy version |
| `outcomes` | The later check-in for a decision |
| `affective_observations`, `model_runs`, `safety_events` | Provenance for each saved reflection |
| `profiles`, `consents`, `intervention_catalog`, `episodic_memories` | Account and research scaffolding |

Writes that touch several tables go through `security invoker` functions so they run in one transaction
and still obey RLS: `save_reflection_bundle`, `save_outcome_record`, `save_conversation_turn`,
`close_conversation`, and `delete_my_journalpulse_data`. Every browser write carries a client UUID, and a
repeated request returns the original record instead of creating a duplicate.

Local development and tests use `SQLiteRepository`, which enforces the same ownership and idempotency
rules.

## Policy

`FixedBaselinePolicy` is transparent and does not learn. It chooses from at most three safety-filtered
catalog actions and logs a non-zero propensity. If the person picks a different option, the decision is
recorded as a user override and excluded from off-policy evaluation. Adaptive policies and episodic memory
exist only behind flags that stay off until the gates in the [research track](RESEARCH_TRACK.md) pass.

## API

| Route | Purpose |
|---|---|
| `GET /health`, `GET /ready` | Liveness, and readiness of configuration, catalog, model, and storage |
| `GET /v1/system/status` | Whether AI is configured and where data is stored |
| `POST /v1/conversations` | Start a chat (`llm_consent`, `retain_text`, `locale`) |
| `GET /v1/conversations/{id}` | Restore an open chat |
| `POST /v1/conversations/{id}/messages` | Send a message, optionally with a `goal` |
| `POST /v1/conversations/{id}/accept` | Save the chosen action with the person's own feelings |
| `POST /v1/conversations/{id}/close`, `DELETE /v1/conversations/{id}` | End or delete a chat |
| `GET /v1/reflections`, `DELETE /v1/reflections/{id}` | Past check-ins |
| `GET /v1/outcomes`, `POST /v1/outcomes` | Check-in results |
| `GET /v1/resources` | The reviewed catalog |
| `GET /v1/insights` | Descriptive totals |
| `GET /v1/export`, `DELETE /v1/account/data` | Download or delete everything |
| `POST /v1/reflections/analyze`, `POST /v1/actions/preview`, `POST /v1/reflections` | The original guided-reflection API. The app no longer calls them; they remain for compatibility and tests |

`web/openapi.json` and `web/lib/generated-api.ts` are generated from the FastAPI app, and CI fails when
they drift.

## Module boundaries

- `journalpulse.safety`: deterministic support routing and exploration shutdown.
- `journalpulse.conversations`: chat routes, the turn logic, goal cards, and accept.
- `journalpulse.guided`: scripted Luna for chats without AI.
- `journalpulse.intelligence`: the OpenRouter clients, prompt, schema, and deterministic fallback.
- `journalpulse.resources`: catalog validation and selection.
- `journalpulse.policy`: the policy contract and fixed baseline.
- `journalpulse.persistence`: the SQLite adapter and the RLS-preserving Supabase adapter.
- `journalpulse.api`: app assembly, reflections, outcomes, export, deletion, and static-site serving.
- `journalpulse.middleware`: request-size limits, request IDs, security headers, redacted logs, and the
  rate limiter.
- `web/`: the PWA. The service worker caches only static assets and an offline page, never API
  responses.

## Hosting

Production runs on Vercel. The build exports the Next.js app into `web-dist/`, and `app.py` exposes the
FastAPI app as a single function that serves both the site and the API from one origin. The Render
blueprint and `Dockerfile.api` package the same thing as a container. Because the rate limiter lives in
memory, each Vercel instance counts separately; move it into Postgres before relying on it at scale.

## Privacy

AI is opt-in per person, and every model request uses zero-data-retention routing. Logs record model,
provider, latency, token counts, schema validity, and fallback reason, never message text or keys. Chat
text is cleared when a chat ends unless the person chose to keep it. Deleting journal data removes every
user-owned record but leaves the Supabase sign-in identity; deleting the identity is a separate
privileged task.
