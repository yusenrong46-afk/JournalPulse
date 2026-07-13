# JournalPulse Research Beta Architecture

```text
Next.js PWA -> FastAPI -> auth boundary -> safety gate
                                | normal
                                v
                       consent-aware OpenRouter
                                v
                       user-corrected state
                                v
                  safe catalog -> fixed policy
                                v
                 Postgres + RLS -> delayed outcome
```

The consumer product is a guided workflow, not a chatbot. Safety runs before external processing.
The model proposes a bounded affective state; the person corrects it; a policy receives only approved
catalog actions. Production currently uses a transparent fixed policy. Adaptive algorithms and episodic
memory remain disabled until their independent evidence gates pass.

## Responsibility boundaries

- `journalpulse.safety`: deterministic support routing and exploration shutdown.
- `journalpulse.intelligence`: consent-aware structured extraction and deterministic fallback.
- `journalpulse.resources`: validation and selection from the reviewed HTTPS catalog only.
- `journalpulse.policy`: policy contract and fixed baseline; no hidden model blending.
- `journalpulse.persistence`: user-scoped local test adapter and RLS-preserving Supabase adapter.
- `journalpulse.api`: orchestration, validation, history, outcomes, insights, export, and deletion.
- `web/`: mobile-first user product; no raw journal content in the service-worker cache.
- `research/`: internal console and manually authored moat work.

## Privacy posture

Raw text retention is off by default. External analysis is opt-in per reflection and requires ZDR routing.
Logs contain model name, provider, latency, token counts, schema validity, and fallback reason, but never
API credentials or journal text. The bulk-delete operation removes all user-owned research records while
leaving the Supabase Auth identity active; auth-account deletion is a separate privileged deployment task.
