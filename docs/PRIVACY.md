# JournalPulse Privacy Design

Last reviewed: 2026-05-23

JournalPulse handles journal-style text, so privacy is part of the product architecture. The public demo should feel useful without asking visitors to share sensitive personal data.

## Current Demo Posture

- The hosted demo should use seeded synthetic entries for `History` and `Insights`.
- The public demo should not require login.
- The deterministic coach is the default.
- Optional LLM mode is disabled unless deployment secrets are configured.
- Crisis-mode routing bypasses LLM generation.
- Raw coach transcript and free-form coach messages are not persisted.

## Data Minimization Rules

Production logging should avoid raw journal text. Prefer metadata only:

```text
entry_length_bucket
predicted_emotion
confidence_band
model_version
safety_mode
resource_intent
resource_action
latency_ms
error_type
```

Do not send raw journal entries to third-party monitoring tools.

## User-Owned Data Target

When moving beyond demo mode:

- Use managed Postgres.
- Use Supabase Auth or a comparable auth provider.
- Enable Row Level Security on user-owned tables.
- Store rows with `user_id`.
- Give users a delete/export path.

Supabase documents Row Level Security as a Postgres security layer that can be combined with Supabase Auth so users only access rows allowed by policy.

## Public Demo Boundary

Add or preserve visible copy:

```text
JournalPulse is not therapy, diagnosis, or crisis counseling.
It is a journaling and reflection support tool.
Please do not enter emergency or highly sensitive personal information into the public demo.
```

## LLM Boundary

If LLM mode is enabled:

- Do not call the LLM in crisis mode.
- Do not pass prior journal history unless the user opts in.
- Validate structured LLM output with Pydantic.
- Use allowed resource IDs only.
- Fall back to deterministic coach output on invalid output.

## Future Tables

```text
users
entries
predictions
coach_sessions
resource_interactions
feedback
model_versions
safety_events
request_events
```

`entries.text` should be treated as sensitive. `request_events` and `safety_events` should store metadata, not raw journal content.
