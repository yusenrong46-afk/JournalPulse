# Safety and Limitations

Last reviewed: 2026-05-09

JournalPulse is a reflective journaling demo, not a mental-health product. It can help a user name an emotional signal, choose a small next step, and find a curated resource. It does not diagnose, treat, or counsel.

## Safety Boundary

- The app is non-clinical by design.
- Crisis language bypasses ordinary coaching and switches to safety-first support copy.
- LLM generation is disabled in crisis mode.
- Crisis-safe resources are restricted to human-support or official support pages.
- Saved coach summaries exclude raw coach transcript and free-form user coach messages.

## LLM Guardrails

`JOURNALPULSE_LLM_MODE` supports:

- `off`: deterministic coach only. This is the default.
- `rewrite`: optional short wording polish for deterministic coach text.
- `structured`: validated reflection-agent output with `assistant_message`, `practical_steps`, `suggested_replies`, `resource_intent`, `resource_ids`, `reflection_question`, optional `communication_draft`, `confidence_note`, and fallback metadata.

Structured mode is constrained by:

- short token limits
- JSON response format plus Pydantic validation
- allowed resource IDs only
- allowed resource intent enums only
- deterministic fallback on invalid output
- non-clinical copy validation

OpenRouter is the preferred v1 provider because it exposes an OpenAI-compatible chat-completions API while allowing model IDs such as Gemma, Llama, or Qwen to be swapped through configuration. Optional app attribution headers are configured with `JOURNALPULSE_LLM_APP_URL` and `JOURNALPULSE_LLM_APP_TITLE`.

This follows the current OpenAI guidance to constrain outputs with structured schemas and to limit input/output surfaces for safety-sensitive features:

- https://platform.openai.com/docs/guides/structured-outputs
- https://platform.openai.com/docs/guides/safety-best-practices/constrain-user-input-and-limit-output-tokens.pls

## Resource Quality

Resource cards include:

- `goal_tags` for routing by user intent
- `source_tier` for credibility display
- `reviewed_at` for catalog maintenance
- `rationale` text explaining why the card was selected

The offline validator checks required fields, URL shape, supported source tiers, supported goals, catalog coverage, and duplicate IDs. Optional online link checking is available:

```bash
PYTHONPATH=src python scripts/validate_resources.py --check-links
```

## Deployment and Data Limits

The Streamlit demo uses local SQLite. Streamlit documents local SQLite as a possible simple starting point, but Community Cloud does not guarantee persistence for local file storage:

- https://docs.streamlit.io/develop/concepts/connections/connecting-to-data

Secrets should be configured outside source code:

- https://docs.streamlit.io/deploy/concepts/secrets

Required model files must be present in the repository or fetched deterministically at startup. JournalPulse keeps the large transformer `model.safetensors` on the Git LFS path documented in `.gitattributes`.

## Residual Risks

- Crisis detection is a bounded heuristic, not a clinical risk assessment.
- A user can write in ways the detector or classifier does not understand.
- Resource links can change after review.
- Hosted demo data is illustrative and may disappear between Streamlit Cloud sessions.
- Optional LLM output can fail validation and fall back; this is expected behavior, not an outage.
