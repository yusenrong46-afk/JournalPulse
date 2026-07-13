# JournalPulse Production Architecture

Last reviewed: 2026-05-23

JournalPulse is moving from a polished local demo to a production-minded AI/NLP product case study. The target architecture separates the recruiter-facing Streamlit UI from the FastAPI inference/product backend while keeping the local single-process mode for development.

## Target Shape

```text
User
  |
  v
Streamlit frontend
  |
  | HTTPS via JOURNALPULSE_API_BASE_URL
  v
FastAPI backend
  |-- emotion classifier
  |-- TF-IDF explanation layer
  |-- resource ranking service
  |-- bounded coach service
  |-- safety/crisis router
  |-- analytics API
  |
  v
Postgres/Supabase later
  |
  v
model/version/eval/monitoring metadata
```

## Current Implementation

- Streamlit can still run fully local for development and demos.
- If `JOURNALPULSE_API_BASE_URL` is set, Streamlit calls the FastAPI backend for predictions, saves, entries, analytics, resource lists, resource recommendations, coach turns, and resource interactions.
- FastAPI exposes `/health` for process health and `/ready` for production readiness.
- The backend can load the transformer from local artifacts or from Hugging Face Hub using `JOURNALPULSE_HF_MODEL_ID`.
- SQLite remains the local demo persistence layer until the Postgres/Supabase adapter is implemented.

## Runtime Modes

| Mode | Configuration | Purpose |
| --- | --- | --- |
| Local demo | no `JOURNALPULSE_API_BASE_URL` | Streamlit calls local Python services directly. |
| Split demo | `JOURNALPULSE_API_BASE_URL=https://...` | Streamlit frontend calls deployed FastAPI backend. |
| API deployment | `JOURNALPULSE_DEPLOYMENT_MODE=api` | Render/Fly/Railway-style FastAPI service. |

## Readiness Contract

- `/health` means the API process is alive.
- `/ready` checks model loading, database initialization, and resource catalog loading.

This distinction is useful for platform health checks and recruiter-facing architecture discussion.

## Model Artifact Strategy

Preferred production story:

```text
JOURNALPULSE_HF_MODEL_ID=<your-huggingface-username>/journalpulse-emotion-distilroberta
```

The API still reads local `production.json` for metadata and the local TF-IDF baseline explainer, but transformer weights can be loaded from a Hugging Face model repository. Hugging Face Hub repositories are designed for ML artifacts, including model files and related repository assets.

Fallback story:

- Keep `artifacts/models/transformer_model/model.safetensors` under Git LFS.
- Keep `artifacts/models/production.json`, tokenizer files, config files, and `baseline.joblib` discoverable at startup.

## Next Architecture Steps

1. Add a Postgres adapter behind the existing `db.py` interface.
2. Add Supabase Auth and Row Level Security for user-owned rows.
3. Add `model_versions`, `predictions`, `safety_events`, and `request_events` tables.
4. Add structured logs that never include raw journal text.
5. Add MLflow or DVC for experiment/model registry discipline.

## Official References

- Streamlit Community Cloud deployment docs: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app
- Render FastAPI deployment docs: https://render.com/docs/deploy-fastapi
- Supabase Row Level Security docs: https://supabase.com/docs/guides/database/postgres/row-level-security
- Hugging Face Hub repository docs: https://huggingface.co/docs/hub/repositories
