# Production Deployment Plan

Last reviewed: 2026-05-23

This is the recommended job-search deployment path:

```text
Streamlit Community Cloud frontend
Render FastAPI backend
Hugging Face Hub model artifacts
Supabase Postgres/Auth later
```

## 1. API Backend on Render

The repo includes:

- `Dockerfile.api`
- `requirements-api.txt`
- `render.yaml`

Render's FastAPI guide uses Uvicorn with `--host 0.0.0.0 --port $PORT`; the Dockerfile follows that pattern through:

```bash
uvicorn emotion_journal.api:app --host 0.0.0.0 --port ${PORT:-8000}
```

Required checks:

```bash
curl https://<render-service>.onrender.com/health
curl https://<render-service>.onrender.com/ready
```

`/ready` should return:

```json
{
  "status": "ready",
  "model_ready": true,
  "database_ready": true,
  "resources_ready": true
}
```

## 2. Model Artifact Hosting

Preferred:

```text
JOURNALPULSE_HF_MODEL_ID=<your-username>/journalpulse-emotion-distilroberta
```

Then the API loads transformer weights from Hugging Face Hub while keeping local `production.json` and the local baseline explainer artifact for metadata and explanation chips.

Fallback:

- Commit `model.safetensors` through Git LFS.
- Keep `.gitattributes` tracking the transformer weight path.

## 3. Streamlit Frontend

Deploy the Streamlit app with:

```text
app/streamlit/app.py
```

Configure secrets/environment:

```text
JOURNALPULSE_API_BASE_URL=https://<render-service>.onrender.com
JOURNALPULSE_LLM_MODE=off
JOURNALPULSE_LLM_BASE_URL=https://openrouter.ai/api/v1
JOURNALPULSE_CLASSIFIER_MODE=calibrated
JOURNALPULSE_ADMIN_MODE=false
```

When `JOURNALPULSE_API_BASE_URL` is present, Streamlit calls the FastAPI backend for predictions, coach turns, resources, saves, entries, analytics, and resource interactions. Without it, Streamlit uses local in-process services.

For the stronger LLM-classifier path, keep Streamlit pointed at the API and configure the API service with:

```text
JOURNALPULSE_CLASSIFIER_MODE=llm
JOURNALPULSE_LLM_API_KEY=<provider-secret>
JOURNALPULSE_LLM_BASE_URL=https://openrouter.ai/api/v1
JOURNALPULSE_LLM_MODEL=<gemma-or-other-instruction-model>
JOURNALPULSE_LLM_APP_URL=https://<streamlit-demo-url>
JOURNALPULSE_LLM_APP_TITLE=JournalPulse
```

The API records whether the classifier response came from the structured LLM or the calibrated artifact fallback.

For the core reflection agent, keep `JOURNALPULSE_CLASSIFIER_MODE=calibrated` and set `JOURNALPULSE_LLM_MODE=structured`. The LLM writes the main practical response, while the API still controls crisis bypass, resource allowlists, schema validation, and deterministic fallback.

## 4. Database Upgrade

Current:

```text
SQLite demo storage
```

Next:

```text
Supabase Postgres + Auth + Row Level Security
```

Initial production tables:

```text
entries
predictions
coach_sessions
resource_interactions
feedback
model_versions
safety_events
request_events
```

Do not log raw journal text into `request_events`.

## 5. Smoke Test

After deployment:

```bash
curl https://<api>/ready
curl -X POST https://<api>/predict \
  -H "Content-Type: application/json" \
  -d '{"text":"I feel nervous but ready to take the next step."}'
```

Then open Streamlit and verify:

1. `Runtime` sidebar shows the deployed API status.
2. Chat returns a coach response and side-panel prediction.
3. Coach replies work.
4. Save reflection works.
5. History and Insights load through the backend.

## Official References

- Render FastAPI deployment: https://render.com/docs/deploy-fastapi
- Streamlit Community Cloud deployment: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app
- Hugging Face repositories: https://huggingface.co/docs/hub/repositories
- Supabase Row Level Security: https://supabase.com/docs/guides/database/postgres/row-level-security
