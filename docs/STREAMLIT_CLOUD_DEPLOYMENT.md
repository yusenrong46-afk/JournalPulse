# Streamlit Cloud Deployment

JournalPulse is optimized for Streamlit Community Cloud as the public demo surface.
FastAPI remains available locally for API walkthroughs.

## Deployment Settings

- Repository path: project root
- App entrypoint: `app/streamlit/app.py`
- Python version: `3.9`
- Dependency file: `requirements-streamlit.txt` for split frontend deployment, or `requirements.txt` for all-in-one local demo deployment
- Streamlit config: `.streamlit/config.toml`

Run locally from the repository root before deploying:

```bash
source .venv/bin/activate
PYTHONPATH=src streamlit run app/streamlit/app.py
```

## Model Artifacts

The current transformer weight file is large:

```text
artifacts/models/transformer_model/model.safetensors
```

Use Git LFS for `*.safetensors` model weights. The repository includes
`.gitattributes` entries for the current production artifact and any optional
second-transformer artifact directories.

Useful setup commands:

```bash
git lfs install
git lfs track "artifacts/models/transformer_model/model.safetensors"
git lfs track "artifacts/models/transformer_*/model.safetensors"
```

Streamlit Community Cloud supports repositories that use Git LFS.

The repository should include the production metadata and tokenizer/config files:

```text
artifacts/models/production.json
artifacts/models/baseline.joblib
artifacts/models/transformer_model/config.json
artifacts/models/transformer_model/tokenizer.json
artifacts/models/transformer_model/model.safetensors
```

The `.gitignore` keeps local databases and obsolete generated artifacts out of
version control while allowing the production artifacts needed at startup.

## Secrets and Optional Modes

Do not commit secrets. Configure these in Streamlit Cloud secrets or local
environment variables:

```text
JOURNALPULSE_LLM_API_KEY=...
JOURNALPULSE_LLM_BASE_URL=https://openrouter.ai/api/v1
JOURNALPULSE_LLM_MODEL=...
JOURNALPULSE_LLM_APP_URL=https://<your-streamlit-app-url>
JOURNALPULSE_LLM_APP_TITLE=JournalPulse
JOURNALPULSE_LLM_MODE=off
JOURNALPULSE_ADMIN_MODE=false
JOURNALPULSE_API_BASE_URL=https://<deployed-fastapi-service>
```

`JOURNALPULSE_LLM_MODE` supports `off`, `rewrite`, and `structured`. Leave it as
`off` for the most predictable public demo, or use `structured` only with
secrets configured in Streamlit Cloud.

If `JOURNALPULSE_API_BASE_URL` is set, Streamlit calls the deployed FastAPI
backend for predictions, resources, coach turns, saves, history, and analytics.
If it is unset, Streamlit runs in local in-process demo mode.

`JOURNALPULSE_ADMIN_MODE=true` exposes the Resource Admin page. Keep it off for
the public demo unless you intentionally want reviewers to inspect the catalog
workbench.

## Persistence Caveat

The demo uses local SQLite by default:

```text
artifacts/journal.db
```

That is fine for local demos and recruiter walkthroughs, but Streamlit Community
Cloud does not guarantee durable local filesystem persistence. Treat saved
entries and resource interactions in the hosted app as demo/session data unless
you later wire the app to a managed database.

The Streamlit app shows seeded recruiter-demo data in `History` and `Insights`
when the local database is empty, so a fresh hosted session still looks useful.

## Verification

Before deployment:

```bash
source .venv/bin/activate
PYTHONPATH=src pytest
PYTHONPATH=src python scripts/validate_resources.py
PYTHONPATH=src python scripts/train.py --dry-run-candidates --transformers distilroberta-base,bert-base-uncased
PYTHONPATH=src streamlit run app/streamlit/app.py
```

Official deployment references used for these settings:

- https://docs.streamlit.io/deploy/concepts/dependencies
- https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/file-organization
- https://docs.streamlit.io/deploy/concepts/secrets
- https://docs.streamlit.io/develop/concepts/connections/connecting-to-data
