# Streamlit Cloud Deployment

JournalPulse is optimized for Streamlit Community Cloud as the public demo surface.
FastAPI remains available locally for API walkthroughs.

## Deployment Settings

- Repository path: project root
- App entrypoint: `app/streamlit/app.py`
- Python version: `3.9`
- Dependency file: `requirements.txt`
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

## Secrets and Optional Modes

Do not commit secrets. Configure these in Streamlit Cloud secrets or local
environment variables:

```text
JOURNALPULSE_LLM_API_KEY=...
JOURNALPULSE_LLM_BASE_URL=https://api.openai.com/v1
JOURNALPULSE_LLM_MODEL=...
JOURNALPULSE_ADMIN_MODE=false
```

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

## Verification

Before deployment:

```bash
source .venv/bin/activate
PYTHONPATH=src pytest
PYTHONPATH=src python scripts/train.py --dry-run-candidates --transformers distilroberta-base,bert-base-uncased
PYTHONPATH=src streamlit run app/streamlit/app.py
```
