# Recruiter Demo Script

Use this for a 3-minute walkthrough of the hosted Streamlit demo.

## Setup

- Open the Streamlit app at `app/streamlit/app.py`.
- Keep `JOURNALPULSE_LLM_MODE=off` unless you want to show the OpenRouter-powered structured agent path.
- If using LLM mode, configure secrets outside source code:
  - `JOURNALPULSE_LLM_MODE=structured`
  - `JOURNALPULSE_LLM_API_KEY`
  - `JOURNALPULSE_LLM_BASE_URL=https://openrouter.ai/api/v1`
  - `JOURNALPULSE_LLM_MODEL`
  - `JOURNALPULSE_LLM_APP_URL`
  - `JOURNALPULSE_LLM_APP_TITLE`

## 3-Minute Walkthrough

1. Chat
   Paste this sample:

   ```text
   The meeting made me angry because I felt talked over. I do not want to explode, but I also do not want to pretend it was fine.
   ```

   Expected talking points:
   - The model detects `anger` with a confidence band.
   - The main surface is a chat, while signals and resources stay in the side context panel.
   - Resource cards include credibility badges, goal tags, and "why this resource" rationale.

2. Guided Coach
   Click `Give me tips`, then `Help me plan`.

   Expected talking points:
   - The agent controls the helpful response shape: practical steps, suggested replies, reflection question, and optional communication draft.
   - The app still controls safety, resource allowlists, and fallback behavior.
   - OpenRouter lets the model ID be swapped without changing the app.

3. Save Reflection
   Mark the reflection `Helpful` and save it.

   Expected talking points:
   - SQLite stores the reflection, structured coach summary, and suggested resource IDs.
   - Raw coach transcript is not persisted.

4. Insights
   Open `Insights`.

   Expected talking points:
   - Empty hosted sessions show seeded recruiter-demo data.
   - Saved session data replaces the demo view locally.
   - Charts show emotion trends, confidence patterns, resource actions, and helpful resources.

5. Model
   Open `Model`.

   Expected talking points:
   - The transformer is the production classifier.
   - The classical linear model remains useful for phrase explanations.
   - The project includes evaluation artifacts, deployment notes, validation scripts, and tests.

## Safety Demo

Use a short, safe-to-demo crisis phrase only if appropriate:

```text
I do not feel safe tonight and I need help.
```

Expected behavior:

- Ordinary reflection coaching is bypassed.
- Crisis-safe support resources appear.
- LLM generation is bypassed even if `JOURNALPULSE_LLM_MODE=structured`.

## Interview Talking Points

- End-to-end product: training, artifacts, FastAPI, Streamlit, SQLite, tests, and docs.
- Safety posture: non-clinical boundary, crisis routing, deterministic fallback, no raw coach transcript persistence.
- Reliability: seeded demo data for empty Streamlit Cloud sessions and CI checks for tests, catalog validation, and training configuration.
- Extensibility: optional second transformer benchmark, resource admin workflow, OpenRouter-powered structured reflection agent.
