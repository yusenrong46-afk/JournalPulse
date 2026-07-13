# JournalPulse Model Card

Last reviewed: 2026-05-09

## Model Summary

JournalPulse uses a six-class emotion classifier for reflective journaling support. The default production artifact is a fine-tuned `distilroberta-base` sequence classifier trained on the `dair-ai/emotion` dataset, then adapted with transparent journal-language calibration. A classical TF-IDF linear model is retained as an explanation layer so the app can show phrase-level signals alongside the prediction.

The app now also supports an optional structured LLM classifier mode (`JOURNALPULSE_CLASSIFIER_MODE=llm|hybrid`) for stronger journal-style understanding. That path can be pointed at Gemma or another OpenAI-compatible chat-completions provider. Its output is validated with Pydantic, crisis entries bypass it, and invalid/missing LLM output falls back to the calibrated artifact.

## Intended Use

- Support non-clinical journaling reflection.
- Classify short journal entries into `sadness`, `joy`, `love`, `anger`, `fear`, or `surprise`.
- Provide confidence bands, reflection prompts, and curated resource recommendations.
- Demonstrate an applied AI product architecture for portfolio/recruiter review.

## Not Intended For

- Therapy, diagnosis, medical advice, or crisis counseling.
- Determining a user's mental health status.
- Replacing professional care, emergency services, or trusted human support.
- High-stakes decision-making.

## Current Production Metrics

From the checked-in evaluation report:

- Selected model: `distilroberta-base`
- Transformer test accuracy: `0.9000`
- Transformer macro F1: `0.8600`
- Logistic Regression test accuracy / macro F1: `0.8295` / `0.7264`
- LinearSVC test accuracy / macro F1: `0.8795` / `0.8224`

## Product Behavior

The classifier returns a dominant emotion, class probabilities, and a confidence band. The product layer then adds:

- a short reflection summary
- an interpretation of the likely signal
- exactly three follow-up prompts
- phrase-level explanation chips from the classical explainer
- resource cards ranked by emotion, user intent, metadata, and interaction feedback
- a bounded coach turn with deterministic fallback
- classifier provenance fields: `classifier_mode`, `classifier_source`, and `classifier_fallback_reason`

## Known Limitations

- The dataset is label-limited and does not represent every culture, age group, writing style, or distress pattern.
- The emotion label can be wrong or overly simple for mixed feelings.
- Confidence is model confidence, not clinical certainty.
- Phrase explanations come from a classical proxy model, not direct transformer internals.
- Local SQLite persistence is demo-oriented and not durable on Streamlit Community Cloud.

## Safety Handling

JournalPulse runs crisis-language detection before ordinary recommendations. If an entry appears to describe acute self-harm risk or immediate unsafety, the app switches to safety mode, suppresses ordinary entertainment/distraction cards, and points to human support resources such as 988 in the United States.

Optional LLM coaching is constrained to wording/structured coach assistance. The deterministic coach and resource engine remain the fallback and source of truth.
