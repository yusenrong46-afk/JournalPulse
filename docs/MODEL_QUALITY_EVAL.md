# Model Quality Evaluation

JournalPulse now includes a product-shaped emotion evaluation set in `data/evals/journalpulse_eval.jsonl`.
It is synthetic, but it is written like real journal entries rather than short benchmark sentences.

Run:

```bash
PYTHONPYCACHEPREFIX=/private/tmp/journalpulse_pycache .venv/bin/python scripts/evaluate_model_quality.py
```

The report is written to `artifacts/reports/model_quality_eval.json` and appears on the Streamlit
`Model` page when present.

Current purpose:

- Find realistic failure modes before changing model families.
- Separate crisis-routing accuracy from ordinary emotion-classification quality.
- Track mixed-signal behavior through `secondary_emotions`, `emotion_tags`, `top_margin`, `is_mixed`, and `uncertainty_reason`.
- Evaluate the journal-specific calibration layer that boosts transparent cue families such as grief, boundary violations, ambiguous disruption, catastrophizing, and connection.
- Provide a baseline for future LLM structured classification or LLM fine-tuning experiments.

This report is not a clinical performance claim. It is a regression and product-quality tool.

Latest local report after journal calibration:

- Primary accuracy: `0.7667`
- Accepted-emotion accuracy: `1.0`
- Non-crisis primary accuracy: `0.7857`
- Non-crisis accepted accuracy: `1.0`
- Top-3 primary recall: `1.0`
