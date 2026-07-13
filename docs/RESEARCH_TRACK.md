# JournalPulse 16-Week Research Track

## Non-negotiable boundary

The product shell, API plumbing, tests, and research instrumentation may be assisted. The moat is not:
you derive and manually implement the simulator, adaptive policies, causal estimators, state-space
model, and memory retrieval algorithms. A generated implementation would weaken both your learning
and the portfolio claim.

## Two-hour session protocol

1. **Recall, 15 minutes:** reproduce the previous session's key equations or invariants without notes.
2. **Derive, 30 minutes:** derive the next update rule, estimator, or state transition on paper.
3. **Implement, 60 minutes:** write the smallest NumPy implementation yourself, without autocomplete.
4. **Verify, 15 minutes:** add a deterministic test and record what failed, changed, and remains unknown.

Use five build sessions and one weekly review. The weekly review produces one experiment artifact and
one short technical note; it does not produce a success claim unless the evidence supports it.

## Weekly gates

| Week | Manual contribution | Evidence required before moving on |
|---|---|---|
| 1 | Define state, action, reward, safety, and logging contracts | Architecture map and invariant tests |
| 2 | Understand Postgres ownership and RLS | Two-user isolation and deletion tests |
| 3 | Implement repository boundaries | Migration and failure-recovery tests |
| 4 | Build frozen structured-extraction cases | Live schema, ZDR, latency, and fallback report |
| 5 | Implement heterogeneous users and temporal drift | Recovery of known simulator parameters |
| 6 | Implement random, fixed, oracle, and context-free policies | Seeded regret curves with confidence intervals |
| 7 | Derive LinUCB confidence ellipsoid | Hand derivation and one-arm sanity test |
| 8 | Implement LinUCB with NumPy | Beats random in a stationary linear environment |
| 9 | Derive Bayesian linear posterior updates | Posterior matches closed-form small case |
| 10 | Implement Thompson Sampling and partial pooling | Calibrated intervals and subgroup comparison |
| 11 | Implement IPS and doubly robust evaluation | Estimates recover simulator ground truth |
| 12 | Implement a linear state-space/Kalman baseline | Filtering improves noisy-state error |
| 13 | Implement recency, vector, and hybrid retrieval | Frozen relevance and provenance benchmark |
| 14 | Implement graph retrieval and ablations | Quality, diversity, latency, and safety report |
| 15 | Integrate one feature-flagged candidate | Kill-switch and propensity audit pass |
| 16 | Analyze at least 40 benign decisions | Failure-first technical report; no clinical claim |

## Promotion checklist

- A policy chooses only from the safety-filtered catalog.
- Every choice logs its non-zero propensity and exact policy version.
- Support mode bypasses LLM analysis, memory retrieval, and exploration.
- Off-policy estimates are validated against known simulator truth before personal data is interpreted.
- Memory evidence preserves source reflection, timestamps, validity, and deletion lineage.
- Candidate performance is reported over preregistered seeds with uncertainty, not one favorable run.
- A human can disable the LLM, memory, and adaptive policy independently.
