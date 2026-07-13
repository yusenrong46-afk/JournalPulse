# Adaptive Engine Lab

This directory is intentionally implementation-light. The fixed baseline in production is safe and
auditable; the adaptive policy is the author's manually implemented research contribution.

Do not paste generated LinUCB, Thompson Sampling, off-policy evaluation, state-space, or retrieval
implementations here. Each implementation must arrive with a handwritten derivation, a small
deterministic test, a seeded simulator comparison, and an experiment artifact under
`research/experiments/<kind>/`.

The production boundary is `journalpulse.policy.ReflectionPolicy`. A candidate policy cannot cross
that boundary until it logs a valid propensity, chooses only from the supplied safe action set, passes
the safety kill-switch suite, and outperforms the fixed baseline on preregistered simulator seeds.
