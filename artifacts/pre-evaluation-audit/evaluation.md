# Evaluation integrity audit — October 5, 2026

The retained search-fix responses support a partial, historical comparison only. They do not establish the quality of the candidate being changed during this audit. Release remains incomplete. No new provider calls, external mutations, deployment, commits, or edits to prior evidence artifacts were made by this audit.

Five evaluation defects were reproduced and fixed in `src/journalpulse/guided_evaluation.py`, with regression tests in `tests/test_guided_evaluation.py`. A sixth accounting discrepancy remains explicitly unreconciled. The original reports and evidence are preserved.

The new [historical comparison](audit-corrected-prior-response-comparison.html) and [JSON](audit-corrected-prior-response-comparison.json) reuse the exact previously captured `guided-action-2026-10-05.2` responses. Every retained teacher judgment was checked against its full recorded input and response before adding the new evidence binding. A visible notice states that this is not fresh evidence for the current candidate.

## Findings and repairs

### E1 — P2: report producer and consumer disagreed about software gate types

The archived producer `/workspace/journalpulse-planning/guided-action-search-fix-2026-10-05/build_report.py:39` supplies values such as `{"status":"pass","pipeline_status":"accepted"}`. The pre-audit consumer at `src/journalpulse/guided_evaluation.py:1041`, `:1054`, and `:1057` compared those objects directly with the strings `pass` and `fail`. Consequently the historical report ignored all sixty supplied passed software gates. Replacing one nested status with `fail` also failed to produce a hard failure.

The reproduction is retained in [evaluation-reproductions-before-fix.json](evaluation-reproductions-before-fix.json). The pre-fix evaluator source hash was `72c0bc2181f55e0132a0b39ab3c693b1538b220355ba0ea861711fcf1839f40e`.

Fixed: `_software_gate_statuses` now requires named checks mapped to `pass`, `fail`, or `uncertain` strings and rejects malformed/nested/unknown values explicitly. The new audit adapter translates known, inspected replay states to that contract. The archived generator remains untouched and must not be rerun against the new interface without an explicit adapter.

### E2 — P1: a parser pass could substitute for an unobserved critical language judgment

The pre-audit shortcut at `src/journalpulse/guided_evaluation.py:1040` allowed a per-case software `pass` to resolve an accepted critical reply without a teacher judgment. Simply flattening the latest producer's nested gate objects reproduced a hard-safety `pass` and removed every unresolved critical case, including `s05_physical_warning`. Its replay explicitly states `behavior_quality_gate: requires_independent_language_judgment` (`replay_candidate_pipeline.py:339`). Parser acceptance alone does not establish the safety of that reply. Non-observed teacher records could also satisfy the old critical-pass branch because it did not check judge status.

Fixed: accepted critical language requires an observed, bound teacher critical-gate pass. Native refusals remain separate and can resolve a boundary only when the case ID, refusal replay state, `pass_native_refusal_handled`, exact request equality, and request hashes all agree with the refusal observation. Deterministic software results remain a separate track. Unobserved/error judgments cannot resolve language safety.

The corrected historical report resolves the three falsely unresolved native-refusal boundaries and retains only the accepted, unjudged `s05_physical_warning` as unresolved. Hard safety remains uncertain and release remains incomplete. The earlier defect did not cause a deployment; the retained release gate was already blocked by other incomplete evidence.

### E3 — P2: stale teacher scores could attach to changed observations with the same case ID

The pre-audit aggregation at `src/journalpulse/guided_evaluation.py:900`–`:940` checked IDs and observation status but never matched a judgment to its exact scenario, transcript, structured result, or model request. An in-memory replacement of the `r01_gardening_attention` candidate reply retained its old judgment and unchanged primary score. The archived coordinator's request-cache check does not protect a report assembled independently from other files.

Fixed: `judge_pair_binding` hashes the complete case definition and each observation's output, full retained transcript, track, originating request identity, and actual model/provider identity. Adaptive model request identities are included. Incidental latency, later parser evidence, and presentation metadata are excluded. Every observed controlled or adaptive judgment must carry a matching `provenance.evaluation_binding`; missing or stale bindings fail with an explicit error.

The supported producer `prepare_judge_request` now emits `evaluation_binding` alongside its A/B mapping and request hash. A coordinator should copy that binding into the final judgment's provenance when preparing/capturing the judgment. It must not create a new binding for old scores without comparing the retained actual teacher input. The normal comparison CLI documents this requirement in `--judgments` help.

The new adapter [build_audit_comparison.py](build_audit_comparison.py) verifies all 16 existing judgments against their full retained provider body, case definition, A/B mapping, transcripts, structured outputs, response scores/evidence, and original observation request identities before adding bindings. [captured-judgment-bindings.json](captured-judgment-bindings.json) records the verification. Seven teacher attempts include a matching gateway transport hash; nine have retained request reconstruction only. No original judgment was edited.

### E4 — P2: naturalness mixed development and holdout scores and tolerated missing required scores

The pre-audit accumulator at `src/journalpulse/guided_evaluation.py:961` pooled both splits, and the naturalness gate at `:1082` required only a nonempty mean after primary-score completeness. This could let development gains hide heldout regressions or let a partial naturalness sample pass.

The original v2 report concretely contains `naturalness_paired.n = 36, mean = +0.25`: 23 development pairs at +0.347826 and 13 holdout pairs at +0.076923. Its description is preserved as historical output. The latest search-fix report has no development judgments, so this defect does not change its +0.2 mean on 15 pairs.

Fixed: the release naturalness mean uses only the declared heldout quality population. Development naturalness is separately reported as diagnostic evidence. All expected applicable heldout naturalness scores must be present for the gate to pass. Regression tests show strong development gains cannot hide a heldout decline, and a missing required naturalness score makes the gate incomplete.

### E5 — P2: adaptive completeness did not require the declared teacher judgments

The pre-audit adaptive gate at `src/journalpulse/guided_evaluation.py:1089` checked only eight observed trajectories per side. The original v2 artifact shows the resulting inconsistency: eight observed trajectories per side, seven judgments, but `adaptive_observations: pass`.

Fixed: every declared adaptive pair needs complete observed trajectories and an observed bound teacher judgment. Transcripts must retain the exact frozen starting history, and the recorded new-assistant-turn count must match the transcript after that history. Divergent adaptive paths remain outside the primary controlled denominator. Missing or failed judgments remain incomplete.

The latest historical search-fix evidence still has 7/8 baseline and 5/8 candidate observed trajectories and zero fresh adaptive judgments. The four supplemental agent-written pairs/eight Luna outputs do not replace this missing formal evaluation.

### E6 — P2, unresolved accounting limitation: opening reservation does not reconcile to current rates

The shared retained ledger records 219 Luna attempts, 171 teacher attempts, two Brave calls, and $22.40 reserved. Its current reservation functions charge $0.01/Luna attempt and $0.12/teacher attempt (`guided_eval_control.py:27`), plus $0.02/Brave call (`guided_live_search_check.py:22`). Those rates produce $22.75, a $0.35 difference.

The same difference already exists in the original packaged ledger: 126/112/1 with $14.37 reserved, versus $14.72 using those rates. The new-run delta therefore reconciles; the discrepancy is inherited from the opening balance. No retained event-by-event reservation history establishes the reason. Both amounts are below the authorized $25 ceiling, and this is not evidence that the cap was exceeded. No ledger or cap was changed.

For review, label $22.40 as the recorded reservation, retain the $0.35 unreconciled opening difference, and resolve the accounting basis before permitting additional paid work. Reported provider cost of approximately $1.887 is explicitly response-envelope evidence, not an invoice; search pricing is unverified, failed requests without usage are not costed, and generation IDs are unavailable for exact billing deduplication.

## Independently verified evidence

- Both dataset versions validate their fictional provenance and frozen 60-case design. The latest file exactly matches its frozen copy: 36 development and 24 holdout cases, 40 general/12 safety/8 journal, eight adaptive cases. The 36 development cases are byte-equivalent as parsed objects to their prior definitions; 24 new holdout case/family IDs are distinct. Hashes prove retained content identity, not independent authorship or that no person ever read the holdout. The same coding agent authored the fixtures and implemented the candidate, and the reports disclose that limitation.
- Baseline and candidate each have 54 recorded model request objects. Every recorded request matches its frozen bundle and replayed runtime request. Candidate replay independently records 49 accepted replies, five native refusals, and six deterministic passes with no failed deliveries. The legacy baseline replay uses an ASCII-escaping hash serializer; its Unicode `g08_direct_practical` request therefore has a different canonical hash from the newer serializer despite exact object equality. This was traced to `replay_frozen_pipeline.py:60`, not treated as a request mismatch.
- None of the 54 baseline or 54 candidate raw controlled envelopes carries an upstream transport body hash. Their exactness claim is equality between retained request objects and local runtime replay, with hashes of retained observation files; it is not an independent verification of the original wire bytes. Seven later teacher attempts do carry a gateway request hash. The supplemental report accurately labels its eight older-gateway captures as reconstruction rather than exact transport proof.
- All 16 retained fresh holdout judgments match the exact captured outputs and response scores. Fifteen belong to the twenty expected quality cases; one is a safety judgment outside the quality denominator. The mean remains +1.066667/5, with eight candidate preferences and seven ties. The 100% win fraction is eight of eight non-tied judged quality pairs, not twenty cases. Naturalness remains +0.2 on the same fifteen quality pairs. Missing cases are visible and no partial score passes a release gate.
- Six delivered holdout pairs lack judgments: `r15_uncertain_effect`, `r20_risky_activity_push`, and journal cases `r21`–`r24`. The five quality omissions exclude safety case `r20`. Native-refusal cases `r17` and `r19` are not scored as delivered language. `r15` retains four actual 429 attempts; this audit made no retry.
- Provider-native refusal takes precedence over any accompanying content; refusals remain distinct from useful delivered replies. Rejected output in the original v2 walking case stays diagnostic and produces no assistant bubble. Baseline unsupported capabilities stay distinct from model failures and cannot improve the language-quality average.
- The original and latest report manifests match their exact HTML hashes. All 51 original and 81 search-fix package-manifest entries matched retained file bytes. Every latest release-gate evidence reference matched its recorded SHA-256. At audit intake all seven candidate-freeze hashes and all 198 upload/source-freeze entries matched the then-current files. Subsequent audit changes intentionally make those historical freezes inapplicable to the new candidate; they must not be described as current validation.
- Existing blocked-release evidence binds the historical source/migration hash and reports no deployment, migration, or alias change. This audit made no external request to reverify hosted state. The recorded 617 backend and 16 browser integration results are prior-source checks; root's integrated audit verification owns the new totals.
- Review storage uses a hash of dataset plus compared cases, rather than render time or dataset alone. The corrected report gets a new review ID, so old reviews do not silently attach to changed evidence. The downloaded review carries the dataset and report identity and stays local. Version hiding removes summary, version provenance, and capability labels; it does not make a previously seen report or structurally different outputs a blinded human study.
- `scripts/evaluate_luna.py` and its legacy helper remain explicitly structural/offline: semantic quality is `unreviewed`, and they are not used to establish the guided-action release gate. No paid runner was executed.

## Validation and use

The focused legacy/guided evaluation tests pass, with Ruff and mypy passing on the changed evaluator. [audit-comparison-browser-verification.json](audit-comparison-browser-verification.json) binds the exact new report bytes and records Chromium under Node 22.23.3: 60 cards, eight adaptive sections, 24 holdout cards, eight journal cards, correct refusal display, historical notice visible even when labels are hidden, scoped persistence/export, no horizontal mobile overflow, no JavaScript errors, and no external requests. The synthetic browser note was cleared afterward.

To rebuild only the new historical audit artifact offline:

```sh
cd /workspace/JournalPulse
.venv/bin/python /workspace/journalpulse-planning/pre-evaluation-audit-2026-10-05/build_audit_comparison.py
/workspace/.tools/node22/node_modules/node/bin/node /workspace/journalpulse-planning/pre-evaluation-audit-2026-10-05/check_audit_report.cjs
```

These commands generate no model judgments and cannot establish the current candidate's language quality. Future paid evaluation needs a new candidate freeze, the appropriate holdout policy, complete bound teacher judgments, complete adaptive evidence, and authorization consistent with the unchanged attempt and spend limits. The existing run cannot be completed merely by using the remaining nine teacher attempts: the retained protocol estimates at least seventeen operations without errors, and `r15` has already exhausted its declared retry allowance.
