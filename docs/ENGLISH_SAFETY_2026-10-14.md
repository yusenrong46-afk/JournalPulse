# English safety regression evidence

Scope: the existing deterministic English phrase rules. No classifier or
multilingual support was added. Public English-only notices and detection limits
remain visible. Support routing continues to bypass the model/provider path.

The assistant-authored development set is
`assets/evaluation/safety/english-development.json`: 60 fictional cases covering
direct and indirect danger, slang, typos, negation, historical statements,
quotations and ordinary frustration. Of these, 52 have explicit regression
expectations and 8 remain labelled `review`; ambiguous cases are not counted as
passes or failures and are not silently used as new rule requirements.

Before repair: 33 clear cases passed, 13 were missed and 6 falsely triggered.
After repair: all 52 clear cases passed; the 8 review cases remain unresolved.
The changes add bounded intent/planning and “unalive myself” phrases and narrowly
scope denials such as “no thoughts of suicide” and “not currently suicidal”. They
do not broadly match harmless words. Existing independent-clause and historical
handling stays in place.

`tests/test_english_safety_entrypoints.py` adds 24 checks across chat, journal
reflection, standalone reflection, discovery and activity feedback. These assert
immediate support without provider calls for clear danger, normal flow for clear
denials, and existing consent boundaries. Existing safety suites are retained.

Before/after results and test output are under
`artifacts/reliability-2026-10-14/english-*`. This is a defect-focused development
set authored during the fixes. Passing it is **not general safety accuracy**, an
independent red-team certification, medical assessment, or proof that slang,
misspellings and indirect distress will always be detected. Human review of the
ambiguous cases and real-world generalization remain unestablished.
