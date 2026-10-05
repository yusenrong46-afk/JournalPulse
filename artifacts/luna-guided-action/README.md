# Luna guided-action implementation review

Implemented locally; release blocked. The existing Preview and shared Supabase
remain unchanged. One held-out walking-search proposal was rejected by the real
parser. One adaptive teacher judgment exhausted bounded rate-limit retries.

Open `comparison.html` in a browser. It is standalone and needs no network access.
Use the filters, inspect each Before/After transcript, or hide version names for
blind review. Review notes stay in your browser and can be exported as JSON.
Diagnostic rejected generations are not delivered assistant messages.

`step-diffs/UNIT_NOTES.md` explains the before/after changes. Ordered patches group
related edits; intermediate groups can depend on later shared contracts. The
combined patch compares against the complete 202-file dirty pre-upgrade workspace,
preserving inherited audit work. It is not a patch against clean Git HEAD.

`evidence/release-gates.json` records failures, incomplete measurements and passed
software checks separately. `blocked-release-validation.json` proves the release
helper rejects this candidate before any external mutation. Evaluation datasets
are fictional; results are teacher-model evidence, not a human usefulness study.
