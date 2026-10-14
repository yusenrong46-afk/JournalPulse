# Trust hardening — October 14, 2026

This is a local repair candidate based on `7be3bf8`, not a deployment record.
Historical release snapshots and benchmark outputs are preserved.

## Public information

The sign-in screen now links to public privacy, terms, and data/AI pages and shows
Canadian crisis support. Information pages do not initialize or depend on authentication.
Private routes retain account isolation and sign-in requirements. Public disclosure
copy describes current processing and known retention limits; operator contact details
still need to be supplied before publication. No legal-compliance certification is claimed.

## Safety routing

**Scope update, October 14:** the user chose English-only support instead of the
proposed multilingual classifier. Sign-in, onboarding, the chat header, Settings,
and the public data/AI page now disclose that limitation. No classifier was added
or enabled and no provider call was made for this change. The multilingual test
cases remain as recorded unsupported-language gaps, not an active delivery target.
English phrase checks can still miss distress or produce false alarms. The
classifier proposal below is retained as background, not current scope.

The development set in `assets/evaluation/safety/development.json` contains 34 fictional,
assistant-authored cases. It is not a held-out evaluation or clinical annotation.
The offline runner records misses, false alarms, and unresolved review cases separately:

```sh
python scripts/evaluate_safety.py --output artifacts/safety-development.json
```

It deliberately exits nonzero while misses, false alarms, or review cases remain.
The first run had 19 passing cases, 9 missed-support cases, 4 false alarms, and 2
review cases. Bounded spelling/Unicode normalization and phrase-scoped denials changed
that to 29 passing cases, 3 missed-support cases, and 2 review cases. This measures
these development examples only. The multilingual positive cases still fail.

**Not fixed:** general language understanding and context-sensitive recovery from false
support routing. No unvalidated classifier has been enabled. A second layer must first
be tested on separately authored, human-reviewed cases, including language-specific
misses, false alarms, quotations, historic statements, adversarial instructions,
timeouts, and malformed outputs. A model must not silently override a clear rule hit.

Before adding a remote classifier, define its privacy/consent behavior for Simple Luna,
which currently makes no model calls; select the model and a bounded evaluation budget.
For uncertain classifier outputs, design an explicit clarification state rather than
converting every uncertainty into a permanent support lock. Preserve the existing
support path until that change has reviewed behavior and integration evidence.

## Evaluation and project claims

Current runtime prompt: `guided-action-2026-10-06.2`. Historical evaluated prompt:
`guided-action-2026-10-05.3`. The October 5 comparison does not qualify today's candidate.
Its final-test result was 8 wins, 5 losses, and 7 ties: 61.5% among 13 non-ties,
not a 61.5% success rate across 20 cases. The primary-gain interval includes zero.
The related model judge and fictional scenarios are additional limits.

A future comparison needs a frozen candidate, fresh held-out cases, a predeclared
sample-size/precision plan, blinded independent human review, and separate critical
safety gates. Do not tune against this development set or rerun until a pass appears.
No fresh paid evaluation or human qualification has occurred in this repair.

Resume wording supported by the project:

> Built and deployed JournalPulse, a research beta for private journaling and optional
> AI-assisted reflection, with account isolation, consent controls, and automated
> regression checks.

Use only responsibilities the author actually performed or directed/reviewed. The
README's October 7 counts are dated historical software checks; they are not model
accuracy, user-benefit results, or new checks of this candidate. No resume file was
found in this checkout, so no external resume has been changed.

## Recovery and key rotation

A backup is not verified recovery. A local synthetic dump/restore exercise, if passed,
only verifies the local procedure. Hosted backup retention, access, point-in-time
recovery and restoration into a separate hosted project remain operational checks.
Never restore an old database over the live service to test recovery.

After a real recovery, reconcile deletions since the backup before reopening access.
A restored deletion-marker table alone cannot protect deletions that happened after
that backup. Keep post-backup deletion requests available through a protected recovery
process, or keep the restored service closed until they have been reconciled.

## Browser security

Exported HTML receives a CSP bound to its actual inline bootstrap scripts. Script
attributes, object embedding, framing and foreign form submissions are blocked. Styles
retain inline support for the current UI. Supabase is the explicit extra connection
origin; provider API keys and AI traffic remain server-side. This applies to the
FastAPI-served static export. Next.js development is a separate server and does not
receive that response policy.

Production disables `/docs`, `/redoc`, and `/openapi.json`; local API documentation
and programmatic schema generation remain available. Authentication still protects
API operations independently of whether documentation is visible.

## Local verification results

- 1,305 backend tests pass; 92.03% measured branch coverage.
- 225 frontend unit tests pass, including auth outages, public-to-private navigation,
  and existing account-switch isolation.
- Production static export, TypeScript, lint, Ruff, mypy, and OpenAPI consistency pass.
- 16 focused desktop/mobile browser and accessibility checks pass. Static-export
  navigation and no-JavaScript disclosures also pass.
- The real static-serving CSP allows normal navigation and blocks an injected script.
- Fifteen isolated PostgreSQL recovery/rotation assertions pass; hosted backup and
  post-backup deletion reconciliation remain explicitly unverified.

Evidence is under `artifacts/trust-hardening-2026-10-14/`. Test totals describe this
candidate only and must not be presented as AI quality or clinical validation.
The new migration has only been applied to the disposable local test database.
No live keys, hosted data, provider configuration, or deployment were changed.

The existing database schema, journal ownership, signed usage, activity lifecycle,
Sentinel, and erasure regression scripts also pass against the new migration in the
network-isolated PostgreSQL container. These use the repository's stand-in auth schema.

Draft PRs #5, #6 and #7 were verified against live GitHub main `7be3bf8`: each exact
head commit was already an ancestor of main. Those redundant drafts were closed, preserving
their branches and review history. This did not merge or deploy this repair candidate.

The first isolated browser run used the wrong API port for two existing chat-dialog
tests. After correcting the test-server configuration, all 16 checks passed. The
initial failure log is retained. No application change was made to hide that failure.

English-only disclosure follow-up: 16 focused frontend unit tests, TypeScript, lint,
and the production static export pass. Earlier full-suite totals above belong to the
preceding repair snapshot. This follow-up changes supported-language notices only;
it does not add language detection, reject non-English text, or qualify the safety router.
