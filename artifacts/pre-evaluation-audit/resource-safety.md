# Resource/search and deterministic safety audit

Initial read-only product audit, October 5, 2026. Reproductions used the current checkout,
temporary SQLite databases under `/tmp`, TestClient, and synthetic provider doubles.
No external requests, paid calls, or deployment were made. After reporting findings,
the parent authorized the narrowly scoped safety fix recorded below; other findings
remained read-only in this subtask.

## P1 — Every current inline search UI payload violates its API contract

- Source: `web/components/activity-session-discovery.tsx:62-70`, especially line 65.
- Contract: `src/journalpulse/inline_discovery.py:38-45`.
- The component spreads `activity_constraints` into the top-level request. The
  endpoint accepts a nested `constraints` object and rejects extra fields.
- A real newly created conversation already contains all five constraint fields,
  including their default false/null values. Thus this is not confined to people
  who requested special limits: the first search fails for the current API's
  ordinary conversation object.
- Expected: consenting to Search activities reaches the discovery service and
  displays offers or an explicit empty/provider result.
- Actual: HTTP 422, five `extra_forbidden` errors, zero discovery calls. Sending
  the same constraints under their declared nested key returns HTTP 200 and an
  offer. The `.2` categorical model-search fix does not address this UI boundary.

Run from `/workspace/JournalPulse`:

```bash
.venv/bin/python - <<'PY'
import runpy, tempfile
from pathlib import Path
from fastapi.testclient import TestClient
from journalpulse.api import create_app

ns = runpy.run_path('tests/test_inline_discovery.py')
with tempfile.TemporaryDirectory(prefix='audit-resource-', dir='/tmp') as d:
    stub = ns['DiscoveryDouble']()
    with TestClient(create_app(settings=ns['configured'](Path(d)), discovery_client=stub)) as client:
        chat = ns['start'](client)
        print('constraints:', chat['activity_constraints'])
        base = {'expected_revision': chat['revision'], 'llm_consent': True,
                'original_query': 'quiet meditation'}
        path = '/v1/conversations/' + chat['id'] + '/discover'
        actual = client.post(path, headers=ns['HEADERS'],
                             json={**base, **chat['activity_constraints']})
        print('UI-shaped:', actual.status_code, actual.json())
        print('provider calls:', len(stub.requests))
        control = client.post(path, headers=ns['HEADERS'],
                              json={**base, 'constraints': chat['activity_constraints']})
        print('nested control:', control.status_code,
              'offers:', len(control.json().get('offers', [])))
PY
```

Observed:

```text
constraints: {'time_minutes': None, 'no_audio': False, 'no_video': False, 'seated': False, 'avoid_breath_focus': False}
UI-shaped: 422 {'detail': [{'type': 'extra_forbidden', 'loc': ['body', 'time_minutes'], 'msg': 'Extra inputs are not permitted', 'input': None}, {'type': 'extra_forbidden', 'loc': ['body', 'no_audio'], 'msg': 'Extra inputs are not permitted', 'input': False}, {'type': 'extra_forbidden', 'loc': ['body', 'no_video'], 'msg': 'Extra inputs are not permitted', 'input': False}, {'type': 'extra_forbidden', 'loc': ['body', 'seated'], 'msg': 'Extra inputs are not permitted', 'input': False}, {'type': 'extra_forbidden', 'loc': ['body', 'avoid_breath_focus'], 'msg': 'Extra inputs are not permitted', 'input': False}]}
provider calls: 0
nested control: 200 offers: 1
```

## P1 — Bare “I don't feel safe” misses the intended safety pattern

- Source: `src/journalpulse/safety.py:10`; normalization and clause stripping at
  lines 43-45 and 62-64 make the missing match deterministic.
- The pattern has a required literal space before its optional time suffix:
  `feel safe (?:right now|tonight|today|alone)?`. A sentence ending in `safe` has
  no such space after normalization/clause stripping.
- Expected: the supported unsafe-feeling phrase routes to support with no model
  request whether or not a time suffix is supplied.
- Actual: `assess_safety("I don't feel safe.").mode == NORMAL`; adding
  `right now` changes it to SUPPORT. A synthetic normal-chat response can then
  set `ready_for_action=True`. This is a syntax bug in an intended explicit
  pattern, separate from documented indirect-language and historical-risk limits.

```bash
.venv/bin/python - <<'PY'
import runpy, tempfile
from pathlib import Path
from fastapi.testclient import TestClient
from journalpulse.api import create_app

ns = runpy.run_path('tests/test_conversations_api.py')
with tempfile.TemporaryDirectory(prefix='audit-safety-', dir='/tmp') as d:
    model = ns['ScriptedClient'](offers=[True])
    with TestClient(create_app(settings=ns['chat_settings'](Path(d)),
                               conversation_client=model)) as client:
        for text in ["I don't feel safe.", "I don't feel safe right now."]:
            chat = ns['start'](client)
            calls = len(model.calls)
            response = ns['say'](client, chat['id'], text)
            result = response.json()['conversation']
            print(text, response.status_code, result['safety_mode'],
                  result['ready_for_action'], len(model.calls) - calls)
PY
```

Observed columns: text, HTTP status, safety mode, ready for action, provider calls.

```text
I don't feel safe. 200 normal True 1
I don't feel safe right now. 200 support False 0
```

### Authorized fix and verification

Moved the whitespace into the optional suffix at `src/journalpulse/safety.py:10`.
No negation, historical/quoted-context handling, other risk vocabulary, prompt,
or model configuration was changed by this fix.

Added five unit cases in `tests/test_research_beta_safety.py` and one real-API
test in `tests/test_conversations_api.py` that covers straight/curly apostrophes,
no action readiness, the safety-router response, and zero model calls. All six
new cases failed before the fix. The focused existing and new suite then passed:

```text
.venv/bin/pytest -q tests/test_research_beta_safety.py tests/test_conversations_api.py -k 'safety or support or unsafe_feeling'
25 passed, 11 deselected
.venv/bin/ruff check src/journalpulse/safety.py tests/test_research_beta_safety.py tests/test_conversations_api.py
All checks passed!
```

The same synthetic API reproduction after the fix gives:

```text
I don't feel safe. 200 support False 0
I don’t feel safe. 200 support False 0
I don't feel safe right now. 200 support False 0
```

## P2 — Catalog format inference contradicts the catalog's video evidence

- Source: `src/journalpulse/activity_resources.py:195` infers `no_audio` and
  `no_video` solely from `resource_type != "video"`.
- `move_nhs_fitness_studio` is a `website`, with the title “NHS Fitness Studio
  Exercise Videos” and a summary explicitly describing guided workout videos.
  It is nevertheless assigned `no_audio=True`, `no_video=True`, and passes
  `ActivityConstraints(no_audio=True, no_video=True)`.
- Expected: hard format matching requires reviewed format metadata. A website
  container must not itself certify that the activity avoids video or audio.
- Local reproduction:

```python
from pathlib import Path
from journalpulse.activity_resources import (
    ActivityConstraints, activity_candidates, activity_resource_matches_constraints,
    resolve_activity_resource,
)
p = Path('assets/resources/catalog.json')
r = resolve_activity_resource(p, 'move_nhs_fitness_studio')
c = ActivityConstraints(no_audio=True, no_video=True)
print(r['title'], r['no_audio'], r['no_video'])
print(activity_resource_matches_constraints(r, c))
print(r['id'] in {r['id'] for r in activity_candidates(p, goal='move', constraints=c, limit=16)})
```

Observed: `NHS Fitness Studio Exercise Videos True True`, `True`, `True`.
Reachability caveat: normal guided context currently supplies the first sixteen
eligible resources without goal ranking; this particular later catalog item is
not in the initial unexcluded model pool. It can enter after earlier candidates
are excluded. The helper-level hard-constraint defect is independently reproducible.

## Other observations and known limits

- The inline refinement placeholder at
  `web/components/activity-session-discovery.tsx:113` recommends `shorter, seated,
  no video`, but the strict public vocabulary validator rejects commas. The
  textarea permits 600 characters while the inline API permits only 160. These
  are additional input-contract inconsistencies, currently masked by the P1
  request-shape defect.
- Initial review also found `I will kill myself tonight.` returned NORMAL.
  Root subsequently classified this direct-language miss as a material routing
  defect and authorized a bounded correction. `I will`/`I'll` kill-myself/end-my-life
  statements now route to support, clear matched denials remain normal, and
  separate affirmative intent survives a preceding denial. The 15 frozen cases
  produced 13 failures before fixing; the combined focused suite passes 39 tests.
  Positive API fixtures make zero model calls. See [ai.md](ai.md) and the preserved
  [before](direct-intent-before.txt)/[after](direct-intent-after.txt) logs. Indirect
  and quoted/historical interpretation limitations remain.
- Snippet offers deliberately fail every nontrivial hard constraint because the
  provider cannot verify duration or accessibility. Existing tests and release
  notes document that behavior; it is not a new finding.
- No reproducible private-query disclosure or receipt-signature bypass was found
  in the reviewed default production path. This is a bounded review result, not
  a proof of absence.
