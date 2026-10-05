"""Verify retained teacher inputs and render historical evidence without paid calls."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path('/workspace/JournalPulse')
SOURCE = Path('/workspace/journalpulse-planning/guided-action-search-fix-2026-10-05')
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))
from journalpulse.guided_evaluation import (
    Observation, build_comparison, digest, judge_pair_binding, load_dataset,
    write_comparison_report,
)

sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
read = lambda path: json.loads(path.read_text())

def main() -> None:
    original = read(SOURCE / 'comparison-data.json')
    dataset = load_dataset(SOURCE / 'dataset.json')
    assert digest(dataset.model_dump()) == original['dataset_sha256']
    cases = {case.id: case for case in dataset.cases}
    before = {'metadata': original['baseline_metadata'], 'cases': [r['baseline'] for r in original['cases']]}
    after = {'metadata': original['candidate_metadata'], 'cases': [r['candidate'] for r in original['cases']]}
    bindings, judgments = [], []
    for row in original['cases']:
        if row['judge'] is None:
            continue
        record = deepcopy(row['judge'])
        case = cases[record['id']]
        judge_file = SOURCE / 'candidate-controlled-holdout-judgments' / f'{case.id}.json'
        assert read(judge_file) == record
        provenance = record['provenance']
        body = provenance['actual_provider_request_body']
        assert digest(body) == provenance['actual_provider_request_sha256']
        payload = json.loads(body['messages'][-1]['content'])
        assert payload['case'] == case.model_dump()
        for label, version in [('A', record['a_version']), ('B', record['b_version'])]:
            participant = json.loads(payload[label])
            observation = row[version]
            raw = read(SOURCE / f'{version}-observations' / f'{case.id}.json')
            assert raw['request_sha256'] == digest(raw['request'])
            assert observation['provenance']['request_sha256'] == raw['request_sha256']
            assert observation['output'] == json.loads(raw['content'])
            assert participant['full_structured_output'] == observation['output']
            transcript = observation['messages'] or [message.model_dump() for message in case.messages] + [{
                'role': 'assistant',
                'content': observation['output'].get('reply', json.dumps(observation['output'], ensure_ascii=False)),
            }]
            assert participant['transcript'] == transcript
        attempt = read(judge_file.with_name(f'{case.id}.attempt{provenance["attempt"]}.json'))
        assert attempt['actual_provider_request'] == body
        assert attempt['a_version'] == record['a_version'] and attempt['b_version'] == record['b_version']
        result = attempt['provider_result']
        assert result['status'] == 'observed' and not result.get('refusal')
        assert result['finish_reason'] not in {'length', 'content_filter'}
        parsed = json.loads(result['content'])
        for key in ['scores', 'evidence', 'preference', 'uncertainty', 'rationale', 'critical_gates']:
            assert parsed[key] == record[key]
        transport_hash = result.get('actual_provider_request_sha256')
        if transport_hash:
            assert transport_hash == digest(body)
        binding = judge_pair_binding(case, Observation.model_validate(row['baseline']), Observation.model_validate(row['candidate']))
        record['provenance']['evaluation_binding'] = binding
        record['provenance']['binding_verification'] = 'Offline verification against full retained teacher body and response; no new judgment.'
        judgments.append(record)
        bindings.append({
            'id': case.id, 'original_judgment_sha256': sha(judge_file),
            'teacher_request_sha256': digest(body), 'evaluation_binding': binding,
            'transport_identity': 'gateway_hash_verified' if transport_hash else 'retained_request_reconstruction_only',
        })
    replay = read(SOURCE / 'candidate-pipeline-replay.json')
    software_gates = {}
    for proof in replay['cases']:
        status, runtime = proof['status'], proof.get('runtime_gate')
        if status == 'deterministic_pass' and runtime == 'pass':
            software_gates[proof['id']] = 'pass'
        elif status in {'accepted', 'provider_refusal'}:
            assert proof['request_comparison']['exact_request_match'] is True
            assert runtime in {'pass', 'pass_native_refusal_handled'}
            software_gates[proof['id']] = 'pass'
        else:
            raise ValueError(f'Unexpected historical replay state: {proof["id"]}')
    notice = (
        'Historical response audit: these are the previously captured guided-action-2026-10-05.2 '
        'Luna responses, re-evaluated only for evidence integrity. They are not fresh language '
        'evidence for the current candidate. No new Luna, teacher, or Brave calls were made; '
        'the current candidate remains unevaluated and release remains blocked.'
    )
    report = build_comparison(
        dataset, before, after, {'cases': judgments, 'metadata': {'binding_verification_count': len(bindings)}},
        metadata={
            'evidence_notice': notice,
            'evidence_scope': 'historical_captured_responses_only',
            'software_gates': software_gates,
            'source_comparison_sha256': sha(SOURCE / 'comparison-data.json'),
            'original_candidate_freeze': read(SOURCE / 'candidate-freeze.json'),
            'current_evaluator_sha256': sha(ROOT / 'src/journalpulse/guided_evaluation.py'),
            'prior_run_metadata': original['metadata'],
            'new_provider_calls': 0,
            'bindings_method': 'Every bound pair was checked against the complete retained actual teacher request body and response. Nine bodies have reconstruction-only transport provenance; seven also carry a matching gateway transport hash.',
        },
        adaptive_baseline={'cases': [r['adaptive_comparison']['baseline'] for r in original['cases'] if 'adaptive_comparison' in r]},
        adaptive_candidate={'cases': [r['adaptive_comparison']['candidate'] for r in original['cases'] if 'adaptive_comparison' in r]},
        adaptive_judgments={'cases': []},
    )
    report['limitations'].insert(0, notice)
    report['limitations'].insert(1, 'A/B version names are hidden on request; this is an inspectable offline report, not a blinded human study. Structured capability differences can still reveal a version.')
    assert report['summary']['heldout_primary'] == original['summary']['heldout_primary']
    assert report['summary']['naturalness_paired'] == original['summary']['naturalness_paired']
    assert report['summary']['release_gate_status'] == 'incomplete'
    assert report['summary']['unresolved_critical_cases'] == ['s05_physical_warning']
    for name, value in [('captured-judgment-bindings.json', bindings), ('audit-corrected-prior-response-comparison.json', report)]:
        (OUT / name).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    write_comparison_report(report, OUT / 'audit-corrected-prior-response-comparison.html')
    print(json.dumps({'judgments_verified':len(bindings), 'summary':report['summary'], 'provider_calls':0}, indent=2))

if __name__ == '__main__':
    main()
