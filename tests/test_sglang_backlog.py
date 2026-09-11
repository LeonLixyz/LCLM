import copy
import json

import pytest

from data.sglang_backlog import (
    EXPECTED_PENDING, attempt_state, circuit_breaker, file_sha, generation_config,
    prepare_source, rollout_one, select_pending, sha, validate_scale_decision,
)


def task(key, kind='retry_rejected'):
    return {'task_id': key, '_attempt_provenance': {'kind': kind}}


def test_selection_excludes_every_pilot_outcome_and_keeps_unattempted():
    rows = [task('pilot'), task('retry'), task('new', 'previously_unattempted')]
    assert [r['task_id'] for r in select_pending(rows, {'pilot'})] == ['retry', 'new']
    with pytest.raises(ValueError, match='Duplicate'):
        list(select_pending(rows + [rows[0]], {'pilot'}))
    with pytest.raises(ValueError, match='absent'):
        list(select_pending(rows, {'missing'}))
    with pytest.raises(ValueError, match='not a prior failure'):
        list(select_pending(rows, {'new'}))
    with pytest.raises(ValueError, match='neither failed nor unattempted'):
        list(select_pending([task('accepted', 'previously_accepted')], set()))


def test_attempt_resume_surfaces_unknown_without_retry_or_false_completion():
    rows = [task('done'), task('unknown'), task('new')]
    journal = [{'task_id': r['task_id'], 'task_sha256': sha(json.dumps(r, sort_keys=True).encode())} for r in rows[:2]]
    result = attempt_state(rows, journal, [{'task_id': 'done'}])
    assert result['unresolved'] == ['unknown']
    assert [r['task_id'] for r in result['pending']] == ['new']
    assert set(result['completed']) == {'done'}
    with pytest.raises(ValueError):
        attempt_state(rows, journal + journal[:1], [])
    with pytest.raises(ValueError):
        attempt_state(rows, journal, [{'task_id': 'new'}])


def gate_fixture():
    ids = {str(i) for i in range(1000)}
    manifest = {'pending_counts': EXPECTED_PENDING, 'generation_config': {'fixed': True}}
    report = {'completed': 1000, 'status': 'complete_diagnostic_pending_review',
              'manifest_sha256': 'inputs', 'results_sha256': 'results'}
    bindings = dict(backlog_sha='backlog', pilot_manifest_sha='inputs', pilot_results_sha='results', pilot_report_sha='report')
    decision = {'decision': 'generate_failed_and_unattempted_with_27b', 'reviewed_by': 'root',
        'approved_for_generation': True, 'approved_for_release': False,
        'backlog_manifest_sha256': 'backlog', 'pilot_manifest_sha256': 'inputs',
        'pilot_results_sha256': 'results', 'pilot_report_sha256': 'report',
        'generation_config': manifest['generation_config'],
        'evidence_review': {key: 'Reviewed evidence and limitations.' for key in
                           ('recovery', 'speed', 'quality', 'parser_failures', 'limitations')}}
    decision['evidence_review']['reviewed_task_ids'] = ['0']
    return decision, manifest, report, ids, [{'task_id': key} for key in ids], bindings


def test_scale_gate_binds_completed_evidence_and_root_review():
    decision, manifest, report, ids, rows, bindings = gate_fixture()
    validate_scale_decision(decision, manifest, report, ids, rows, **bindings)
    for key, value in [('reviewed_by', 'self_judge'), ('approved_for_generation', False),
                       ('pilot_results_sha256', 'changed'), ('approved_for_release', True)]:
        broken = {**decision, key: value}
        with pytest.raises(ValueError):
            validate_scale_decision(broken, manifest, report, ids, rows, **bindings)
    with pytest.raises(ValueError):
        validate_scale_decision(decision, manifest, {**report, 'completed': 999}, ids, rows, **bindings)
    broken = copy.deepcopy(decision)
    broken['evidence_review'].pop('quality')
    with pytest.raises(ValueError, match='quality'):
        validate_scale_decision(broken, manifest, report, ids, rows, **bindings)
    with pytest.raises(ValueError):
        validate_scale_decision(decision, manifest, report, ids, rows[:-1], **bindings)


def test_prepare_source_hashes_parent_and_excludes_pilot(tmp_path, monkeypatch):
    import data.sglang_backlog as backlog
    source, pilot, destination, previous = [tmp_path / x for x in ('source', 'pilot', 'dest', 'previous')]
    source.mkdir()
    previous.mkdir()
    rows = [task('pilot'), task('retry'), task('new', 'previously_unattempted')]
    path = source / 'finqa.tasks.jsonl'
    path.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    parent = {'source': 'finqa', 'previous_root': str(previous), 'parent_audit': {'files': []},
        'counts': {'retry_rejected': 2, 'previously_unattempted': 1},
        'prepared_files': {path.name: file_sha(path)}}
    (source / 'finqa.build.json').write_text(json.dumps(parent))
    (source / 'retry-manifest.json').write_text(json.dumps({'sources': [parent]}))
    monkeypatch.setattr(backlog, 'pilot_selection', lambda root: ({}, {'pilot': 'finqa'}, 'pilot-hash'))
    report = prepare_source('finqa', source, pilot, destination, tmp_path / 'unused-split')
    assert report['rows'] == 2
    assert report['excluded_pilot_ids'] == ['pilot']
    assert report['counts'] == {'retry_rejected': 1, 'previously_unattempted': 1}
    assert prepare_source('finqa', source, pilot, destination, tmp_path / 'unused-split') == report
    chunk = destination / 'finqa' / report['chunks'][0]['name']
    chunk.write_text(chunk.read_text().replace('retry', 'xxxxx'))
    with pytest.raises(ValueError, match='chunk changed'):
        prepare_source('finqa', source, pilot, destination, tmp_path / 'unused-split')


def synthetic_case():
    from data.synthetic_expansion_agent import generate_task
    row = generate_task(seed=11, index=0)
    row['_attempt_provenance'] = {'kind': 'previously_unattempted'}
    return row


def test_synthetic_uses_exact_verifier_and_strips_thinking():
    row = synthetic_case()
    calls = 0
    def request(messages, tools, phase):
        nonlocal calls
        assert phase == 'rollout'  # No same-model judge for procedural synthetic.
        calls += 1
        if calls == 1:
            return {'content': '<think>hidden</think> I will look.', 'tool_calls': [
                {'id': str(i), 'type': 'function', 'function': {'name': 'expand', 'arguments': {'segment_id': segment}}}
                for i, segment in enumerate(row['support_segment_ids'])]}
        return {'content': '<think>hidden</think> Preamble\n' + row['expected_final']}
    result = rollout_one(row, 'synthetic', request, {'fixture': True})
    assert result['verification']['accepted'] is True
    assert result['approved_for_release'] is False
    messages = result['trace']['messages']
    assert messages[-1]['content'] == row['expected_final']
    assert all(not m.get('content') for m in messages[:-1] if m['role'] == 'assistant')
    assert '<think>' not in json.dumps(messages)


def test_truncation_stays_distinct_from_invalid_tool_call():
    from data.expansion_retry_diagnostic import TruncatedResponseError
    def request(*args):
        raise TruncatedResponseError('length')
    result = rollout_one(synthetic_case(), 'synthetic', request, {})
    assert result['verification']['accepted'] is False
    assert result['error']['kind'] == 'truncation'
    assert 'invalid_assistant_response' not in result['verification']['reason']


def test_circuit_breaker_separates_infrastructure_and_zero_yield():
    failed = {'verification': {'accepted': False, 'reason': 'missing_support'}}
    assert circuit_breaker([failed] * 63) is None
    assert circuit_breaker([failed] * 64) == 'zero_automatic_acceptance_after_64'
    good = {'verification': {'accepted': True, 'reason': 'accepted'}}
    infrastructure = {**failed, 'error': {'phase': 'rollout', 'kind': 'infrastructure'}}
    assert circuit_breaker([good] * 12 + [infrastructure] * 4).startswith('infrastructure')


def test_generation_configuration_has_fixed_non_thinking_model_and_hashes():
    config = generation_config()
    assert config['sampling']['extra_body']['chat_template_kwargs']['enable_thinking'] is False
    assert config['model'] == 'Qwen/Qwen3.8-27B'
    assert config['max_tool_calls'] == 16
    assert config['approved_for_release'] is False
    assert len(config['code_sha256']) >= 10
