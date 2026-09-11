import copy
import json
import threading
from collections import Counter

import pytest

from data.sglang_backlog import MODEL, REVISION, IMAGE, sha
from data.sglang_backlog_fast import (
    checked_rows, prepare_source, teacher_messages, TEACHER_GUIDANCE, BILLSUM_GUIDANCE,
    recover_attempts, continuous_queue, validate_source_decision, validate_serving_evidence,
)
from data.sglang_27b_serving import server_command


def task(index):
    return {'task_id': f't{index}', '_attempt_provenance': {'kind': 'retry_rejected'},
            'source_row_id': str(index), 'payload': 'original evidence'}


def write_chunk(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = b''.join((json.dumps(row) + '\n').encode() for row in rows)
    path.write_bytes(raw)
    return {'path': str(path), 'rows': len(rows), 'sha256': sha(raw), 'name': path.name}


def test_referenced_hash_and_exclusions_are_enforced(tmp_path):
    spec = write_chunk(tmp_path / 'rows.jsonl', [task(0), task(1)])
    spec.update(rows=1, exclude_task_ids=['t0'], task_ids_sha256=sha(json.dumps(['t1']).encode()))
    assert list(checked_rows(spec)) == [task(1)]
    with pytest.raises(ValueError):
        list(checked_rows({**spec, 'exclude_task_ids': ['absent']}))
    (tmp_path / 'rows.jsonl').write_text(json.dumps(task(0)) + '\n')
    with pytest.raises(ValueError):
        list(checked_rows(spec))


def test_probe_plus_referenced_continuation_covers_every_id_once(tmp_path):
    parent = tmp_path / 'parent'
    rows = [task(i) for i in range(70)]
    specs = [write_chunk(parent / 'inputs' / 'finqa' / f'chunk-{i}.jsonl', batch)
             for i, batch in enumerate([rows[:64], rows[64:]])]
    source = {'source': 'finqa', 'rows': 70, 'counts': {'retry_rejected': 70},
              'chunks': specs, 'retained_prior_outputs': []}
    report = prepare_source(source, parent, tmp_path / 'output', {'old-accepted', 'old-pilot'})
    assert report['probe_rows'] == 64
    assert sum(c['rows'] for c in report['chunks']) == 70
    effective = [row for spec in report['chunks'] for row in checked_rows(spec)]
    assert Counter(r['task_id'] for r in effective) == Counter(r['task_id'] for r in rows)
    assert all(row == rows[int(row['task_id'][1:])] for row in effective)
    ledger = list(checked_rows(report['coverage']))
    assert len(ledger) == 70
    assert {r['task_id'] for r in ledger} == {r['task_id'] for r in rows}
    assert prepare_source(source, parent, tmp_path / 'output', {'old-accepted', 'old-pilot'}) == report


def test_probe_rejects_pilot_or_accepted_overlap(tmp_path):
    parent = tmp_path / 'parent'
    spec = write_chunk(parent / 'inputs' / 'finqa' / 'chunk.jsonl', [task(0)])
    source = {'source': 'finqa', 'rows': 1, 'counts': {'retry_rejected': 1},
              'chunks': [spec], 'retained_prior_outputs': []}
    with pytest.raises(ValueError, match='overlaps'):
        prepare_source(source, parent, tmp_path / 'output', {'t0'})


def test_zero_pending_source_is_preserved(tmp_path):
    source = {'source': 'watsonx_docs_qa', 'rows': 0, 'counts': {}, 'chunks': [], 'retained_prior_outputs': []}
    report = prepare_source(source, tmp_path, tmp_path / 'output', set())
    assert report['rows'] == report['probe_rows'] == 0
    assert report['chunks'] == []
    assert list(checked_rows(report['coverage'])) == []


def test_teacher_canary_changes_only_request_copy_and_never_judge():
    messages = [{'role': 'system', 'content': 'original teacher'}, {'role': 'user', 'content': 'question'}]
    original = copy.deepcopy(messages)
    request = teacher_messages(messages, 'billsum', 'rollout')
    assert TEACHER_GUIDANCE in request[0]['content'] and BILLSUM_GUIDANCE in request[0]['content']
    assert messages == original
    assert teacher_messages(messages, 'billsum', 'judge') == original
    assert BILLSUM_GUIDANCE not in teacher_messages(messages, 'finqa', 'rollout')[0]['content']


IDENTITY = {'function_call_id': 'fc-same', 'input_id': 'in-same'}
BINDING = {'input': 'hash', 'settings': 'frozen'}


def journal_row(row, attempt=1, identity=None):
    return {'task_id': row['task_id'], 'attempt': attempt,
        'task_sha256': sha(json.dumps(row, sort_keys=True).encode()),
        'identity': identity or IDENTITY, 'binding': BINDING}


def owner(finished=False):
    return {'identity': IDENTITY, 'binding': BINDING, 'finished': finished}


def recover(rows, journal, results=(), previous=None, identity=None):
    return recover_attempts(rows, journal, results, previous_owner=previous,
                            identity=identity or IDENTITY, binding=BINDING)


def test_same_input_restart_retries_only_missing_result_with_fresh_attempt():
    rows = [task(0), task(1), task(2)]
    result = {**journal_row(rows[0]), 'verification': {'accepted': False}}
    state = recover(rows, [journal_row(rows[0]), journal_row(rows[1])], [result], previous=owner())
    assert state['status'] == 'ready'
    assert [(r['task']['task_id'], r['attempt']) for r in state['pending']] == [('t1', 2), ('t2', 1)]
    assert state['interrupted_attempts'][0]['task_id'] == 't1'
    assert set(state['committed']) == {'t0'}


@pytest.mark.parametrize('previous,identity', [
    (None, IDENTITY), (owner(True), IDENTITY), (owner(), {'function_call_id': 'other', 'input_id': 'in-same'}),
    (owner(), {'function_call_id': 'fc-same', 'input_id': 'other'}),
])
def test_unclassified_or_other_input_cannot_retry(previous, identity):
    state = recover([task(0)], [journal_row(task(0))], previous=previous, identity=identity)
    assert state['status'] == 'needs_reconciliation'
    assert state['pending'] == [] and state['unresolved_ids'] == ['t0']


def test_exhausted_attempts_are_visible_and_never_completed():
    state = recover([task(0)], [journal_row(task(0), i) for i in (1, 2, 3)], previous=owner())
    assert state['status'] == 'exhausted' and state['exhausted_ids'] == ['t0']
    assert state['committed'] == {} and state['pending'] == []


def test_changed_or_duplicate_attempt_journal_is_rejected():
    row = journal_row(task(0))
    for bad in [[row, row], [{**row, 'task_sha256': 'changed'}], [{**row, 'attempt': 4}]]:
        with pytest.raises(ValueError):
            recover([task(0)], bad, previous=owner())


def test_continuous_queue_refills_before_slowest_prior_task_finishes():
    release, saved, reserved, sequence = threading.Event(), [], set(), []
    def reserve(batch):
        reserved.update(t['task_id'] for t in batch)
    def run(row, replica):
        assert row['task_id'] in reserved
        sequence.append(('start', row['task_id']))
        if row['task_id'] == 't0':
            assert release.wait(5), 'Queue waited for the slow prior task before refilling'
        if row['task_id'] == 't2':
            release.set()
        sequence.append(('end', row['task_id']))
        return {'task_id': row['task_id'], 'replica': replica}
    report = continuous_queue([task(i) for i in range(4)], run=run, reserve=reserve, save=saved.append,
        stop=lambda _: False, concurrency=2, replicas=2, reservation_block=2)
    assert report['completed'] == 4
    assert sequence.index(('start', 't2')) < sequence.index(('end', 't0'))
    assert {r['task_id'] for r in saved} == reserved == {f't{i}' for i in range(4)}


def test_worker_failure_drains_other_submitted_results_without_retry():
    saved = []
    def run(row, replica):
        if row['task_id'] == 't0':
            raise RuntimeError('infrastructure')
        return row
    report = continuous_queue([task(i) for i in range(6)], run=run, reserve=lambda _: None,
        save=saved.append, stop=lambda _: False, concurrency=2, replicas=2, reservation_block=2)
    assert report['stopped'] and report['worker_or_persistence_errors'][0]['task_id'] == 't0'
    assert [r['task_id'] for r in saved] == ['t1']
    assert report['never_reserved_ids'] == ['t2', 't3', 't4', 't5']


def decision_fixture():
    manifest = {'sources': [{'source': 'finqa', 'rows': 100, 'probe_rows': 64,
                            'chunks': [{'sha256': 'input'}]}]}
    decision = {'scope': 'reviewed_sources', 'probe_decision_sha256': 'probe-decision',
        'backlog_manifest_sha256': 'manifest', 'source_reviews': {'finqa': {
            'approved_for_generation': True, 'evidence_review': 'Reviewed source evidence',
            'probe_report_sha256': 'report'}}}
    probes = {'finqa': {'sha256': 'report', 'report': {'source': 'finqa', 'chunk': 0,
        'status': 'complete', 'rows': 64, 'completed': 64, 'input_sha256': 'input',
        'backlog_manifest_sha256': 'manifest', 'decision_sha256': 'probe-decision'}}}
    return manifest, decision, probes


def test_source_continuation_requires_exact_probe_binding():
    manifest, decision, probes = decision_fixture()
    assert validate_source_decision(decision, manifest, stage='continuation',
        base_decision_sha='probe-decision', probe_reports=probes) == {'finqa'}
    with pytest.raises(ValueError):
        validate_source_decision(decision, manifest, stage='continuation', probe_reports=probes)
    for field, value in [('rows', None), ('completed', None), ('rows', 0), ('completed', 63),
                         ('input_sha256', 'changed'), ('backlog_manifest_sha256', 'changed')]:
        broken = copy.deepcopy(probes)
        broken['finqa']['report'][field] = value
        with pytest.raises(ValueError):
            validate_source_decision(decision, manifest, stage='continuation',
                base_decision_sha='probe-decision', probe_reports=broken)


def test_smoke_evidence_requires_native_parser_observation():
    manifest = {'model': MODEL, 'model_revision': REVISION, 'image': IMAGE}
    report = {'status': 'complete_serving_smoke', 'total_replayed_requests': 128,
        'manifest_sha256': 'manifest', 'all_requests': {'counts': {'successful_http_and_capture': 128}},
        'eight_replica_wave': {'requests_per_second': 18}}
    kwargs = dict(report_sha='report', manifest_sha='manifest', requested_report_sha='report')
    with pytest.raises(ValueError):
        validate_serving_evidence(report, manifest, **kwargs)
    report['all_requests']['counts']['valid_native_call'] = 128
    validate_serving_evidence(report, manifest, **kwargs)


def test_replica_command_matches_validated_smoke():
    command = server_command(7)
    assert command[command.index('--port') + 1] == '8007'
    assert command[command.index('--tp') + 1] == '1'
    assert '--disable-cuda-graph' not in command
    assert command[command.index('--max-running-requests') + 1] == '16'
    assert command[command.index('--max-mamba-cache-size') + 1] == '80'
