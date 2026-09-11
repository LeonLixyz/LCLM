import copy
import json
from pathlib import Path
import pytest
from data.sglang_backlog import sha
from data.sglang_backlog_storage import (
    ORIGIN_MANIFEST_SHA256, ORIGIN_DECISION_SHA256, EXPECTED_REMAINING,
    generation_config, verify_storage_only_config, continuation_coverage,
    predecessor_path, validate_resume_decision, validate_reused_probe,
    validate_never_attempted, resolve_recorded_successor, compact_completion,
    validate_continuation_decision,
)
from data.sglang_backlog_fast import generation_config as original_config


def test_configuration_is_exact_original_plus_declared_storage_code():
    config = generation_config()
    verify_storage_only_config(config, original_config())
    config['temperature'] = 'changed'
    with pytest.raises(ValueError, match='changed model'):
        verify_storage_only_config(config, original_config())


def manifest_fixture():
    source = {'source': 'maud', 'rows': 3, 'probe_rows': 2, 'probe_ids': ['a', 'b'],
        'chunks': [{'index': 0, 'rows': 2, 'sha256': 'input', 'task_ids_sha256': 'ids'}]}
    origin = {'sources': [source], 'generation_config': original_config()}
    manifest = {'sources': [dict(source, rows=1)], 'pending_rows': 272287,
        'pending_counts': EXPECTED_REMAINING, 'generation_config': generation_config()}
    original_decision = {'probe_sources': ['maud']}
    resume = {'decision': 'resume_original_source_probes_on_volume_v2', 'reviewed_by': 'root',
        'approved_for_generation': True, 'approved_for_release': False, 'scope': 'source_probes_only',
        'original_coordinator_stopped': True, 'original_coordinator_call_id': 'fc-old',
        'reconciliation_review': 'Original app stopped and no reservations exist.',
        'backlog_manifest_sha256': 'new', 'origin_probe_manifest_sha256': ORIGIN_MANIFEST_SHA256,
        'origin_probe_decision_sha256': ORIGIN_DECISION_SHA256, 'generation_config': manifest['generation_config'],
        'reused_probe_reports': {}, 'never_attempted_probe_sources': ['maud']}
    return source, origin, manifest, original_decision, resume


def test_resume_requires_exact_original_probe_partition_and_stopped_review():
    _, origin, manifest, original_decision, resume = manifest_fixture()
    assert validate_resume_decision(resume, manifest, origin, original_decision, manifest_sha='new')[0] == {'maud'}
    for change in [{'original_coordinator_stopped': False}, {'reused_probe_reports': {'maud': 'a'*64}},
                   {'never_attempted_probe_sources': []}, {'origin_probe_decision_sha256': 'wrong'}]:
        with pytest.raises(ValueError):
            validate_resume_decision({**resume, **change}, manifest, origin, original_decision, manifest_sha='new')


def test_never_attempted_rejects_any_reservation_even_without_model_request(tmp_path):
    validate_never_attempted(tmp_path)
    (tmp_path / 'attempts.jsonl').write_text('{}\n')
    with pytest.raises(ValueError, match='attempts/results'):
        validate_never_attempted(tmp_path)


def test_never_attempted_rejects_orphan_raw_or_committed_result(tmp_path):
    p = tmp_path / 'attempts' / 't' / 'attempt-01'
    p.mkdir(parents=True)
    (p / 'raw.json').write_text('[]')
    with pytest.raises(ValueError):
        validate_never_attempted(tmp_path)


def test_reused_report_requires_all_exact_probe_ids_and_binding():
    source, *_ = manifest_fixture()
    report = {'status': 'complete', 'source': 'maud', 'chunk': 0, 'rows': 2, 'completed': 2,
        'input_sha256': 'input', 'effective_task_ids_sha256': 'ids',
        'backlog_manifest_sha256': ORIGIN_MANIFEST_SHA256, 'decision_sha256': ORIGIN_DECISION_SHA256,
        'result_hashes': {'a': 'hash-a', 'b': 'hash-b'}}
    validate_reused_probe(report, source=source)
    for change in [{'completed': 1}, {'result_hashes': {'a': 'hash-a'}}, {'decision_sha256': 'wrong'}]:
        with pytest.raises(ValueError):
            validate_reused_probe({**report, **change}, source=source)


def test_predecessor_routes_original_maud_and_resumed_other_probe_separately():
    assert str(predecessor_path('maud', 1, origin_root='/old', output_root='/new', reused_probes={'maud': 'h'})).startswith('/old/')
    assert str(predecessor_path('finqa', 1, origin_root='/old', output_root='/new', reused_probes={'maud': 'h'})).startswith('/new/')
    assert str(predecessor_path('maud', 2, origin_root='/old', output_root='/new', reused_probes={'maud': 'h'})).startswith('/new/')
    with pytest.raises(ValueError):
        predecessor_path('maud', 0, origin_root='/old', output_root='/new')


def test_handoff_never_duplicates_uncertain_dispatch():
    parent = {'function_call_id': 'fc-parent', 'input_id': 'in-parent'}
    intent = {'parent_identity': parent, 'known_successor_call_ids': ['old']}
    assert resolve_recorded_successor(intent, parent_identity=parent, child_call_ids=['old', 'new']) == 'new'
    assert resolve_recorded_successor({**intent, 'successor_call_id': 'known'}, parent_identity=parent, child_call_ids=[]) == 'known'
    for candidates in [[], ['old'], ['new1', 'new2']]:
        with pytest.raises(ValueError):
            resolve_recorded_successor(intent, parent_identity=parent, child_call_ids=candidates)
    with pytest.raises(ValueError):
        resolve_recorded_successor(intent, parent_identity={}, child_call_ids=['new'])


def test_terminal_accounting_keeps_unresolved_and_unapproved_visible():
    manifest = {'sources': [{'source': 'maud', 'rows': 8, 'probe_rows': 2},
                           {'source': 'finqa', 'rows': 7, 'probe_rows': 2},
                           {'source': 'watsonx_docs_qa', 'rows': 0, 'probe_rows': 0}],
                'generation_config': {}}
    report = compact_completion(manifest, stage='continuation', allowed={'maud'},
        chunks={'maud': [{'completed': 3, 'unresolved': 1, 'counts': {'completed': 3, 'accepted': 1}}]},
        holds={'maud': {'reason': 'interrupted'}}, binding={}, coordinator='state', handoffs=[])
    assert report['status'] == 'approved_plan_drained_with_holds'
    assert report['sources']['maud']['unattempted'] == 4
    assert report['sources']['finqa']['unattempted'] == 7
    assert report['totals'] == {'expected_rows': 15, 'completed': 3, 'unresolved': 1, 'unattempted': 11}
    assert report['approved_for_release'] is False


def test_coverage_checks_probe_disjointness_and_preserves_zero_source(tmp_path):
    rows = [{'task_id': 'a', 'chunk': 0, 'kind': 'retry_rejected'},
            {'task_id': 'b', 'chunk': 1, 'kind': 'previously_unattempted'}]
    raw = ''.join(json.dumps(r)+'\n' for r in rows).encode()
    path = tmp_path / 'coverage.jsonl'; path.write_bytes(raw)
    source = {'source': 'maud', 'rows': 2, 'probe_rows': 1, 'probe_ids': ['a'],
        'coverage': {'path': str(path), 'sha256': sha(raw), 'rows': 2},
        'chunks': [{'index': i, 'rows': 1, 'task_ids_sha256': sha(json.dumps([k]).encode())} for i,k in enumerate(['a','b'])]}
    empty = tmp_path / 'empty.jsonl'; empty.write_bytes(b'')
    zero = {'source': 'watsonx_docs_qa', 'rows': 0, 'probe_rows': 0, 'probe_ids': [], 'chunks': [],
            'coverage': {'path': str(empty), 'rows': 0, 'sha256': sha(b'')}}
    origin = {'sources': [source, zero], 'pending_rows': 2, 'probe_rows': 1, 'continuation_rows': 1}
    sources, counts, digest = continuation_coverage(origin)
    assert [s['rows'] for s in sources] == [1, 0]
    assert counts == {'previously_unattempted': 1}
    assert digest == sha(json.dumps(['b']).encode())
    source['probe_ids'] = ['b']
    with pytest.raises(ValueError):
        continuation_coverage(origin)


def test_continuation_accepts_real_mixed_storage_probe_binding_only():
    source, origin, manifest, original_decision, _ = manifest_fixture()
    decision = {'decision': 'continue_reviewed_sources_with_27b_on_volume_v2', 'reviewed_by': 'root',
        'approved_for_generation': True, 'approved_for_release': False, 'scope': 'reviewed_sources',
        'backlog_manifest_sha256': 'new', 'origin_probe_manifest_sha256': ORIGIN_MANIFEST_SHA256,
        'origin_probe_decision_sha256': ORIGIN_DECISION_SHA256, 'generation_config': manifest['generation_config'],
        'resume_probe_decision_sha256': 'resume', 'source_reviews': {'maud': {
            'approved_for_generation': True, 'evidence_review': 'Manually reviewed.', 'probe_report_sha256': 'report'}}}
    report = {'status': 'complete', 'source': 'maud', 'chunk': 0, 'rows': 2, 'completed': 2,
        'input_sha256': 'input', 'backlog_manifest_sha256': 'new', 'decision_sha256': 'resume'}
    def check(r):
        return validate_continuation_decision(decision, manifest, origin, original_decision,
            {'maud': {'sha256': 'report', 'report': r}}, manifest_sha='new',
            origin_manifest_sha=ORIGIN_MANIFEST_SHA256, origin_decision_sha=ORIGIN_DECISION_SHA256)
    assert check(report) == {'maud'}
    assert report['backlog_manifest_sha256'] == 'new'  # no mutation of stored evidence
    with pytest.raises(ValueError):
        check({**report, 'decision_sha256': 'unreviewed'})
