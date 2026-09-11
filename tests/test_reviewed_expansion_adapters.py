import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from data import reviewed_expansion_adapters as adapters
from data import export_reviewed_sglang27 as core
from test_export_reviewed_sglang27 import fixture as sg_fixture, write_json, id_digest


def jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = b''.join((json.dumps(row) + '\n').encode() for row in rows)
    path.write_bytes(raw)
    return {'path': str(path), 'sha256': core.sha(raw), 'rows': len(rows)}


def counts(results, accepted, excluded=0):
    return {'results': results, 'accepted': accepted, 'excluded_accepted': excluded, 'exported': accepted - excluded}


def legacy(tmp_path, keys=('legacy-1',), missing_source_id=False):
    sg_fixture(tmp_path, rows=[(key, True) for key in keys])
    traces = []
    for key in keys:
        trace = json.loads((tmp_path / 'results' / (key + '.json')).read_text())['trace']
        trace.update(model='Qwen/235B', model_revision='actual-legacy-revision', segment_count=1)
        if missing_source_id: trace.pop('source_row_id')
        traces.append(trace)
    input_spec = jsonl(tmp_path / 'accepted.jsonl', traces)
    generation = write_json(tmp_path / 'generation.json', {'model': 'Qwen/235B',
                            'model_revision': 'actual-legacy-revision', 'temperature': 0})
    report = write_json(tmp_path / 'source-report.json', {'source': 'fixture', 'status': 'complete',
                         'reasons': {'accepted:fixture': len(traces), 'wrong_answer:fixture': 8}})
    audit = write_json(tmp_path / 'audit.json', {'source': 'fixture', 'status': 'passed',
        'accepted_file_sha256': input_spec['sha256'], 'generation_manifest_file_sha256': generation['sha256'],
        'rows': len(traces), 'expected_accepted': len(traces), 'failed_rows': 0, 'errors': []})
    review = write_json(tmp_path / 'source-review.json', {'source': 'fixture',
                        'accepted_file_sha256': input_spec['sha256'], 'status': 'sample_review_passed'})
    return {'stream_id': 'legacy', 'kind': 'legacy_accepted_jsonl', 'source': 'fixture',
            'source_allowlist': ['fixture'], 'input': input_spec, 'generation_manifest': generation,
            'source_report': report, 'format_audit': audit, 'review_artifacts': [review],
            'raw_capture_status': 'unavailable_legacy', 'counts': {'fixture': counts(len(keys), len(keys))},
            'task_ids_sha256': id_digest(keys)}


def pilot(tmp_path, rows=(('pilot-1', True), ('pilot-rejected', False))):
    sg_fixture(tmp_path, rows=list(rows))
    tasks = {t['task_id']: t for t in map(json.loads, (tmp_path / 'tasks.jsonl').read_text().splitlines())}
    cases, results = [], []
    for key, accepted in rows:
        result = json.loads((tmp_path / 'results' / (key + '.json')).read_text())
        result['trace'].update(model=None, model_revision=None, segment_count=1)
        result.pop('source_row_id'); result.pop('attempt'); result.pop('task_sha256')
        records = json.loads((tmp_path / 'raw' / key / 'attempt-01/raw.json').read_text())
        raw = write_json(tmp_path / 'pilot-raw' / (key + '.json'), records)
        result['raw_responses_sha256'] = raw['sha256']
        results.append(result)
        cases.append({'source': 'fixture', 'task': core.normalization.prepare_teacher_task(tasks[key]),
                      'baseline': {'verification': {'accepted': False}}})
    inputs = write_json(tmp_path / 'fixture.inputs.json', cases)
    manifest = write_json(tmp_path / 'pilot-manifest.json', {'rows': len(rows), 'model': core.MODEL,
        'model_revision': core.REVISION, 'temperature': 0.7,
        'sources': [{'source': 'fixture', 'count': len(rows), 'inputs_sha256': inputs['sha256']}]})
    result_spec = jsonl(tmp_path / 'pilot-results.jsonl', results)
    n = sum(a for _, a in rows)
    report = write_json(tmp_path / 'pilot-report.json', {'manifest_sha256': manifest['sha256'],
        'results_sha256': result_spec['sha256'], 'completed': len(rows), 'counts': {'automatic_accepted': n},
        'sources': {'fixture': {'completed': len(rows), 'automatic_accepted': n}}})
    audit = write_json(tmp_path / 'pilot-audit.json', {'status': 'failed', 'pilot_manifest_sha256': manifest['sha256'],
        'pilot_report_sha256': report['sha256'], 'pilot_results_sha256': result_spec['sha256'],
        'trace_errors': [], 'automatic_accepted': n, 'accepted_trace_counts': {'accepted': n},
        'response_errors': [{'task_id': key, 'error': 'rejected_format_case'} for key, accepted in rows if not accepted]})
    review = write_json(tmp_path / 'pilot-review.json', {'fixture': True, 'status': 'reviewed'})
    return {'stream_id': 'pilot', 'kind': 'frozen_pilot_jsonl', 'source_allowlist': ['fixture'],
            'manifest': manifest, 'report': report, 'results': result_spec, 'format_audit': audit,
            'review_artifacts': [review], 'inputs': [{'source': 'fixture', **inputs}],
            'raw_root': str(tmp_path / 'pilot-raw'), 'counts': {'fixture': counts(len(rows), n)},
            'task_ids_sha256': id_digest([key for key, accepted in rows if accepted])}


def union_spec(tmp_path, streams, exported_ids, excluded=('known-bad',)):
    review = write_json(tmp_path / 'root-review.json', {'fixture': True, 'reviewed_by': 'root'})
    exclusions = write_json(tmp_path / 'exclusions.json', {'reviewed_by': 'root', 'count': len(excluded),
        'entries': [{'task_id': key, 'source': 'fixture', 'reason': 'Fixture concern'} for key in excluded]})
    total = dict.fromkeys(counts(0, 0), 0)
    for stream in streams:
        for key, value in stream['counts']['fixture'].items(): total[key] += value
    return {'schema': adapters.SCHEMA, 'status': 'reviewed_complete', 'reviewed_sources': ['fixture'],
            'excluded_task_ids': list(excluded), 'exclusions': exclusions,
            'approval': {'status': 'approved', 'scope': 'internal_private_packing',
                         'public_release_approved': False, 'artifacts': [review]},
            'base_snapshot': {'root': str(tmp_path / 'base'),
                              'manifest_sha256': {'16384': 'a' * 64, '32768': 'b' * 64}},
            'streams': streams, 'source_counts': {'fixture': total}, 'task_ids_sha256': id_digest(exported_ids)}


def run_union(tmp_path, selection):
    spec = write_json(tmp_path / 'union-selection.json', selection)
    return adapters.export_union(spec['path'], spec['sha256'], tmp_path / 'union-transport',
        reviewed_sources=['fixture'], excluded_task_ids=selection['excluded_task_ids'], shard_rows=1)


def test_legacy_preserves_teacher_and_explicitly_unavailable_raw(tmp_path):
    spec = legacy(tmp_path)
    candidate, = list(adapters.iter_legacy_accepted(spec, selection_sha='a' * 64))
    row = candidate['row']; provenance = json.loads(row['provenance'])
    assert row['task_id'] == 'legacy-1' and row['source_row_id'] == 'native-legacy-1'
    assert row['raw_responses_sha256'] is None and row['task_sha256'] is None
    assert provenance['raw_capture']['status'] == 'unavailable_legacy' and 'sha256' not in provenance['raw_capture']
    assert provenance['teacher']['model'] == 'Qwen/235B' and provenance['teacher']['temperature'] == 0
    assert provenance['original_trace_metadata']['model_revision'] == 'actual-legacy-revision'
    assert provenance['original_jsonl_row']['scope'] == 'jsonl_line_bytes'


def test_missing_legacy_source_row_requires_explicit_frozen_join(tmp_path):
    spec = legacy(tmp_path, missing_source_id=True)
    with pytest.raises(ValueError, match='requires frozen task join'):
        list(adapters.iter_legacy_accepted(spec, selection_sha='a' * 64))
    source = tmp_path / 'tasks.jsonl'
    spec['source_tasks'] = {'path': str(source), 'sha256': core.sha(source.read_bytes()), 'rows': 1}
    candidate, = list(adapters.iter_legacy_accepted(spec, selection_sha='a' * 64))
    provenance = json.loads(candidate['row']['provenance'])
    assert provenance['joined_fields']['source_row_id']['value'] == 'native-legacy-1'
    assert 'source_row_id' not in provenance['original_trace_metadata']


@pytest.mark.parametrize('failure', ['input_bytes', 'audit_binding', 'teacher', 'fabricated_raw', 'source_review'])
def test_legacy_rejects_changed_artifact_or_provenance(tmp_path, failure):
    spec = legacy(tmp_path)
    if failure == 'input_bytes':
        with Path(spec['input']['path']).open('ab') as handle: handle.write(b' ')
    elif failure == 'audit_binding':
        value = adapters.json_document(spec['format_audit']); value['accepted_file_sha256'] = '0' * 64
        spec['format_audit'] = write_json(Path(spec['format_audit']['path']), value)
    elif failure == 'teacher':
        value = adapters.json_document(spec['generation_manifest']); value['model'] = core.MODEL
        spec['generation_manifest'] = write_json(Path(spec['generation_manifest']['path']), value)
    elif failure == 'fabricated_raw': spec['raw_responses_sha256'] = 'a' * 64
    elif failure == 'source_review': spec['review_artifacts'] = []
    with pytest.raises(ValueError): list(adapters.iter_legacy_accepted(spec, selection_sha='a' * 64))


def test_pilot_joins_teacher_without_overwriting_original_null_fields(tmp_path):
    spec = pilot(tmp_path)
    candidates = list(adapters.iter_frozen_pilot(spec, selection_sha='a' * 64))
    assert len(candidates) == 2 and candidates[1]['row'] is None
    row = candidates[0]['row']; provenance = json.loads(row['provenance'])
    assert row['source_row_id'] == 'native-pilot-1'
    assert provenance['original_trace_metadata']['model'] is None
    assert provenance['teacher']['model'] == core.MODEL
    assert provenance['joined_fields']['teacher']['artifact'] == spec['manifest']
    assert provenance['raw_capture']['status'] == 'captured_and_hash_verified'
    assert row['raw_responses_sha256'] == provenance['raw_capture']['sha256']
    assert provenance['original_jsonl_row']['sha256'] == core.sha(Path(spec['results']['path']).read_bytes().splitlines(keepends=True)[0])


@pytest.mark.parametrize('failure', ['raw', 'input', 'audit', 'source_count', 'accepted_raw_error'])
def test_pilot_validates_frozen_inputs_raw_and_counts(tmp_path, failure):
    spec = pilot(tmp_path)
    if failure in ('raw', 'input'):
        path = Path(spec['raw_root']) / 'pilot-1.json' if failure == 'raw' else Path(spec['inputs'][0]['path'])
        with path.open('ab') as handle: handle.write(b' ')
    elif failure == 'audit':
        value = adapters.json_document(spec['format_audit']); value['trace_errors'] = [{'task_id': 'pilot-1'}]
        spec['format_audit'] = write_json(Path(spec['format_audit']['path']), value)
    elif failure == 'source_count':
        value = adapters.json_document(spec['report']); value['sources']['fixture']['completed'] += 1
        spec['report'] = write_json(Path(spec['report']['path']), value)
        audit = adapters.json_document(spec['format_audit']); audit['pilot_report_sha256'] = spec['report']['sha256']
        spec['format_audit'] = write_json(Path(spec['format_audit']['path']), audit)
    elif failure == 'accepted_raw_error':
        audit = adapters.json_document(spec['format_audit']); audit['response_errors'][0]['task_id'] = 'pilot-1'
        spec['format_audit'] = write_json(Path(spec['format_audit']['path']), audit)
    with pytest.raises(ValueError): list(adapters.iter_frozen_pilot(spec, selection_sha='a' * 64))


def test_three_generation_union_preserves_distinctions_and_append_schema(tmp_path):
    from data.grouped_stage3_expansion_append import load_manifest
    old = legacy(tmp_path / 'old')
    pil = pilot(tmp_path / 'pilot')
    current = sg_fixture(tmp_path / 'current', rows=[('current-1', True)])
    current_spec = write_json(tmp_path / 'current/selection.json', current)
    new = {'stream_id': 'current', 'kind': 'sglang27_task_results', 'source_allowlist': ['fixture'],
           'selection': current_spec, 'counts': {'fixture': counts(1, 1)}, 'task_ids_sha256': id_digest(['current-1'])}
    selection = union_spec(tmp_path, [old, pil, new], ['legacy-1', 'pilot-1', 'current-1'])
    manifest = run_union(tmp_path, selection)
    assert manifest['rows'] == 3 and manifest['source_counts']['fixture'] == counts(4, 3)
    rows = [r for file in manifest['files'] for r in pq.read_table(Path(manifest['transport_root']) / file['name']).to_pylist()]
    assert {r['task_id'] for r in rows} == {'legacy-1', 'pilot-1', 'current-1'}
    assert all(r['approved_for_release'] is False for r in rows)
    path = Path(manifest['transport_root']) / 'manifest.json'
    assert load_manifest(path, core.sha(path.read_bytes())) == manifest
    assert (Path(manifest['transport_root']) / 'components/current/manifest.json').exists()


def test_union_rejects_duplicate_accepted_ids_across_generations(tmp_path):
    old = legacy(tmp_path / 'old', keys=('same',))
    pil = pilot(tmp_path / 'pilot', rows=(('same', True),))
    selection = union_spec(tmp_path, [old, pil], ['same'])
    with pytest.raises(ValueError, match='duplicate exported ID'):
        run_union(tmp_path, selection)
    assert not (tmp_path / 'union-transport/manifest.json').exists()


def test_union_excludes_documented_bad_ids_and_requires_all_root_exclusions(tmp_path):
    old = legacy(tmp_path / 'old', keys=('good', 'known-bad'))
    old['counts']['fixture'] = counts(2, 2, 1); old['task_ids_sha256'] = id_digest(['good'])
    selection = union_spec(tmp_path, [old], ['good'])
    result = run_union(tmp_path, selection)
    assert result['rows'] == 1 and result['source_counts']['fixture']['excluded_accepted'] == 1
    other = tmp_path / 'missing'; other.mkdir()
    selection['excluded_task_ids'] = []
    with pytest.raises(ValueError, match='exclusions omitted'):
        run_union(other, selection)


def completed_component(tmp_path):
    selection = sg_fixture(tmp_path / 'current', rows=[('current-1', True)])
    spec = write_json(tmp_path / 'current/selection.json', selection)
    core.export_reviewed(spec['path'], spec['sha256'], tmp_path / 'completed-core',
                         reviewed_sources=['fixture'], excluded_task_ids=['known-bad'])
    path = tmp_path / 'completed-core/manifest.json'
    return {'stream_id': 'current', 'kind': 'sglang27_task_results', 'source_allowlist': ['fixture'],
            'selection': spec, 'transport': {'path': str(path), 'sha256': core.sha(path.read_bytes())},
            'counts': {'fixture': counts(1, 1)}, 'task_ids_sha256': id_digest(['current-1'])}


def test_completed_core_transport_reuse_does_not_export_again(tmp_path, monkeypatch):
    stream = completed_component(tmp_path)
    def forbidden(*args, **kwargs): raise AssertionError('Core export must not run again')
    monkeypatch.setattr(core, 'export_reviewed', forbidden)
    result = run_union(tmp_path, union_spec(tmp_path, [stream], ['current-1']))
    assert result['rows'] == 1 and not (tmp_path / 'union-transport/components').exists()
    row = pq.read_table(Path(result['transport_root']) / result['files'][0]['name']).to_pylist()[0]
    assert json.loads(row['provenance'])['upstream_transport_manifest'] == stream['transport']


@pytest.mark.parametrize('problem', ['selection', 'sources', 'shard', 'ids', 'counts'])
def test_completed_transport_reuse_rejects_changed_bindings(tmp_path, problem):
    stream = completed_component(tmp_path)
    path = Path(stream['transport']['path']); manifest = json.loads(path.read_text())
    if problem == 'selection': manifest['selection_sha256'] = 'f' * 64
    elif problem == 'sources': manifest['reviewed_sources'] = ['unreviewed']
    elif problem == 'shard':
        shard = Path(manifest['transport_root']) / manifest['files'][0]['name']
        with shard.open('ab') as handle: handle.write(b'changed')
    elif problem == 'ids': manifest['task_ids_sha256'] = '0' * 64
    elif problem == 'counts': manifest['source_counts']['fixture']['exported'] = 3
    if problem != 'shard': stream['transport'] = write_json(path, manifest)
    with pytest.raises(ValueError):
        run_union(tmp_path, union_spec(tmp_path, [stream], ['current-1']))
    assert not (tmp_path / 'union-transport/manifest.json').exists()
