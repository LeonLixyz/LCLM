import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from data import export_reviewed_sglang27 as export


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(value) + '\n').encode()
    path.write_bytes(raw)
    return {'path': str(path), 'sha256': export.sha(raw)}


def id_digest(ids):
    return export.sha(('\n'.join(sorted(ids)) + '\n').encode())


def fixture(tmp_path, *, rows=None, mutate=None):
    rows = rows or [('task-1', True), ('task-2', False), ('known-bad', True)]
    tasks = []
    results = {}
    for key, accepted in rows:
        task = {'task_id': key, 'source_row_id': 'native-' + key, 'source_dataset': 'source/original',
                'family': 'synthetic', 'question': 'What reading?',
                'segments': [{'record_id': key, 'segment_id': 'seg_1', 'summary': 'A reading', 'text': 'The reading is 42.'}]}
        tasks.append(task)
        prepared = export.normalization.prepare_teacher_task(task)
        messages = [{'role': 'system', 'content': prepared['training_system_prompt']},
                    {'role': 'user', 'content': prepared['training_user_prompt']},
                    {'role': 'assistant', 'content': '', 'tool_calls': [{'id': 'call-1', 'type': 'function',
                        'function': {'name': 'expand', 'arguments': {'segment_id': 'seg_1'}}}]},
                    {'role': 'tool', 'tool_call_id': 'call-1', 'name': 'expand', 'content': 'The reading is 42.'},
                    {'role': 'assistant', 'content': 'FINAL: 42'}]
        verification = {'accepted': accepted, 'reason': 'fixture'}
        trace = {'task_id': key, 'source_row_id': task['source_row_id'], 'source_dataset': task['source_dataset'],
                 'sub_dataset': 'fixture', 'compression_scope': 'input_segments', 'data_type': 'synthetic_expansion_agent',
                 'model': export.MODEL, 'model_revision': export.REVISION, 'verification': verification,
                 'messages': messages, 'tools': [{'type': 'function', 'function': {'name': 'expand', 'parameters': {}}}],
                 'attempt_provenance': {'kind': 'retry_rejected'}, 'gold_answer': '42'}
        records = [{'phase': 'rollout', 'request': {'model': export.MODEL,
                    'extra_body': {'chat_template_kwargs': {'enable_thinking': False}}},
                    'response': {'choices': [{'message': {'content': 'FINAL: 42', 'reasoning_content': None}}]},
                    'raw_generated_text': 'FINAL: 42<|im_end|>'}]
        result = {'task_id': key, 'source': 'fixture', 'source_row_id': task['source_row_id'],
                  'source_dataset': task['source_dataset'], 'task_sha256': export.sha(json.dumps(task, sort_keys=True).encode()),
                  'verification': verification, 'trace': trace, 'requests': 1, 'attempt': 1,
                  'teacher': {'layout': 'fixture'}, 'approved_for_release': False}
        if mutate:
            mutate(key, result, records)
        raw = write_json(tmp_path / 'raw' / key / 'attempt-01' / 'raw.json', records)
        result['raw_responses_sha256'] = raw['sha256']
        results[key] = write_json(tmp_path / 'results' / (key + '.json'), result)['sha256']
    raw_tasks = b''.join((json.dumps(t) + '\n').encode() for t in tasks)
    task_path = tmp_path / 'tasks.jsonl'; task_path.write_bytes(raw_tasks)
    report = write_json(tmp_path / 'report.json', {'source': 'fixture', 'status': 'complete',
        'rows': len(tasks), 'completed': len(tasks), 'input_sha256': export.sha(raw_tasks),
        'result_hashes': results, 'counts': {'automatic_accepted': sum(a for _, a in rows)}})
    excluded = ['known-bad']
    selected = [key for key, accepted in rows if accepted and key not in excluded]
    source = {'source': 'fixture', 'counts': {'results': len(rows), 'accepted': sum(a for _, a in rows),
              'excluded_accepted': sum(a and key in excluded for key, a in rows), 'exported': len(selected)},
              'task_ids_sha256': id_digest(selected), 'chunks': [{'tasks': {'path': str(task_path),
                  'sha256': export.sha(raw_tasks), 'rows': len(tasks)}, 'report': report,
                  'results_root': str(tmp_path / 'results'), 'attempts_root': str(tmp_path / 'raw')}]}
    selection = {'schema': export.SELECTION_SCHEMA, 'status': 'reviewed_complete',
        'model': export.MODEL, 'model_revision': export.REVISION,
        'normalization': {'module_sha256': export.sha(Path(export.normalization.__file__).read_bytes())},
        'reviewed_sources': ['fixture'], 'excluded_task_ids': excluded,
        'approval': {'status': 'approved', 'scope': 'internal_private_packing', 'public_release_approved': False,
                     'artifacts': [write_json(tmp_path / 'review.json', {'fixture_reviewed': True})]},
        'sources': [source], 'base_snapshot': {'root': str(tmp_path / 'base'),
                'manifest_sha256': {'16384': 'a' * 64, '32768': 'b' * 64}}}
    return selection


def run(tmp_path, selection, **kwargs):
    spec = write_json(tmp_path / 'selection.json', selection)
    return export.export_reviewed(spec['path'], spec['sha256'], tmp_path / 'transport',
        reviewed_sources=['fixture'], excluded_task_ids=['known-bad'], **kwargs)


def test_accepted_only_excludes_known_bad_preserves_provenance_and_append_schema(tmp_path):
    from data.grouped_stage3_expansion_append import load_manifest, normalize_transport_row
    selection = fixture(tmp_path)
    result = run(tmp_path, selection, shard_rows=1)
    assert result['rows'] == 1 and result['task_ids_sha256'] == id_digest(['task-1'])
    assert result['source_counts']['fixture'] == selection['sources'][0]['counts']
    path = tmp_path / 'transport' / result['files'][0]['name']
    rows = pq.read_table(path).to_pylist()
    assert rows[0]['task_id'] == 'task-1' and rows[0]['source_row_id'] == 'native-task-1'
    original = json.loads((tmp_path / 'results/task-1.json').read_text())
    assert json.loads(rows[0]['messages']) == original['trace']['messages']
    assert json.loads(rows[0]['tools']) == original['trace']['tools']
    provenance = json.loads(rows[0]['provenance'])
    assert provenance['result'] == {k: v for k, v in original.items() if k != 'trace'}
    assert provenance['trace']['attempt_provenance'] == original['trace']['attempt_provenance']
    assert rows[0]['approved_for_release'] is False and result['approval']['public_release_approved'] is False
    normalize_transport_row(rows[0])
    manifest_path = tmp_path / 'transport/manifest.json'
    assert load_manifest(manifest_path, export.sha(manifest_path.read_bytes())) == result
    assert export.sha(path.read_bytes()) == result['files'][0]['sha256']
    with pytest.raises(FileExistsError):
        run(tmp_path, selection)


def test_assistant_thought_stripped_without_changing_calls_source_or_final(tmp_path):
    def dirty(key, result, raw):
        result['trace']['messages'][2].update(content='<think>Inspect evidence.</think> A preamble', reasoning_content='private')
        result['trace']['messages'][-1]['content'] = '<think>Calculate.</think>\nFINAL: 42'
    selection = fixture(tmp_path, mutate=dirty)
    result = run(tmp_path, selection)
    row = pq.read_table(tmp_path / 'transport' / result['files'][0]['name']).to_pylist()[0]
    cleaned = json.loads(row['messages'])
    assert cleaned[2]['content'] == '' and 'reasoning_content' not in cleaned[2]
    assert cleaned[-1]['content'] == 'FINAL: 42'
    assert 'seg_1\n<|memory_start|>' in cleaned[1]['content']
    assert cleaned[3]['content'] == 'The reading is 42.'


@pytest.mark.parametrize('target', ['selection', 'report', 'tasks', 'result', 'raw', 'review'])
def test_changed_frozen_bytes_fail_closed(tmp_path, target):
    selection = fixture(tmp_path)
    spec = write_json(tmp_path / 'selection.json', selection)
    paths = {'selection': tmp_path / 'selection.json', 'report': tmp_path / 'report.json',
             'tasks': tmp_path / 'tasks.jsonl', 'result': tmp_path / 'results/task-1.json',
             'raw': tmp_path / 'raw/task-1/attempt-01/raw.json', 'review': tmp_path / 'review.json'}
    with paths[target].open('ab') as handle:
        handle.write(b' ')
    with pytest.raises(ValueError):
        export.export_reviewed(spec['path'], spec['sha256'], tmp_path / 'transport',
            reviewed_sources=['fixture'], excluded_task_ids=['known-bad'])
    assert not (tmp_path / 'transport/manifest.json').exists()


@pytest.mark.parametrize('problem', ['counts', 'ids', 'approval', 'allowlist', 'exclusions', 'teacher', 'bound'])
def test_review_scope_counts_and_ids_must_match(tmp_path, problem):
    selection = fixture(tmp_path)
    if problem == 'counts': selection['sources'][0]['counts']['results'] += 1
    elif problem == 'ids': selection['sources'][0]['task_ids_sha256'] = 'f' * 64
    elif problem == 'approval': selection['approval']['scope'] = 'generation_only'
    elif problem == 'allowlist': selection['reviewed_sources'].append('unreviewed')
    elif problem == 'exclusions': selection['excluded_task_ids'] = []
    elif problem == 'teacher': selection['model'] = '235B'
    elif problem == 'bound': selection['sources'][0]['chunks'][0]['tasks']['rows'] = 513
    with pytest.raises(ValueError):
        run(tmp_path, selection)
    assert not (tmp_path / 'transport/manifest.json').exists()


def test_duplicate_id_across_chunks_is_rejected_even_if_rejected(tmp_path):
    selection = fixture(tmp_path, rows=[('rejected', False)])
    selection['sources'][0]['chunks'] *= 2
    with pytest.raises(ValueError, match='Duplicate global task ID'):
        run(tmp_path, selection)
    assert not (tmp_path / 'transport/manifest.json').exists()


@pytest.mark.parametrize('problem', ['body', 'identity', 'raw_thinking', 'final_tags', 'acceptance', 'prompt'])
def test_invalid_trace_or_raw_contract_is_rejected(tmp_path, problem):
    def bad(key, result, raw):
        if key != 'task-1': return
        if problem == 'body': result['trace']['messages'][3]['content'] = 'The reading is 43.'
        elif problem == 'identity': result['trace']['source_row_id'] = 'fabricated'
        elif problem == 'raw_thinking': raw[0]['response']['choices'][0]['message']['reasoning_content'] = 'private'
        elif problem == 'final_tags': result['trace']['messages'][-1]['content'] += '\n</tool_call>'
        elif problem == 'acceptance': result['verification']['accepted'] = 1
        elif problem == 'prompt': result['trace']['messages'][1]['content'] += 'Fabricated source hint.'
    selection = fixture(tmp_path, mutate=bad)
    with pytest.raises(ValueError):
        run(tmp_path, selection)
    assert not (tmp_path / 'transport/manifest.json').exists()


def test_shards_are_bounded_and_global_digest_sorted(tmp_path):
    selection = fixture(tmp_path, rows=[('z', True), ('a', True), ('b', True)])
    result = run(tmp_path, selection, shard_rows=2)
    assert [f['rows'] for f in result['files']] == [2, 1]
    assert result['task_ids_sha256'] == id_digest(['z', 'a', 'b'])


def test_reused_continuation_chunk_excludes_probe_ids_before_export(tmp_path):
    selection = fixture(tmp_path, rows=[('probe-id', True), ('continuation-id', True)])
    source = selection['sources'][0]
    spec = source['chunks'][0]['tasks']
    spec.update(rows=1, exclude_task_ids=['probe-id'],
                task_ids_sha256=export.sha(json.dumps(['continuation-id']).encode()))
    report_path = tmp_path / 'report.json'
    report = json.loads(report_path.read_text())
    report['rows'] = report['completed'] = report['counts']['automatic_accepted'] = 1
    del report['result_hashes']['probe-id']
    source['chunks'][0]['report'] = write_json(report_path, report)
    source['counts'] = {'results': 1, 'accepted': 1, 'excluded_accepted': 0, 'exported': 1}
    source['task_ids_sha256'] = id_digest(['continuation-id'])
    result = run(tmp_path, selection)
    assert result['rows'] == 1 and result['task_ids_sha256'] == id_digest(['continuation-id'])
