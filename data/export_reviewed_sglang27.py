"""Bounded CPU export of an explicitly reviewed, frozen 27B selection.

This module neither selects sources nor launches work. ``export_reviewed``
requires the caller's selection SHA, reviewed source allowlist and known-bad
exclusions. Input schema ``reviewed-sglang27-export-v1`` contains:

* status=reviewed_complete, model/model_revision, reviewed_sources,
  excluded_task_ids, base_snapshot (the grouped append adapter's schema);
* normalization={module_sha256, maud_choices:{path,sha256}}; maud_choices
  is required only when MAUD is in the reviewed source allowlist;
* approval={status:approved, scope:internal_private_packing,
  public_release_approved:false, artifacts:[{path,sha256}, ...]};
* sources=[{source, counts:{results,accepted,excluded_accepted,exported},
  task_ids_sha256, chunks:[{tasks:{path,sha256,rows},
  report:{path,sha256}, results_root, attempts_root}, ...]}, ...].

Source task_ids_sha256 covers exported IDs, using sorted IDs plus a final
newline. Chunk reports are the generator's complete reports with result_hashes
and counts. Every path is explicit and absolute. All results, including rejected
ones, have their bytes, raw captures, task identity and counts checked. Only
accepted, nonexcluded rows reach transport. Each task chunk is at most 512 rows;
global uniqueness and sorted ID digests use a temporary disk SQLite index.

The destination must not exist. A failed export stays incomplete and is never
silently resumed or overwritten. The transport manifest is written last. Its
approval applies to internal/private packing only, never public HF release.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import tempfile

from data.clean_agent_trajectories import strip_cot
from data.harvest_expansion_trace import harvest_training_messages
from data import expansion_task_normalization as normalization

SELECTION_SCHEMA = 'reviewed-sglang27-export-v1'
TRANSPORT_SCHEMA = 'reviewed-expansion-transport-v1'
MODEL = 'Qwen/Qwen3.8-27B'
REVISION = '1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0'
MAX_CHUNK_ROWS = 512
THINK = re.compile(r'</?(?:think|analysis)>|<\|(?:think|analysis)', re.I)
REASON_FIELDS = ('reasoning', 'reasoning_content', 'analysis', 'thinking')
STRING_COLUMNS = ('task_id', 'source_row_id', 'source', 'source_dataset',
                  'sub_dataset', 'compression_scope', 'data_type', 'messages',
                  'tools', 'provenance', 'task_sha256', 'result_sha256',
                  'raw_responses_sha256', 'selection_sha256')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def valid_id(value):
    require(isinstance(value, str) and value and not any(c in value for c in '\r\n\0'),
            'Missing or malformed identity')
    return value


def absolute(value):
    path = Path(value)
    require(path.is_absolute(), 'All frozen paths must be absolute')
    return path


def digest(value):
    require(isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value), 'Invalid SHA256')
    return value


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def frozen_bytes(spec):
    raw = absolute(spec['path']).read_bytes()
    require(sha(raw) == digest(spec['sha256']), 'Frozen file hash changed: ' + spec['path'])
    return raw


def frozen_tasks(spec):
    """At most 512 physical rows, including excluded probe IDs in reused chunks."""
    require(type(spec['rows']) is int and 0 < spec['rows'] <= MAX_CHUNK_ROWS, 'Task chunk exceeds memory bound')
    omitted = spec.get('exclude_task_ids', [])
    require(len(omitted) == len(set(omitted)), 'Duplicate chunk exclusion')
    omitted = set(omitted)
    tasks, found, h = {}, set(), hashlib.sha256()
    with absolute(spec['path']).open('rb') as handle:
        for index, line in enumerate(handle):
            require(index < MAX_CHUNK_ROWS and line.endswith(b'\n'), 'Oversized or incomplete frozen task JSONL')
            h.update(line)
            task = json.loads(line)
            key = valid_id(task['task_id'])
            require(key not in found, 'Duplicate task in source chunk')
            found.add(key)
            if key not in omitted:
                tasks[key] = task
    require(h.hexdigest() == digest(spec['sha256']), 'Frozen file hash changed: ' + spec['path'])
    require(omitted <= found and len(tasks) == spec['rows'], 'Task chunk count/exclusion mismatch')
    if 'task_ids_sha256' in spec:
        require(sha(json.dumps(sorted(tasks)).encode()) == spec['task_ids_sha256'], 'Effective chunk ID digest mismatch')
    return tasks


def checked_counts(value):
    keys = ('results', 'accepted', 'excluded_accepted', 'exported')
    require(set(value) == set(keys), 'Expected exact source count fields')
    require(all(type(value[k]) is int and value[k] >= 0 for k in keys), 'Invalid source counts')
    require(value['accepted'] <= value['results'] and
            value['accepted'] == value['excluded_accepted'] + value['exported'], 'Inconsistent source counts')
    return value


def clean_messages(messages):
    """Remove only assistant annotations; keep every call and observation intact."""
    require(isinstance(messages, list) and messages, 'Missing trace messages')
    cleaned = copy.deepcopy(messages)
    for message in cleaned:
        require(isinstance(message, dict), 'Invalid message')
        if message.get('role') == 'assistant':
            message['content'] = strip_cot(message.get('content'))
            for field in REASON_FIELDS:
                message.pop(field, None)
        else:
            require(not any(field in message for field in REASON_FIELDS), 'Reasoning field on nonassistant message')
    cleaned, harvesting = harvest_training_messages(cleaned)
    require(harvest_training_messages(cleaned)[0] == cleaned, 'Non-idempotent harvesting')
    for before, after in zip(messages, cleaned):
        if after.get('role') == 'assistant':
            require(not THINK.search(after.get('content') or ''), 'Thinking marker remains')
            require(not any(field in after for field in REASON_FIELDS), 'Thinking field remains')
            require(before.get('tool_calls') == after.get('tool_calls'), 'Tool calls changed during cleaning')
        else:
            require(before == after, 'Source prompt or observation changed during cleaning')
    return cleaned, harvesting


def validate_native(messages, tools, task):
    require(isinstance(tools, list) and len(tools) == 1 and
            tools[0].get('type') == 'function' and tools[0].get('function', {}).get('name') == 'expand',
            'Expected preserved native expand schema')
    originals = {s['segment_id']: s['text'] for s in task['segments']}
    require(len(originals) == len(task['segments']), 'Duplicate source segments')
    pending, seen, observations = {}, set(), 0
    for message in messages:
        if message.get('role') == 'assistant':
            require(not pending, 'Missing tool observation')
            for call in message.get('tool_calls') or []:
                key = valid_id(call['id'])
                fn = call['function']
                args = fn['arguments']
                require(call.get('type') == 'function' and fn['name'] == 'expand' and
                        isinstance(args, dict) and set(args) == {'segment_id'} and
                        args['segment_id'] in originals and key not in seen, 'Invalid native expand call')
                pending[key] = args['segment_id']
                seen.add(key)
        elif message.get('role') == 'tool':
            key = message.get('tool_call_id')
            require(key in pending and message.get('name') == 'expand', 'Orphan/wrong tool observation')
            require(message.get('content') == originals[pending.pop(key)], 'Tool observation differs from source')
            observations += 1
    require(observations > 0 and not pending, 'Incomplete native trajectory')


def check_raw(records, result):
    require(isinstance(records, list) and len(records) == result['requests'] > 0, 'Raw request count mismatch')
    for record in records:
        request = record['request']
        require(request['model'] == MODEL and
                request['extra_body']['chat_template_kwargs']['enable_thinking'] is False,
                'Unexpected model or thinking request')
        if result['verification']['accepted']:
            require(not record.get('error') and record.get('response'), 'Accepted task has failed raw request')
        if record.get('response'):
            message = record['response']['choices'][0]['message']
            # The source audit is authoritative for semantic correctness. Export
            # additionally fails closed on any unexpected thinking-mode output.
            require(not any(message.get(k) for k in REASON_FIELDS), 'Raw reasoning field is nonempty')
            require(isinstance(record.get('raw_generated_text'), str) and
                    not THINK.search(record['raw_generated_text']), 'Raw thinking marker or missing capture')


def ids_digest(connection, source=None):
    query = 'SELECT task_id FROM ids WHERE exported = 1'
    args = ()
    if source is not None:
        query += ' AND source = ?'
        args = (source,)
    h = hashlib.sha256()
    count = 0
    for task_id, in connection.execute(query + ' ORDER BY task_id', args):
        h.update((task_id + '\n').encode())
        count += 1
    if not count:
        h.update(b'\n')
    return h.hexdigest()


def export_reviewed(selection_path, expected_sha256, destination, *, reviewed_sources,
                    excluded_task_ids, shard_rows=2000):
    """Export a supplied approved selection; never infer approval or source scope."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    selection_path = absolute(selection_path)
    selection = json.loads(frozen_bytes({'path': str(selection_path), 'sha256': expected_sha256}))
    require(selection['schema'] == SELECTION_SCHEMA and selection['status'] == 'reviewed_complete', 'Unreviewed selection')
    require(selection['model'] == MODEL and selection['model_revision'] == REVISION, 'Unexpected teacher revision')
    require(sha(Path(normalization.__file__).read_bytes()) == digest(selection['normalization']['module_sha256']),
            'Task normalization code differs from reviewed snapshot')
    approval = selection['approval']
    require(approval['status'] == 'approved' and approval['scope'] == 'internal_private_packing' and
            approval['public_release_approved'] is False and approval['artifacts'], 'Explicit private packing approval required')
    allowed = [valid_id(s) for s in reviewed_sources]
    excluded = [valid_id(s) for s in excluded_task_ids]
    require(len(allowed) == len(set(allowed)) and len(excluded) == len(set(excluded)), 'Duplicate allowlist/exclusion ID')
    require(set(allowed) == set(selection['reviewed_sources']) and
            set(excluded) == set(selection['excluded_task_ids']), 'Caller review/exclusions differ from frozen selection')
    sources = selection['sources']
    require(len(sources) == len(allowed) and {s['source'] for s in sources} == set(allowed), 'Unreviewed/duplicate source')
    require(type(shard_rows) is int and 0 < shard_rows <= 2000, 'Invalid shard bound')
    base = selection['base_snapshot']
    absolute(base['root'])
    require(set(base['manifest_sha256']) == {'16384', '32768'}, 'Both frozen base versions required')
    for value in base['manifest_sha256'].values():
        digest(value)
    for artifact in approval['artifacts']:
        frozen_bytes(artifact)
    choices = json.loads(frozen_bytes(selection['normalization']['maud_choices'])) if 'maud' in allowed else {}
    for source in sources:
        checked_counts(source['counts'])
        digest(source['task_ids_sha256'])
        require(source['chunks'], 'Missing source chunks')
    output = absolute(destination)
    output.mkdir(parents=True, exist_ok=False)
    schema = pa.schema([(key, pa.string()) for key in STRING_COLUMNS] + [('approved_for_release', pa.bool_())])
    buffer, files, source_counts = [], [], {}
    excluded_set = set(excluded)

    def flush():
        if not buffer:
            return
        path = output / f'accepted-{len(files):06d}.parquet'
        with path.open('xb') as handle:
            pq.write_table(pa.Table.from_pylist(buffer, schema=schema), handle, compression='zstd')
        require(pq.ParquetFile(path).metadata.num_rows == len(buffer), 'Transport row count mismatch')
        files.append({'name': path.name, 'rows': len(buffer), 'sha256': sha(path.read_bytes())})
        buffer.clear()

    with tempfile.TemporaryDirectory(prefix='reviewed-export-') as temporary:
        connection = sqlite3.connect(str(Path(temporary) / 'ids.sqlite'))
        try:
            connection.execute('PRAGMA cache_size=-1024')
            connection.execute('CREATE TABLE ids (task_id TEXT PRIMARY KEY, source TEXT, exported INTEGER)')
            for source in sources:
                name = source['source']
                counts = dict.fromkeys(('results', 'accepted', 'excluded_accepted', 'exported'), 0)
                for chunk in source['chunks']:
                    task_spec = chunk['tasks']
                    tasks = frozen_tasks(task_spec)
                    report = json.loads(frozen_bytes(chunk['report']))
                    require(report['status'] == 'complete' and report['source'] == name and
                            report['rows'] == report['completed'] == len(tasks) and
                            report['input_sha256'] == task_spec['sha256'] and
                            set(report['result_hashes']) == set(tasks), 'Incomplete/inconsistent chunk report')
                    result_root, raw_root = absolute(chunk['results_root']), absolute(chunk['attempts_root'])
                    chunk_accepted = 0
                    for task_id in sorted(tasks):
                        # IDs are also filenames in the frozen generator layout.
                        require(Path(task_id).name == task_id and task_id not in ('.', '..'), 'Unsafe task filename')
                        result_spec = {'path': str(result_root / (task_id + '.json')), 'sha256': report['result_hashes'][task_id]}
                        result = json.loads(frozen_bytes(result_spec))
                        task = tasks[task_id]
                        require(result['task_id'] == task_id and result['source'] == name and
                                result['task_sha256'] == sha(json.dumps(task, sort_keys=True).encode()), 'Frozen task/result identity mismatch')
                        accepted = result['verification']['accepted']
                        require(type(accepted) is bool, 'Acceptance must be an explicit boolean')
                        attempt = result['attempt']
                        require(type(attempt) is int and 1 <= attempt <= 3, 'Invalid result attempt')
                        raw_spec = {'path': str(raw_root / task_id / f'attempt-{attempt:02d}' / 'raw.json'),
                                    'sha256': result['raw_responses_sha256']}
                        check_raw(json.loads(frozen_bytes(raw_spec)), result)
                        selected = accepted and task_id not in excluded_set
                        try:
                            connection.execute('INSERT INTO ids VALUES (?, ?, ?)', (task_id, name, int(selected)))
                        except sqlite3.IntegrityError as exc:
                            raise ValueError('Duplicate global task ID: ' + task_id) from exc
                        counts['results'] += 1
                        counts['accepted'] += accepted
                        chunk_accepted += accepted
                        counts['excluded_accepted'] += accepted and not selected
                        if not selected:
                            continue
                        trace = result['trace']
                        require(trace['task_id'] == task_id and trace['verification'] == result['verification'] and
                                trace['model'] == MODEL and trace['model_revision'] == REVISION and
                                trace['compression_scope'] == 'input_segments' and
                                trace['source_dataset'] == result['source_dataset'], 'Trace/result mismatch')
                        source_row_id = valid_id(result['source_row_id'])
                        expected_row_id = task.get('source_row_id', str(task.get('index')))
                        require(source_row_id == expected_row_id == trace['source_row_id'], 'Source row identity changed')
                        messages, harvesting = clean_messages(trace['messages'])
                        prepared = normalization.prepare_teacher_task(task, choices)
                        require(messages[:2] == [
                            {'role': 'system', 'content': prepared['training_system_prompt']},
                            {'role': 'user', 'content': prepared['training_user_prompt']}], 'Saved prompt differs from frozen task')
                        validate_native(messages, trace['tools'], prepared)
                        provenance = {'result': {k: v for k, v in result.items() if k != 'trace'},
                                      'trace': {k: v for k, v in trace.items() if k not in ('messages', 'tools')},
                                      'export_harvesting': harvesting, 'frozen_result': result_spec,
                                      'frozen_raw': raw_spec, 'frozen_tasks': task_spec,
                                      'frozen_report': chunk['report']}
                        row = {
                            'task_id': task_id, 'source_row_id': source_row_id, 'source': name,
                            'source_dataset': valid_id(trace['source_dataset']), 'sub_dataset': valid_id(trace['sub_dataset']),
                            'compression_scope': 'input_segments', 'data_type': 'agent_trajectory',
                            'messages': json.dumps(messages, ensure_ascii=False),
                            'tools': json.dumps(trace['tools'], ensure_ascii=False),
                            'provenance': json.dumps(provenance, ensure_ascii=False),
                            'task_sha256': result['task_sha256'], 'result_sha256': result_spec['sha256'],
                            'raw_responses_sha256': raw_spec['sha256'], 'selection_sha256': expected_sha256,
                            'approved_for_release': False}
                        buffer.append(row)
                        counts['exported'] += 1
                        if len(buffer) >= shard_rows:
                            flush()
                    require(chunk_accepted == report['counts']['automatic_accepted'], 'Chunk accepted count mismatch')
                    connection.commit()
                require(counts == source['counts'], 'Exact reviewed source counts differ: ' + name)
                require(ids_digest(connection, name) == source['task_ids_sha256'], 'Reviewed source ID digest differs: ' + name)
                source_counts[name] = counts
            flush()
            total = sum(c['exported'] for c in source_counts.values())
            require(total > 0 and total == sum(f['rows'] for f in files), 'No exportable rows or shard count mismatch')
            selected_digest = ids_digest(connection)
        finally:
            connection.close()
    # Recheck the governing review documents before writing the completion marker.
    frozen_bytes({'path': str(selection_path), 'sha256': expected_sha256})
    for artifact in approval['artifacts']:
        frozen_bytes(artifact)
    manifest = {'schema': TRANSPORT_SCHEMA, 'status': 'reviewed_complete',
                'selection_sha256': expected_sha256, 'selection_path': str(selection_path),
                'approval': {**approval, 'selection_sha256': expected_sha256,
                             'artifacts': [*approval['artifacts'], {'path': str(selection_path), 'sha256': expected_sha256}]},
                'transport_root': str(output), 'rows': total, 'task_ids_sha256': selected_digest,
                'files': files, 'source_counts': source_counts, 'base_snapshot': base,
                'reviewed_sources': allowed, 'excluded_task_ids': excluded,
                'approved_for_release': False, 'public_release_approved': False,
                'transport': 'JSON messages/tools preserve native objects; provenance preserves original result and trace metadata.'}
    with (output / 'manifest.json').open('x') as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write('\n')
    return manifest
