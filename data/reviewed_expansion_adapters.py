"""Inert typed readers and a private-packing union writer for reviewed expansion.

Union selection schema: reviewed-expansion-union-v1; status=reviewed_complete;
reviewed_sources; excluded_task_ids; exclusions={path,sha256} (root's documented
exclusion file); approval={status:approved,scope:internal_private_packing,
public_release_approved:false,artifacts:[{path,sha256}]}; base_snapshot; streams.

Each stream has stream_id, kind, source_allowlist, counts per source using the
results/accepted/excluded_accepted/exported keys, and exported task_ids_sha256.
Types:

legacy_accepted_jsonl: source, input={path,sha256,rows}, generation_manifest,
source_report, format_audit, review_artifacts, raw_capture_status=unavailable_legacy;
optional source_tasks={path,sha256,rows} for missing source-row identity.

frozen_pilot_jsonl: manifest, report, results={path,sha256,rows}, format_audit,
review_artifacts, inputs=[{source,path,sha256}], raw_root. The full pilot is
validated; only source_allowlist results are yielded for selection.

sglang27_task_results: selection={path,sha256} pointing to the separately
root-approved export_reviewed_sglang27 selection. Its frozen transport is built
in a new durable components/<stream_id> directory and reused losslessly. An
optional transport={path,sha256} reuses an already completed core transport
manifest after verifying its selection, source scope, artifacts, files and IDs.

No function discovers sources, approves selection, submits a job or publishes.
Legacy raw-capture absence and pilot joined fields remain explicit provenance.
All originals remain unchanged. A completion manifest appears only after every
stream digest, count and exported-ID check succeeds.
"""
from __future__ import annotations

import json
from pathlib import Path
import re
import sqlite3
import tempfile

from data import export_reviewed_sglang27 as core

SCHEMA = 'reviewed-expansion-union-v1'


def json_document(spec):
    return json.loads(core.frozen_bytes(spec))


def frozen_jsonl(spec):
    """Yield exact line identity; verify the entire stream on exhaustion."""
    import hashlib
    count, digest = 0, hashlib.sha256()
    with core.absolute(spec['path']).open('rb') as handle:
        for line in handle:
            core.require(line.endswith(b'\n') and line.strip(), 'Incomplete/empty JSONL line')
            digest.update(line)
            count += 1
            yield json.loads(line), {'file': spec, 'line_number': count,
                                    'sha256': core.sha(line), 'scope': 'jsonl_line_bytes'}
    core.require(digest.hexdigest() == core.digest(spec['sha256']), 'Frozen JSONL hash changed')
    core.require(type(spec['rows']) is int and count == spec['rows'], 'Frozen JSONL count differs')


def validate_review_artifacts(spec, *, source=None, accepted_sha=None):
    core.require(spec['review_artifacts'], 'Missing source review evidence')
    values = [json_document(a) for a in spec['review_artifacts']]
    if accepted_sha is not None:
        core.require(any(v.get('source') == source and v.get('accepted_file_sha256') == accepted_sha
                         for v in values), 'Source review is not bound to the accepted file')


def saved_segments(trace):
    messages = trace['messages']
    users = [m for m in messages if m.get('role') == 'user']
    core.require(len(users) == 1, 'Expected one saved source prompt')
    matches = re.findall(r'(?m)^(seg_[1-9][0-9]*)\n<\|memory_start\|>(.*?)<\|memory_end\|>',
                         users[0]['content'], re.S)
    core.require(len(matches) == trace['segment_count'] > 0 and
                 len({key for key, _ in matches}) == len(matches), 'Invalid saved memory segments')
    return {'segments': [{'segment_id': key, 'text': text} for key, text in matches]}


def transport(trace, *, task, source, source_row_id, line, provenance, selection_sha):
    core.require(trace['verification']['accepted'] is True and trace['compression_scope'] == 'input_segments',
                 'Only accepted input-segment traces are eligible')
    messages, cleaning = core.clean_messages(trace['messages'])
    core.validate_native(messages, trace['tools'], task)
    provenance = {**provenance, 'original_trace_metadata': {k: v for k, v in trace.items()
                  if k not in ('messages', 'tools')}, 'export_harvesting': cleaning}
    return {'task_id': core.valid_id(trace['task_id']), 'source_row_id': core.valid_id(source_row_id),
            'source': source, 'source_dataset': core.valid_id(trace['source_dataset']),
            'sub_dataset': core.valid_id(trace['sub_dataset']), 'compression_scope': 'input_segments',
            'data_type': 'agent_trajectory', 'messages': json.dumps(messages, ensure_ascii=False),
            'tools': json.dumps(trace['tools'], ensure_ascii=False),
            'provenance': json.dumps(provenance, ensure_ascii=False),
            'task_sha256': None, 'result_sha256': line['sha256'],
            'raw_responses_sha256': provenance.get('raw_capture', {}).get('sha256'),
            'selection_sha256': selection_sha, 'approved_for_release': False}


def iter_legacy_accepted(spec, *, selection_sha):
    """Read one explicitly enumerated legacy accepted file, never old selectors."""
    source = core.valid_id(spec['source'])
    core.require(spec['source_allowlist'] == [source], 'Legacy stream must name its one source')
    core.require(spec['raw_capture_status'] == 'unavailable_legacy' and
                 not spec.get('raw_responses_sha256'), 'Do not fabricate legacy raw-capture coverage')
    generation, report, audit = (json_document(spec[k]) for k in
                                  ('generation_manifest', 'source_report', 'format_audit'))
    accepted = sum(v for k, v in report['reasons'].items() if k.startswith('accepted:') or k == 'accepted')
    core.require(report['source'] == source and report['status'] == 'complete' and
                 accepted == spec['input']['rows'], 'Legacy source report/count mismatch')
    core.require(audit['status'] == 'passed' and audit['source'] == source and
                 audit['accepted_file_sha256'] == spec['input']['sha256'] and
                 audit['generation_manifest_file_sha256'] == spec['generation_manifest']['sha256'] and
                 audit['rows'] == audit['expected_accepted'] == accepted and
                 audit['failed_rows'] == 0 and not audit['errors'], 'Legacy format audit mismatch')
    validate_review_artifacts(spec, source=source, accepted_sha=spec['input']['sha256'])
    core.valid_id(generation['model']); core.valid_id(generation['model_revision'])
    with tempfile.TemporaryDirectory(prefix='legacy-source-join-') as directory:
        db = sqlite3.connect(str(Path(directory) / 'join.sqlite'))
        try:
            db.execute('PRAGMA cache_size=-1024')
            db.execute('CREATE TABLE task_rows (id TEXT PRIMARY KEY, row_id TEXT, line TEXT)')
            db.execute('CREATE TABLE attempted (id TEXT PRIMARY KEY)')
            if spec.get('source_tasks'):
                for task, line in frozen_jsonl(spec['source_tasks']):
                    row_id = task.get('source_row_id')
                    if not row_id and task.get('family') == 'synthetic' and 'index' in task:
                        row_id = str(task['index'])
                    core.valid_id(row_id)
                    try:
                        db.execute('INSERT INTO task_rows VALUES (?, ?, ?)',
                                   (core.valid_id(task['task_id']), row_id, json.dumps(line)))
                    except sqlite3.IntegrityError as exc:
                        raise ValueError('Duplicate legacy source-task ID') from exc
                db.commit()
            for trace, line in frozen_jsonl(spec['input']):
                key = core.valid_id(trace['task_id'])
                try:
                    db.execute('INSERT INTO attempted VALUES (?)', (key,))
                except sqlite3.IntegrityError as exc:
                    raise ValueError('Duplicate legacy accepted ID') from exc
                core.require(trace['verification']['accepted'] is True and
                             trace['model'] == generation['model'] and
                             trace['model_revision'] == generation['model_revision'], 'Legacy acceptance/teacher mismatch')
                row_id, joined = trace.get('source_row_id'), {}
                join = db.execute('SELECT row_id, line FROM task_rows WHERE id=?', (key,)).fetchone()
                if not row_id:
                    core.require(join is not None, 'Missing legacy source_row_id requires frozen task join')
                    row_id = join[0]
                    joined['source_row_id'] = {'value': row_id, 'provenance': json.loads(join[1])}
                elif join:
                    core.require(row_id == join[0], 'Legacy source-row join conflict')
                provenance = {'input_kind': 'legacy_accepted_jsonl', 'original_jsonl_row': line,
                    'generation_manifest': spec['generation_manifest'], 'teacher': generation,
                    'source_report': spec['source_report'], 'format_audit': spec['format_audit'],
                    'review_artifacts': spec['review_artifacts'], 'joined_fields': joined,
                    'raw_capture': {'status': 'unavailable_legacy',
                        'scope': 'Canonical saved messages audited; pre-parser raw output unavailable.'}}
                row = transport(trace, task=saved_segments(trace), source=source, source_row_id=row_id,
                                line=line, provenance=provenance, selection_sha=selection_sha)
                yield {'task_id': key, 'source': source, 'accepted': True, 'row': row}
        finally:
            db.close()


def iter_frozen_pilot(spec, *, selection_sha):
    """Validate the frozen pilot and join missing metadata without rewriting it."""
    manifest, report, audit = (json_document(spec[k]) for k in ('manifest', 'report', 'format_audit'))
    core.require(manifest['model'] == core.MODEL and manifest['model_revision'] == core.REVISION,
                 'Unexpected frozen pilot teacher')
    core.require(report['manifest_sha256'] == spec['manifest']['sha256'] and
                 report['results_sha256'] == spec['results']['sha256'] and
                 report['completed'] == manifest['rows'] == spec['results']['rows'], 'Pilot manifest/result binding mismatch')
    # The original audit status is failed because of 14 rejected final-response
    # formatting cases. Require its accepted-trace checks, and retain all errors.
    core.require(audit['pilot_manifest_sha256'] == spec['manifest']['sha256'] and
                 audit['pilot_report_sha256'] == spec['report']['sha256'] and
                 audit['pilot_results_sha256'] == spec['results']['sha256'] and
                 audit['trace_errors'] == [] and audit['automatic_accepted'] ==
                 report['counts']['automatic_accepted'] == audit['accepted_trace_counts']['accepted'],
                 'Pilot accepted-trace audit mismatch')
    validate_review_artifacts(spec)
    response_error_ids = {e.get('task_id') for e in audit.get('response_errors', [])}
    core.require(None not in response_error_ids, 'Unscoped pilot response audit error')
    expected = {s['source']: s for s in manifest['sources']}
    inputs = {s['source']: s for s in spec['inputs']}
    allowed = set(spec['source_allowlist'])
    core.require(len(inputs) == len(spec['inputs']) and set(inputs) == set(expected) and
                 allowed <= set(expected), 'Pilot frozen input source set mismatch')
    with tempfile.TemporaryDirectory(prefix='pilot-source-join-') as directory:
        db = sqlite3.connect(str(Path(directory) / 'join.sqlite'))
        try:
            db.execute('PRAGMA cache_size=-1024')
            db.execute('CREATE TABLE tasks (id TEXT PRIMARY KEY, source TEXT, task TEXT, used INTEGER DEFAULT 0)')
            for source, entry in inputs.items():
                core.require(entry['sha256'] == expected[source]['inputs_sha256'] and
                             expected[source]['count'] <= core.MAX_CHUNK_ROWS, 'Pilot input hash/count mismatch')
                cases = json_document(entry)
                core.require(len(cases) == expected[source]['count'], 'Pilot input row count mismatch')
                for case in cases:
                    core.require(case['source'] == source, 'Pilot source case mismatch')
                    task = case['task']
                    try:
                        db.execute('INSERT INTO tasks (id, source, task) VALUES (?, ?, ?)',
                                   (core.valid_id(task['task_id']), source, json.dumps(task)))
                    except sqlite3.IntegrityError as exc:
                        raise ValueError('Duplicate pilot task input ID') from exc
            db.commit()
            core.require(all(db.execute('SELECT 1 FROM tasks WHERE id=?', (key,)).fetchone()
                             for key in response_error_ids), 'Pilot response audit references unknown task')
            counts = {source: {'completed': 0, 'automatic_accepted': 0} for source in expected}
            for result, line in frozen_jsonl(spec['results']):
                key, source = core.valid_id(result['task_id']), result['source']
                joined = db.execute('SELECT source, task, used FROM tasks WHERE id=?', (key,)).fetchone()
                core.require(joined is not None and joined[0] == source and joined[2] == 0,
                             'Duplicate or unjoined pilot result')
                db.execute('UPDATE tasks SET used=1 WHERE id=?', (key,))
                accepted = result['verification']['accepted']
                core.require(type(accepted) is bool, 'Pilot accepted must be boolean')
                counts[source]['completed'] += 1
                counts[source]['automatic_accepted'] += accepted
                core.require(Path(key).name == key and key not in ('.', '..'), 'Unsafe pilot task filename')
                raw_spec = {'path': str(core.absolute(spec['raw_root']) / (key + '.json')),
                            'sha256': result['raw_responses_sha256']}
                core.check_raw(json_document(raw_spec), result)
                if source not in allowed:
                    continue
                if not accepted:
                    yield {'task_id': key, 'source': source, 'accepted': False, 'row': None}
                    continue
                core.require(key not in response_error_ids, 'Accepted pilot row has unresolved raw-response audit errors')
                trace, task = result['trace'], json.loads(joined[1])
                core.require(trace['task_id'] == key and trace['verification'] == result['verification'] and
                             trace.get('model') in (None, manifest['model']) and
                             trace.get('model_revision') in (None, manifest['model_revision']), 'Pilot trace metadata mismatch')
                row_id = core.valid_id(task['source_row_id'])
                core.require(trace.get('source_row_id') in (None, row_id) and
                             trace['source_dataset'] == task['source_dataset'], 'Pilot source-row join conflict')
                core.require(trace['messages'][:2] == [
                    {'role': 'system', 'content': task['training_system_prompt']},
                    {'role': 'user', 'content': task['training_user_prompt']}], 'Pilot saved prompt differs from frozen input')
                provenance = {'input_kind': 'frozen_pilot_jsonl', 'original_jsonl_row': line,
                    'original_result_metadata': {k: v for k, v in result.items() if k != 'trace'},
                    'teacher': manifest, 'generation_manifest': spec['manifest'],
                    'format_audit': spec['format_audit'], 'review_artifacts': spec['review_artifacts'],
                    'raw_capture': {'status': 'captured_and_hash_verified', **raw_spec},
                    'joined_fields': {'source_row_id': {'value': row_id, 'artifact': inputs[source]},
                        'teacher': {'model': manifest['model'], 'model_revision': manifest['model_revision'],
                                    'artifact': spec['manifest']}}}
                row = transport(trace, task=task, source=source, source_row_id=row_id,
                                line=line, provenance=provenance, selection_sha=selection_sha)
                yield {'task_id': key, 'source': source, 'accepted': True, 'row': row}
            core.require(db.execute('SELECT COUNT(*) FROM tasks WHERE used=0').fetchone()[0] == 0,
                         'Missing frozen pilot results')
            for source, value in counts.items():
                core.require(all(value[k] == report['sources'][source][k] for k in value), 'Pilot exact source counts differ')
        finally:
            db.close()


def iter_sglang27_component(spec, *, selection_sha, component_root):
    """Use the existing strict result adapter under its own explicit approval."""
    import pyarrow.parquet as pq
    upstream = json_document(spec['selection'])
    core.require(set(upstream['reviewed_sources']) == set(spec['source_allowlist']), 'Component source scope differs')
    if spec.get('transport'):
        transport_spec = spec['transport']
        manifest = json_document(transport_spec)
    else:
        manifest = core.export_reviewed(spec['selection']['path'], spec['selection']['sha256'], component_root,
                                       reviewed_sources=spec['source_allowlist'],
                                       excluded_task_ids=upstream['excluded_task_ids'])
        path = component_root / 'manifest.json'
        transport_spec = {'path': str(path), 'sha256': core.sha(path.read_bytes())}
    approval = manifest['approval']
    core.require(manifest['schema'] == core.TRANSPORT_SCHEMA and manifest['status'] == 'reviewed_complete' and
                 manifest['selection_sha256'] == approval['selection_sha256'] == spec['selection']['sha256'] and
                 set(manifest['reviewed_sources']) == set(upstream['reviewed_sources']) and
                 set(manifest['excluded_task_ids']) == set(upstream['excluded_task_ids']) and
                 manifest['base_snapshot'] == upstream['base_snapshot'] and
                 approval['status'] == 'approved' and approval['scope'] == 'internal_private_packing' and
                 approval['public_release_approved'] is False and manifest['approved_for_release'] is False,
                 'Completed component selection/approval binding differs')
    for artifact in approval['artifacts']: core.frozen_bytes(artifact)
    files = manifest['files']
    core.require(files and len({f['name'] for f in files}) == len(files) and
                 all(Path(f['name']).name == f['name'] and f['name'].endswith('.parquet') and
                     type(f['rows']) is int and f['rows'] > 0 for f in files) and
                 sum(f['rows'] for f in files) == manifest['rows'] > 0,
                 'Invalid completed component shard manifest')
    transport_root = core.absolute(manifest['transport_root'])
    with tempfile.TemporaryDirectory(prefix='component-ids-') as temporary:
        db = sqlite3.connect(str(Path(temporary) / 'ids.sqlite'))
        try:
            db.execute('PRAGMA cache_size=-1024')
            db.execute('CREATE TABLE ids (task_id TEXT PRIMARY KEY, source TEXT, exported INTEGER)')
            counts = dict.fromkeys(manifest['reviewed_sources'], 0)
            for file in files:
                path = transport_root / file['name']
                core.require(core.sha(path.read_bytes()) == file['sha256'], 'Component transport changed')
                parquet = pq.ParquetFile(path)
                core.require(parquet.metadata.num_rows == file['rows'], 'Component shard row count differs')
                for batch in parquet.iter_batches(batch_size=32):
                    for row in batch.to_pylist():
                        key, source = core.valid_id(row['task_id']), row['source']
                        core.require(source in counts and row['selection_sha256'] == spec['selection']['sha256'] and
                                     row['approved_for_release'] is False, 'Component row identity/selection differs')
                        try:
                            db.execute('INSERT INTO ids VALUES (?, ?, 1)', (key, source))
                        except sqlite3.IntegrityError as exc:
                            raise ValueError('Duplicate completed component ID') from exc
                        counts[source] += 1
                        provenance = json.loads(row['provenance'])
                        provenance['upstream_selection'] = spec['selection']
                        provenance['upstream_transport_manifest'] = transport_spec
                        provenance['upstream_transport_shard'] = {'path': str(path), 'sha256': file['sha256']}
                        row['provenance'] = json.dumps(provenance, ensure_ascii=False)
                        row['selection_sha256'] = selection_sha
                        yield {'task_id': key, 'source': source, 'accepted': True, 'row': row}
                db.commit()
            core.require(core.ids_digest(db) == manifest['task_ids_sha256'] and
                         all(counts[s] == manifest['source_counts'][s]['exported'] for s in counts),
                         'Completed component source counts/ID digest differ')
        finally:
            db.close()


def export_union(selection_path, expected_sha256, destination, *, reviewed_sources,
                 excluded_task_ids, shard_rows=2000):
    """Export only the explicit reviewed union; its IDs are globally unique."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    selection_spec = {'path': str(core.absolute(selection_path)), 'sha256': expected_sha256}
    selection = json_document(selection_spec)
    core.require(selection['schema'] == SCHEMA and selection['status'] == 'reviewed_complete', 'Unreviewed union')
    approval = selection['approval']
    core.require(approval['status'] == 'approved' and approval['scope'] == 'internal_private_packing' and
                 approval['public_release_approved'] is False and approval['artifacts'], 'Explicit private approval required')
    allowed, excluded = set(reviewed_sources), set(excluded_task_ids)
    core.require(len(allowed) == len(reviewed_sources) and len(excluded) == len(excluded_task_ids) and
                 allowed == set(selection['reviewed_sources']) and excluded == set(selection['excluded_task_ids']),
                 'Union caller scope/exclusions differ')
    exclusions = json_document(selection['exclusions'])
    documented = [e['task_id'] for e in exclusions['entries']]
    core.require(exclusions['reviewed_by'] == 'root' and len(documented) == len(set(documented)) == exclusions['count']
                 and set(documented) <= excluded, 'Documented root exclusions omitted')
    for artifact in approval['artifacts']:
        core.frozen_bytes(artifact)
    names = [core.valid_id(s['stream_id']) for s in selection['streams']]
    core.require(names and len(names) == len(set(names)) and
                 all(Path(n).name == n and n not in ('.', '..') for n in names), 'Invalid/duplicate stream IDs')
    core.require(type(shard_rows) is int and 0 < shard_rows <= 2000, 'Invalid shard bound')
    base = selection['base_snapshot']
    core.absolute(base['root'])
    core.require(set(base['manifest_sha256']) == {'16384', '32768'}, 'Both base snapshots required')
    for value in base['manifest_sha256'].values(): core.digest(value)
    output = core.absolute(destination); output.mkdir(parents=True, exist_ok=False)
    schema = pa.schema([(k, pa.string()) for k in core.STRING_COLUMNS] + [('approved_for_release', pa.bool_())])
    buffer, files, stream_reports = [], [], []
    totals = {s: dict.fromkeys(('results', 'accepted', 'excluded_accepted', 'exported'), 0) for s in allowed}

    def flush():
        if not buffer: return
        path = output / f'accepted-{len(files):06d}.parquet'
        with path.open('xb') as handle:
            pq.write_table(pa.Table.from_pylist(buffer, schema=schema), handle, compression='zstd')
        core.require(pq.ParquetFile(path).metadata.num_rows == len(buffer), 'Union shard row count differs')
        files.append({'name': path.name, 'rows': len(buffer), 'sha256': core.sha(path.read_bytes())})
        buffer.clear()

    with tempfile.TemporaryDirectory(prefix='expansion-union-ids-') as temporary:
        db = sqlite3.connect(str(Path(temporary) / 'ids.sqlite'))
        try:
            db.execute('PRAGMA cache_size=-1024')
            db.execute('CREATE TABLE ids (task_id TEXT PRIMARY KEY, source TEXT, exported INTEGER)')
            db.execute('CREATE TABLE stream_ids (stream TEXT, task_id TEXT, exported INTEGER, PRIMARY KEY(stream,task_id))')
            for stream in selection['streams']:
                stream_allowed = set(stream['source_allowlist'])
                core.require(stream_allowed and stream_allowed <= allowed and set(stream['counts']) == stream_allowed,
                             'Unreviewed stream source or count set')
                for value in stream['counts'].values(): core.checked_counts(value)
                name, kind = stream['stream_id'], stream['kind']
                if kind == 'legacy_accepted_jsonl': iterator = iter_legacy_accepted(stream, selection_sha=expected_sha256)
                elif kind == 'frozen_pilot_jsonl': iterator = iter_frozen_pilot(stream, selection_sha=expected_sha256)
                elif kind == 'sglang27_task_results':
                    iterator = iter_sglang27_component(stream, selection_sha=expected_sha256,
                                                       component_root=output / 'components' / name)
                else: raise ValueError('Unknown typed expansion stream')
                counts = {s: dict.fromkeys(totals[s], 0) for s in stream_allowed}
                for candidate in iterator:
                    key, source, accepted = candidate['task_id'], candidate['source'], candidate['accepted']
                    core.require(source in stream_allowed and type(accepted) is bool, 'Invalid adapter candidate')
                    selected = accepted and key not in excluded
                    try:
                        db.execute('INSERT INTO stream_ids VALUES (?, ?, ?)', (name, key, int(selected)))
                        if selected: db.execute('INSERT INTO ids VALUES (?, ?, 1)', (key, source))
                    except sqlite3.IntegrityError as exc:
                        raise ValueError('Duplicate attempted ID within stream or duplicate exported ID across streams: ' + key) from exc
                    counts[source]['results'] += 1
                    counts[source]['accepted'] += accepted
                    counts[source]['excluded_accepted'] += accepted and not selected
                    counts[source]['exported'] += selected
                    if selected:
                        buffer.append(candidate['row'])
                        if len(buffer) >= shard_rows: flush()
                core.require(counts == stream['counts'], 'Exact reviewed stream counts differ: ' + name)
                import hashlib
                h, n = hashlib.sha256(), 0
                for key, in db.execute('SELECT task_id FROM stream_ids WHERE stream=? AND exported=1 ORDER BY task_id', (name,)):
                    h.update((key + '\n').encode()); n += 1
                if not n: h.update(b'\n')
                core.require(h.hexdigest() == stream['task_ids_sha256'], 'Reviewed stream ID digest differs')
                for source, values in counts.items():
                    for key, value in values.items(): totals[source][key] += value
                count_scope = {'legacy_accepted_jsonl': 'Frozen accepted JSONL input rows',
                               'frozen_pilot_jsonl': 'All frozen pilot result rows for allowed sources',
                               'sglang27_task_results': 'Rows exported by the separately approved 27B component; its manifest retains full generation counts'}[kind]
                stream_reports.append({'stream_id': name, 'kind': kind, 'counts': counts, 'count_scope': count_scope,
                                       'task_ids_sha256': h.hexdigest()})
                db.commit()
            flush()
            total = sum(v['exported'] for v in totals.values())
            core.require(total > 0 and total == sum(f['rows'] for f in files), 'Empty/inconsistent union export')
            core.require(totals == selection['source_counts'], 'Combined reviewed source counts differ')
            id_sha = core.ids_digest(db)
            core.require(id_sha == selection['task_ids_sha256'], 'Combined reviewed ID digest differs')
        finally:
            db.close()
    core.frozen_bytes(selection_spec); core.frozen_bytes(selection['exclusions'])
    for artifact in approval['artifacts']: core.frozen_bytes(artifact)
    manifest = {'schema': core.TRANSPORT_SCHEMA, 'status': 'reviewed_complete',
                'selection_sha256': expected_sha256, 'selection_path': selection_spec['path'],
                'approval': {**approval, 'selection_sha256': expected_sha256,
                             'artifacts': [*approval['artifacts'], selection_spec, selection['exclusions']]},
                'transport_root': str(output), 'rows': total, 'task_ids_sha256': id_sha,
                'files': files, 'source_counts': totals, 'streams': stream_reports, 'base_snapshot': base,
                'source_counts_scope': 'Typed transport-selection candidates, not combined generation attempt totals; see each stream count_scope and upstream artifacts.',
                'reviewed_sources': sorted(allowed), 'excluded_task_ids': sorted(excluded),
                'approved_for_release': False, 'public_release_approved': False}
    with (output / 'manifest.json').open('x') as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2); handle.write('\n')
    return manifest
