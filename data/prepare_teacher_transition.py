"""Create an immutable delta task set; never mutate earlier teacher records."""
import hashlib
import json
from pathlib import Path
from data.expansion_checkpoint_audit import audit_source

SOURCES = ('billsum', 'lex_glue', 'synthetic')


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def prepare_remaining(task_root, previous_root, destination):
    task_root, previous_root, destination = map(Path, (task_root, previous_root, destination))
    manifest_path = destination/'transition-manifest.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest['previous_root'] != str(previous_root) or manifest['task_root'] != str(task_root):
            raise ValueError('Changed transition input roots')
        for source in manifest['sources']:
            actual_files = {str(previous_root/(source['source']+'.'+kind+'.jsonl'))
                            for kind in ('accepted','rejected')
                            if (previous_root/(source['source']+'.'+kind+'.jsonl')).exists()}
            if actual_files != {item['path'] for item in source['files']}:
                raise ValueError('Previous teacher file set changed')
            if file_sha(destination/(source['source']+'.tasks.jsonl')) != source['remaining_sha256']:
                raise ValueError('Changed delta tasks')
            for artifact in source['files']:
                if file_sha(artifact['path']) != artifact['sha256']:
                    raise ValueError('Previous teacher still writing or changed')
        return manifest
    destination.mkdir(parents=True, exist_ok=True)
    reports = []
    for source in SOURCES:
        task_path = task_root/(source+'.tasks.jsonl')
        report = audit_source(task_path, previous_root, source)
        completed = set()
        for item in report['files']:
            with Path(item['path']).open() as stream:
                completed.update(json.loads(line)['task_id'] for line in stream)
        out_path = destination/(source+'.tasks.jsonl')
        if out_path.exists():
            raise ValueError('Partial transition exists; inspect before recovery')
        remaining = 0
        with task_path.open('rb') as stream, out_path.open('xb') as out:
            for line in stream:
                if not line.endswith(b'\n'):
                    raise ValueError('Incomplete prepared task line')
                if json.loads(line)['task_id'] not in completed:
                    out.write(line)
                    remaining += 1
        if remaining != report['remaining']:
            raise ValueError('Delta row count mismatch')
        for item in report['files']:
            if file_sha(item['path']) != item['sha256']:
                raise ValueError('Previous teacher changed during transition')
        report.update(remaining_sha256=file_sha(out_path), prepared_sha256=file_sha(task_path))
        reports.append(report)
        (destination/(source+'.build.json')).write_text(json.dumps({
            'status':'complete', 'scope':'unattempted_delta_only',
            'source':source,'tasks':remaining,'previous_attempts':report['persisted']}))
    manifest = {'status':'prepared', 'scope':'unattempted_delta_only',
                'previous_root':str(previous_root), 'task_root':str(task_root),
                'sources':reports,'approved_for_release':False,
                'completed_records_modified':False,
                'requires_composition_with_previous_and_corrected_md':True}
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest
