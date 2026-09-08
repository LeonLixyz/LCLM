"""Select one fresh attempt per not-yet-accepted task; preserve prior outputs."""
import json
from pathlib import Path
from collections import Counter
from data.expansion_checkpoint_audit import audit_source
from data.prepare_teacher_transition import file_sha

SOURCES = ('maud', 'finqa', 'pubmedqa_labeled', 'clapnq', 'contract_nli', 'tatqa',
           'convfinqa', 'multihiertt', 'multidoc2dial', 'faithdial', 'watsonx_docs_qa',
           'acord', 'billsum', 'lex_glue', 'synthetic')


def validate_prepared(report, destination):
    destination = Path(destination)
    for item in report['parent_audit']['files']:
        if file_sha(item['path']) != item['sha256']:
            raise ValueError('Parent checkpoint changed: ' + item['path'])
    parent = Path(report['previous_root'])
    actual = {str(parent / (report['source'] + '.' + k + '.jsonl'))
              for k in ('accepted', 'rejected')
              if (parent / (report['source'] + '.' + k + '.jsonl')).exists()}
    if actual != {x['path'] for x in report['parent_audit']['files']}:
        raise ValueError('Parent checkpoint file set changed')
    for name, digest in report['prepared_files'].items():
        if file_sha(destination / name) != digest:
            raise ValueError('Prepared retry input changed: ' + name)


def prepare_source(source, task_path, previous_root, destination):
    task_path, previous_root, destination = map(Path, (task_path, previous_root, destination))
    destination.mkdir(parents=True, exist_ok=True)
    marker = destination / (source + '.build.json')
    if marker.exists():
        report = json.loads(marker.read_text())
        if report['task_path'] != str(task_path) or report['previous_root'] != str(previous_root):
            raise ValueError('Changed retry input roots')
        validate_prepared(report, destination)
        return report
    audit = audit_source(task_path, previous_root, source)
    previous = {}
    for item in audit['files']:
        with Path(item['path']).open() as stream:
            for line in stream:
                row = json.loads(line)
                previous[row['task_id']] = row['verification']
    target = destination / (source + '.tasks.jsonl')
    if target.exists():
        raise ValueError('Partial retry preparation exists; inspect before recovery')
    counts = Counter(); choices = {}
    with task_path.open('rb') as stream, target.open('xb') as out:
        for line in stream:
            if not line.endswith(b'\n'):
                raise ValueError('Incomplete task input')
            task = json.loads(line)
            if source == 'maud':
                from data.expansion_task_normalization import maud_field
                choices.setdefault(maud_field(task), set()).add(task['gold_answer'])
            verdict = previous.get(task['task_id'])
            if verdict and verdict['accepted']:
                counts['previously_accepted'] += 1
                continue
            kind = 'retry_rejected' if verdict else 'previously_unattempted'
            counts[kind] += 1
            task['_attempt_provenance'] = {'kind': kind, 'previous_root': str(previous_root),
                'previous_reason': verdict['reason'] if verdict else None,
                'corrected_multidoc2dial': source == 'multidoc2dial'}
            out.write((json.dumps(task, ensure_ascii=False) + '\n').encode())
    prepared = {target.name: file_sha(target)}
    if choices:
        ontology = destination / 'maud-choices.json'
        ontology.write_text(json.dumps({k: sorted(v) for k, v in choices.items()}, sort_keys=True))
        prepared[ontology.name] = file_sha(ontology)
    selected = counts['retry_rejected'] + counts['previously_unattempted']
    if selected + counts['previously_accepted'] != audit['tasks']:
        raise ValueError('Retry accounting mismatch')
    report = {'status': 'complete', 'source': source, 'tasks': selected,
        'scope': 'not_previously_accepted', 'counts': dict(counts),
        'task_path': str(task_path), 'task_sha256': file_sha(task_path),
        'previous_root': str(previous_root), 'parent_audit': audit,
        'prepared_files': prepared, 'approved_for_release': False}
    validate_prepared(report, destination)
    marker.write_text(json.dumps(report, indent=2))
    return report
