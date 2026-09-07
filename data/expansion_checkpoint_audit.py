"""Read-only validation of rollout JSONL before resuming an interrupted job."""
import hashlib
import json
from collections import Counter


def audit_source(task_path, output_root, source):
    paths = [output_root / f'{source}.{kind}.jsonl' for kind in ('accepted', 'rejected')]
    seen = set(); reasons = Counter(); files = []; samples = []
    for path, accepted in zip(paths, (True, False)):
        if not path.exists():
            continue
        digest = hashlib.sha256(); rows = 0
        with path.open('rb') as stream:
            for number, line in enumerate(stream, 1):
                digest.update(line)
                if not line.endswith(b'\n'):
                    raise ValueError(f'Incomplete JSONL line: {path}:{number}')
                row = json.loads(line)
                identifier = row['task_id']; verdict = row['verification']
                if not isinstance(identifier, str) or identifier in seen:
                    raise ValueError(f'Duplicate/invalid task ID: {path}:{number}')
                if (verdict.get('accepted') is not accepted
                        or verdict['reason'].startswith('accepted') != accepted):
                    raise ValueError(f'Verdict/file mismatch: {path}:{number}')
                seen.add(identifier); reasons[verdict['reason']] += 1; rows += 1
                if accepted and len(samples) < 2:
                    samples.append({'task_id':identifier, 'question':row.get('task'),
                        'gold_answer':row.get('gold_answer'),
                        'answer':row['messages'][-1].get('content'),
                        'verification':verdict})
        files.append({'path':str(path), 'rows':rows, 'sha256':digest.hexdigest()})
    expected = set()
    with task_path.open() as stream:
        for line in stream:
            identifier = json.loads(line)['task_id']
            if identifier in expected:
                raise ValueError(f'Duplicate input task ID: {source}')
            expected.add(identifier)
    if not seen <= expected:
        raise ValueError(f'Checkpoint contains unknown task IDs: {source}')
    report_path = output_root / f'{source}.generation.json'
    if report_path.exists():
        report = json.loads(report_path.read_text())
        if (report.get('status') != 'complete' or seen != expected
                or report.get('reasons') != dict(reasons)):
            raise ValueError(f'Completed source report disagrees with JSONL: {source}')
    return {'source':source, 'tasks':len(expected), 'persisted':len(seen),
        'remaining':len(expected - seen), 'accepted':sum(r['rows'] for r in files
            if r['path'].endswith('.accepted.jsonl')), 'reasons':dict(reasons),
        'source_report_complete':report_path.exists(), 'files':files, 'samples':samples}
