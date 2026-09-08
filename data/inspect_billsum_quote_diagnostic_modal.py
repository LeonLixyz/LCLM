"""CPU-only replay of saved BillSum diagnostic quotations; no inference or recovery."""
import hashlib
import json
import subprocess
import modal
from data.stage3_full_modal import ROOT, image, volume

app = modal.App('lclm-billsum-quote-diagnostic')
SNAPSHOT = 'a7a07b63b20ddde750a654bc977a94c10c90633bf780bf8c7f572d92c34914ad'


@app.function(image=image, cpu=2, memory=8192, timeout=900, volumes={'/data': volume})
def inspect():
    from data.billsum_quote_diagnostic import diagnose_quotes
    subprocess.run(['python', '-m', 'pytest', '-q', '/opt/lclm/tests/test_billsum_quote_diagnostic.py'], check=True)
    volume.reload()
    source = ROOT/'running-source-diagnostics/billsum'/SNAPSHOT
    raw = (source/'samples.json').read_bytes()
    rows = {e['training_row']['task_id']: e['training_row'] for e in json.loads(raw)
            if e['training_row']['verification'].get('semantic_review', {}).get('summary_claims')}
    results = []; prepared_hash = hashlib.sha256()
    from pathlib import Path
    prepared = Path('/data/stage3-agent/real-expansion/pilots/full-20260906-v3/billsum.tasks.jsonl')
    matched = set()
    with prepared.open('rb') as stream:
        for line in stream:
            prepared_hash.update(line)
            task = json.loads(line)
            if task['task_id'] in rows:
                if task['task_id'] in matched: raise ValueError('Duplicate prepared task')
                matched.add(task['task_id']); results.append(diagnose_quotes(rows[task['task_id']], task))
    if matched != rows.keys(): raise ValueError('Missing prepared task')
    report = {'status': 'saved_sample_quote_diagnostic_only', 'snapshot_sha256': SNAPSHOT,
              'samples_sha256': hashlib.sha256(raw).hexdigest(),
              'prepared_file_sha256': prepared_hash.hexdigest(), 'results': results,
              'approved_for_release': False, 'training_rows_modified': False}
    output = source/'contiguous-quote-diagnostic-v1.json'
    payload = json.dumps(report, indent=2).encode()
    if output.exists() and output.read_bytes() != payload: raise ValueError('Changed diagnostic inputs')
    output.write_bytes(payload); volume.commit()
    return report


@app.local_entrypoint()
def main():
    print(json.dumps(inspect.remote(), indent=2))
