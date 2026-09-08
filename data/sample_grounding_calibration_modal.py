"""Choose additional diagnostic examples without changing decisions or traces."""
import hashlib
import json
import modal
from data.stage3_full_modal import ROOT, image, volume

app = modal.App('lclm-grounding-calibration-manual-samples-v2')


@app.function(image=image, cpu=2, memory=8192, timeout=600, volumes={'/data': volume})
def sample():
    from data.run_grounding_calibration_modal import load_inputs
    from data.grounding_claim_review import primary_evidence
    OUTPUT = ROOT/'grounding-calibration-claims-v2'  # This artifact samples V2, not the latest runner.
    rows, manifest, _ = load_inputs(); by_id = {r['task_id']: r for r in rows}
    report = json.loads((OUTPUT/'report.json').read_text())
    if report['status'] != 'complete' or not report['calibration_passed']:
        raise ValueError('Calibration did not pass')
    content = (OUTPUT/'decisions.jsonl').read_bytes()
    decisions = [json.loads(line) for line in content.splitlines()]
    if len(decisions) != 58 or {r['task_id'] for r in decisions} != set(by_id):
        raise ValueError('Incomplete or duplicated diagnostic results')
    sources = {r['task_id']: r['source'] for r in manifest['candidates']}
    buckets = {}; controls = set(manifest['calibration_controls'])
    for decision in decisions:
        task_id = decision['task_id']
        if task_id in controls or 'error' in decision: continue
        key = (sources[task_id], 'keep' if decision['keep'] else 'reject')
        score = hashlib.sha256(('grounding-extra-review-v1:'+task_id).encode()).hexdigest()
        buckets.setdefault(key, []).append((score, task_id, decision))
    if set(buckets) != {(s, k) for s in ('faithdial', 'clapnq') for k in ('keep', 'reject')}:
        raise ValueError('Missing outcome/source strata')
    examples = []
    for (source, outcome), candidates in sorted(buckets.items()):
        # Two per stratum: eight non-control examples total.
        for score, task_id, decision in sorted(candidates)[:2]:
            row = by_id[task_id]
            examples.append({'source': source, 'outcome': outcome, 'task_id': task_id,
                'selection_hash': score, 'question': row['task'], 'answer': row['messages'][-1]['content'],
                'primary_evidence': primary_evidence(row), 'decision': decision})
    artifact = {'status': 'awaiting_manual_review', 'approved': False,
                'decisions_sha256': hashlib.sha256(content).hexdigest(),
                'protocol_sha256': report['manifest']['protocol_sha256'],
                'examples': examples,
                'errors': [r for r in decisions if 'error' in r]}
    (OUTPUT/'manual-samples.json').write_text(json.dumps(artifact, ensure_ascii=False, indent=2)); volume.commit()
    return {'path': str(OUTPUT/'manual-samples.json'), 'decisions_sha256': artifact['decisions_sha256'],
            'examples': [{k: r[k] for k in ('source', 'outcome', 'task_id')} for r in examples],
            'errors': artifact['errors']}


@app.local_entrypoint()
def main():
    print(json.dumps(sample.remote(), indent=2))
