"""Five additional hash-selected kept CLAPNQ candidates; no release writes."""
import json
import modal
from data.stage3_full_modal import ROOT, image, volume

app = modal.App('lclm-clapnq-kept-manual-samples')


@app.function(image=image, cpu=2, memory=32768, timeout=1800, volumes={'/data': volume})
def sample():
    from data.full_grounding_review_modal import load_job, OUTPUT
    from data.grounding_subset_accounting import digest, completed_decisions
    from data.grounding_claim_review import primary_evidence, answer_sentences
    from data.grounding_decision_replay import replay_decision
    volume.reload(); items, manifest = load_job()
    report = json.loads((OUTPUT/'report.json').read_text())
    content = (OUTPUT/'decisions.jsonl').read_bytes()
    if (report['manifest'] != manifest
            or digest(content) != '4cbec1396bcb3716c2a8c95ab62c5d470a820eb81470084b38717795f2d4d358'):
        raise ValueError('Changed completed review')
    decisions = completed_decisions(content, report)
    inspection = ROOT/'full-grounding-review-v3-inspection-v1'
    samples_bytes = (inspection/'samples.json').read_bytes()
    if digest(samples_bytes) != '55ea31fe9c2c230897d6fcbbfb6542f1fc1648d5893d960ab69300e810e501a6':
        raise ValueError('Changed initial samples')
    samples = json.loads(samples_bytes)
    excluded = set(samples['calibration_ids_excluded']) | {r['training_row']['task_id'] for r in samples['examples']}
    ranked = sorted((digest(('clapnq-kept-review-v1:'+row['task_id']).encode()), row)
        for source, row in items if source == 'clapnq' and row['task_id'] not in excluded
        and decisions[row['task_id']]['keep'])[:5]
    if len(ranked) != 5: raise ValueError('Insufficient fresh kept examples')
    examples = []
    for score, row in ranked:
        decision = decisions[row['task_id']]; replay_decision(row, decision)
        examples.append({'selection_hash': score, 'training_row': row, 'decision': decision})
    result = {'status': 'awaiting_manual_review', 'source': 'clapnq',
        'decisions_sha256': digest(content), 'initial_samples_sha256': digest(samples_bytes),
        'selection': 'Lowest five SHA256(clapnq-kept-review-v1:task_id), excluding all calibration and initial inspection sample IDs',
        'examples': examples, 'approved_for_release': False}
    primary = [{'task_id': r['training_row']['task_id'], 'question': r['training_row']['task'],
        'answer': answer_sentences(r['training_row'])[0], 'primary_evidence': primary_evidence(r['training_row']),
        'decision': r['decision']} for r in examples]
    destination = ROOT/'clapnq-kept-review-v1'; destination.mkdir(exist_ok=True)
    payloads = {'samples.json': json.dumps(result, ensure_ascii=False, indent=2).encode(),
                'primary-evidence.json': json.dumps(primary, ensure_ascii=False, indent=2).encode()}
    for name, raw in payloads.items():
        path = destination/name
        if path.exists() and path.read_bytes() != raw: raise ValueError('Changed samples; use a new version')
        path.write_bytes(raw)
    volume.commit()
    return {'status': result['status'], 'samples_sha256': digest(payloads['samples.json']),
            'task_ids': [r['training_row']['task_id'] for r in examples]}


@app.local_entrypoint()
def main():
    print(json.dumps(sample.remote(), indent=2))
