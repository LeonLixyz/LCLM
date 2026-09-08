"""CPU-only completed-review integrity audit and fresh manual samples.

Never materializes a training subset or approves release. A running review
returns waiting without reading its partial decisions or writing artifacts.
"""
import json
from collections import Counter
import modal
from data.stage3_full_modal import ROOT, image, volume

INPUT = ROOT/'full-grounding-review-v3'
OUTPUT = ROOT/'full-grounding-review-v3-inspection-v1'
app = modal.App('lclm-full-grounding-review-inspection')


@app.function(image=image, cpu=2, memory=32768, timeout=3600, volumes={'/data': volume})
def inspect():
    from data.full_grounding_review_modal import load_job, GENERATED
    from data.run_grounding_calibration_modal import load_inputs
    from data.grounding_subset_accounting import digest, completed_decisions, partition_candidate
    from data.grounding_decision_replay import replay_decision
    from data.grounding_claim_review import primary_evidence, answer_sentences
    from data.expansion_checkpoint_audit import audit_source
    volume.reload()
    if not (INPUT/'report.json').exists():
        return {'status': 'waiting_for_complete_review', 'training_rows_modified': False}
    report = json.loads((INPUT/'report.json').read_text())
    items, expected_manifest = load_job()
    if (report['manifest'] != expected_manifest
            or json.loads((INPUT/'manifest.json').read_text()) != expected_manifest):
        raise ValueError('Completed review manifest differs from calibrated job')
    decision_bytes = (INPUT/'decisions.jsonl').read_bytes()
    decisions = completed_decisions(decision_bytes, report)
    if set(decisions) != {row['task_id'] for _, row in items}:
        raise ValueError('Review IDs do not match original sources')
    calibration, _, _ = load_inputs(); old_ids = {row['task_id'] for row in calibration}
    replay_counts = Counter(); selected = {}; buckets = Counter()
    for source, row in items:
        decision = decisions[row['task_id']]
        if decision['source'] != source: raise ValueError('Source identity mismatch')
        replay_counts[replay_decision(row, decision)] += 1
        if row['task_id'] in old_ids: continue
        outcome = 'error' if 'error' in decision else 'kept' if decision['keep'] else 'rejected'
        segments = {c['function']['arguments']['segment_id'] for m in row['messages'] for c in m.get('tool_calls', [])}
        category = 'single' if len(segments) == 1 else 'multi'
        key = (source, outcome, category); buckets['/'.join(key)] += 1
        score = digest(('full-grounding-inspection-v1:'+row['task_id']).encode())
        if key not in selected or score < selected[key][0]: selected[key] = (score, row, decision)
    sources = []
    for source in sorted({source for source, _ in items}):
        audit = audit_source(GENERATED.parent/'full-20260906-v3'/f'{source}.tasks.jsonl', GENERATED, source)
        if not audit['source_report_complete'] or audit['remaining']:
            raise ValueError('Original generation ledger incomplete')
        original = (GENERATED/f'{source}.accepted.jsonl').read_bytes()
        generation = json.loads((GENERATED/f'{source}.generation.json').read_text())
        partition = partition_candidate(source, original, decision_bytes, report, generation)
        sources.append({'source': source, 'original_generation_audit': audit,
                        'candidate_partition_accounting': partition['manifest']})
        del partition, original
    examples = [{'source': key[0], 'outcome': key[1], 'category': key[2], 'selection_hash': value[0],
                 'training_row': value[1], 'decision': value[2]} for key, value in sorted(selected.items())]
    samples = {'status': 'awaiting_manual_review', 'decisions_sha256': digest(decision_bytes),
        'calibration_ids_excluded': sorted(old_ids), 'selection': 'Lowest SHA256(full-grounding-inspection-v1:task_id) per source/outcome/single-multi bucket, excluding all 58 calibration IDs',
        'examples': examples, 'approved_for_release': False}
    primary = [{'source': r['source'], 'outcome': r['outcome'], 'category': r['category'],
        'task_id': r['training_row']['task_id'], 'question': r['training_row']['task'],
        'answer': answer_sentences(r['training_row'])[0], 'primary_evidence': primary_evidence(r['training_row']),
        'decision': r['decision']} for r in examples]
    samples_bytes = json.dumps(samples, indent=2, ensure_ascii=False).encode()
    summary = {'status': 'integrity_passed_awaiting_manual_review', 'review_report': report,
        'sources': sources, 'replay_counts': dict(replay_counts), 'eligible_sample_buckets': dict(buckets),
        'samples_sha256': digest(samples_bytes), 'sample_count': len(examples),
        'training_rows_modified': False, 'approved_for_release': False,
        'limits': 'Replays quote/schema/coverage decisions without new inference, not proof of factual correctness. Candidate partitions only computed in memory; no training subset written.'}
    payloads = {'report.json': json.dumps(summary, indent=2).encode(), 'samples.json': samples_bytes,
                'primary-evidence.json': json.dumps(primary, indent=2, ensure_ascii=False).encode()}
    OUTPUT.mkdir(exist_ok=True)
    for name, content in payloads.items():
        path = OUTPUT/name
        if path.exists() and path.read_bytes() != content: raise ValueError('Changed inspection; use a new version')
        path.write_bytes(content)
    volume.commit()
    return {k: summary[k] for k in ('status', 'replay_counts', 'sample_count', 'samples_sha256', 'training_rows_modified')}


@app.local_entrypoint()
def main():
    print(json.dumps(inspect.remote(), indent=2))
