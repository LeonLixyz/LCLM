"""Prepare bounded MAUD/TAT-QA diagnostics; never edit or release source traces."""
import hashlib
import json
from pathlib import Path
import modal
from data.stage3_full_modal import ROOT, image, volume

app = modal.App('lclm-label-grounding-calibration-inputs-v1')
GENERATED = Path('/data/stage3-agent/real-expansion/pilots/full-20260906-v6')
EXPECTED = {'maud': 11851, 'tatqa': 4055}


@app.function(image=image, cpu=2, memory=16384, timeout=3600, volumes={'/data': volume})
def prepare():
    collision_bytes = (ROOT/'full-source-review-findings-v1/accounting-candidates.json').read_bytes()
    collisions = json.loads(collision_bytes)
    flagged = {r['task_id'] for r in collisions if r['source'] == 'tatqa'}
    if len(flagged) != 34: raise ValueError('Unexpected accounting diagnostic inventory')
    selected = {}; controls = {}; receipts = []
    for source, count in EXPECTED.items():
        audit = json.loads((ROOT/'full-expansion-format-audit'/f'{source}.json').read_text())
        review = json.loads((ROOT/'full-expansion-manual-review-samples'/f'{source}.review.json').read_text())
        if (audit['status'] != 'passed' or audit['rows'] != count
                or review['status'] != 'sample_review_failed'
                or review['accepted_file_sha256'] != audit['accepted_file_sha256']):
            raise ValueError('Missing or stale audited source hold')
        known = {r['task_id']: r for r in review['reviewed_examples'] + review.get('additional_reviewed_examples', [])}
        if any(r['result'] not in ('pass', 'fail') for r in known.values()):
            raise ValueError('Invalid manual control')
        forced_ids = set(known) | (flagged if source == 'tatqa' else set())
        forced = {}; buckets = {'single': [], 'multi': []}; seen = set(); digest = hashlib.sha256()
        with (GENERATED/f'{source}.accepted.jsonl').open('rb') as stream:
            for line in stream:
                digest.update(line); row = json.loads(line); task_id = row['task_id']
                if not line.endswith(b'\n') or task_id in seen or row['verification']['accepted'] is not True:
                    raise ValueError('Invalid accepted source checkpoint')
                seen.add(task_id)
                segments = {c['function']['arguments']['segment_id'] for m in row['messages'] for c in m.get('tool_calls', [])}
                category = 'single' if len(segments) == 1 else 'multi'
                score = hashlib.sha256(('label-calibration-v1:'+task_id).encode()).hexdigest()
                item = (score, task_id, line, category)
                buckets[category].append(item); buckets[category].sort()
                if len(buckets[category]) > 8: buckets[category].pop()
                if task_id in forced_ids: forced[task_id] = item
        if digest.hexdigest() != audit['accepted_file_sha256'] or len(seen) != count or set(forced) != forced_ids:
            raise ValueError('Changed source bytes or missing diagnostic rows')
        picks = {x[1]: x for bucket in buckets.values() for x in bucket}; picks.update(forced)
        for task_id, (_, _, line, category) in sorted(picks.items()):
            if task_id in selected: raise ValueError('Cross-source duplicate task')
            selected[task_id] = (source, category, line)
        controls.update({task_id: {'source': source, 'expected_keep': r['result'] == 'pass',
                                  'manual_evidence': r['evidence']} for task_id, r in known.items()})
        receipts.append({'source': source, 'accepted_rows_scanned': count, 'selected_rows': len(picks),
                         'input_sha256': digest.hexdigest(),
                         'manual_review_sha256': hashlib.sha256(json.dumps(review, sort_keys=True).encode()).hexdigest()})
    rows = b''.join(selected[k][2] for k in sorted(selected))
    if len(selected) > 72 or len(controls) != 6: raise ValueError('Diagnostic bound/control coverage changed')
    manifest = {'version': 'label-grounding-calibration-v1', 'status': 'prepared',
        'approved_for_generation': False, 'approved_for_release': False, 'rows': len(selected),
        'sources': receipts, 'selected_rows_sha256': hashlib.sha256(rows).hexdigest(),
        'accounting_candidates_sha256': hashlib.sha256(collision_bytes).hexdigest(),
        'selection': 'Up to eight hash-selected rows per source/single-multi bucket, all six manual controls, all 34 TAT-QA accounting-notation candidates; at most 72 rows',
        'candidates': [{'task_id': k, 'source': selected[k][0], 'category': selected[k][1]} for k in sorted(selected)],
        'calibration_controls': controls,
        'instruction_separation': 'Neither references nor control expectations/manual evidence may enter judge prompts or training messages.',
        'next_step': 'Implement a separate bounded Qwen235B no-thinking diagnostic. Check question-conditioned label entailment, requested years/order, table sign conventions and units. Require all controls and fresh manual review before any full-source review. Do not redeploy the active FaithDial/CLAPNQ reviewer.'}
    output = ROOT/'label-grounding-calibration-v1'; output.mkdir(exist_ok=True)
    manifest_path = output/'manifest.json'; row_path = output/'candidates.jsonl'
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest or row_path.read_bytes() != rows:
            raise ValueError('Existing diagnostic differs; use a new version')
    else:
        if row_path.exists(): raise ValueError('Partial diagnostic output requires inspection')
        row_path.write_bytes(rows); manifest_path.write_text(json.dumps(manifest, indent=2)); volume.commit()
    return {k: manifest[k] for k in ('status', 'rows', 'sources', 'selected_rows_sha256', 'accounting_candidates_sha256')}


@app.local_entrypoint()
def main():
    print(json.dumps(prepare.remote(), indent=2))
