"""Read-only source scan; prepare bounded diagnostics, never edit/release traces."""
import hashlib
import json
from pathlib import Path

import modal
from data.stage3_full_modal import ROOT, image, volume

app = modal.App('lclm-grounding-calibration-inputs-v1')
GENERATED = Path('/data/stage3-agent/real-expansion/pilots/full-20260906-v6')
SOURCES = ('faithdial', 'clapnq')


@app.function(image=image, cpu=2, memory=8192, timeout=3600, volumes={'/data': volume})
def prepare():
    selected = {}; receipts = []; controls = {}
    for source in SOURCES:
        audit = json.loads((ROOT/'full-expansion-format-audit'/f'{source}.json').read_text())
        review = json.loads((ROOT/'full-expansion-manual-review-samples'/f'{source}.review.json').read_text())
        if (audit['status'] != 'passed' or review['status'] != 'sample_review_failed'
                or audit['accepted_file_sha256'] != review['accepted_file_sha256']):
            raise ValueError('Calibration must reference audited, manually held source bytes')
        known = {r['task_id']: r for r in review['reviewed_examples']}
        buckets = {'single': [], 'multi': []}; forced = {}; seen = set(); digest = hashlib.sha256()
        with (GENERATED/f'{source}.accepted.jsonl').open('rb') as stream:
            for line in stream:
                digest.update(line); row = json.loads(line); task_id = row['task_id']
                if task_id in seen or row['verification']['accepted'] is not True:
                    raise ValueError('Duplicate/unverified calibration input')
                seen.add(task_id)
                segments = {c['function']['arguments']['segment_id'] for m in row['messages']
                            for c in m.get('tool_calls', [])}
                category = 'single' if len(segments) == 1 else 'multi'
                score = hashlib.sha256(('claim-calibration-v1:'+task_id).encode()).hexdigest()
                candidate = (score, task_id, line, category)
                buckets[category].append(candidate); buckets[category].sort()
                if len(buckets[category]) > 16: buckets[category].pop()
                if task_id in known: forced[task_id] = candidate
        if digest.hexdigest() != audit['accepted_file_sha256'] or len(seen) != audit['rows'] or set(forced) != set(known):
            raise ValueError('Changed/incomplete source or missing calibration controls')
        picks = {x[1]: x for bucket in buckets.values() for x in bucket}; picks.update(forced)
        for task_id, (_, _, line, category) in sorted(picks.items()):
            if task_id in selected: raise ValueError('Cross-source duplicate task ID')
            selected[task_id] = (source, category, line)
        controls.update({task_id: {'source': source, 'expected_keep': r['result'] == 'pass',
                                  'manual_evidence': r['evidence']} for task_id, r in known.items()})
        receipts.append({'source': source, 'accepted_rows_scanned': len(seen), 'selected_rows': len(picks),
                         'input_sha256': digest.hexdigest(), 'manual_review_sha256':
                         hashlib.sha256(json.dumps(review, sort_keys=True).encode()).hexdigest()})
    rows = b''.join(selected[k][2] for k in sorted(selected))
    manifest = {'version': 'claim-calibration-v1', 'status': 'prepared', 'approved_for_generation': False,
                'approved_for_release': False, 'sources': receipts, 'rows': len(selected),
                'selected_rows_sha256': hashlib.sha256(rows).hexdigest(),
                'selection': 'lowest SHA256(claim-calibration-v1:task_id), up to 16 per single/multi source bucket, plus every manually reviewed positive/negative control',
                'candidates': [{'task_id': k, 'source': selected[k][0], 'category': selected[k][1]} for k in sorted(selected)],
                'calibration_controls': controls,
                'instruction_separation': 'Controls and manual labels are for evaluator calibration only. Never include them in judge inputs or training messages.',
                'next_step': 'Implement/pilot evidence-quoted per-claim judgment with no thinking on pinned Qwen235B. Require rejecting both known unsupported/ambiguous answers and retaining grounded controls, plus manual review, before any all-source re-review. No automatic release approval; entailment remains a heuristic.'}
    destination = ROOT/'grounding-calibration-v1'
    destination.mkdir(exist_ok=True)
    report = destination/'manifest.json'; target = destination/'candidates.jsonl'
    if report.exists():
        if json.loads(report.read_text()) != manifest or hashlib.sha256(target.read_bytes()).hexdigest() != manifest['selected_rows_sha256']:
            raise ValueError('Existing calibration artifact differs; use a new version')
        return manifest
    if target.exists(): raise ValueError('Partial calibration output requires inspection')
    target.write_bytes(rows); report.write_text(json.dumps(manifest, indent=2)); volume.commit()
    return manifest


@app.local_entrypoint()
def main():
    result = prepare.remote()
    print(json.dumps({k: result[k] for k in ('status', 'rows', 'sources', 'selected_rows_sha256')}, indent=2))
