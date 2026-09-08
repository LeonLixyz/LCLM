"""Read-only diagnostic for new manual-review findings; no generation changes."""
import hashlib
import json
from pathlib import Path
import modal
from data.stage3_full_modal import ROOT, image, volume

app = modal.App('lclm-full-source-review-findings-v1')
GENERATED = Path('/data/stage3-agent/real-expansion/pilots/full-20260906-v6')


@app.function(image=image, cpu=2, memory=16384, timeout=3600, volumes={'/data': volume})
def inspect():
    from datasets import load_from_disk
    from data.accounting_answer_review import accounting_collision
    from data.real_expansion_agent import _final_payload
    output = ROOT/'full-source-review-findings-v1'
    output.mkdir(exist_ok=True)
    counts = {}; sources = []; findings = []
    for source in ('finqa', 'tatqa', 'convfinqa', 'multihiertt'):
        audit = json.loads((ROOT/'full-expansion-format-audit'/f'{source}.json').read_text())
        content = (GENERATED/f'{source}.accepted.jsonl').read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        rows = [json.loads(line) for line in content.splitlines()]
        if audit['status'] != 'passed' or digest != audit['accepted_file_sha256'] or len(rows) != audit['rows']:
            raise ValueError('Changed or unaudited financial input')
        count = 0
        for row in rows:
            candidate = _final_payload(row['messages']); reference = str(row['gold_answer'])
            if accounting_collision(candidate, reference):
                count += 1
                findings.append({'source': source, 'task_id': row['task_id'], 'question': row['task'],
                    'candidate': candidate, 'reference': reference, 'training_row': row})
        counts[source] = {'accepted_rows': len(rows), 'potential_accounting_collisions': count}
        sources.append({'source': source, 'accepted_file_sha256': digest})
    (output/'accounting-candidates.json').write_text(json.dumps(findings, ensure_ascii=False, indent=2))
    views = [{k: v for k, v in f.items() if k != 'training_row'} | {
        'messages_without_user_context': [m for m in f['training_row']['messages'] if m['role'] != 'user']}
        for f in findings]
    (output/'accounting-evidence.json').write_text(json.dumps(views, ensure_ascii=False, indent=2))
    dataset = load_from_disk('/data/stage3-agent/real-expansion/sources/maud/materialized/default/train')
    row = dict(dataset[15517])
    matching = [{'index': i, 'question': other.get('question'), 'answer': other.get('answer'),
                 'contract_name': other.get('contract_name')} for i, other in enumerate(dataset)
                if other.get('text') == row.get('text')]
    (output/'maud-row-15517.json').write_text(json.dumps({'index': 15517, 'row': row,
        'other_annotations_for_same_text': matching}, ensure_ascii=False, indent=2))
    result = {'status': 'diagnostic_complete', 'sources': sources, 'counts': counts,
              'total_accounting_candidates': len(findings), 'maud_same_text_annotations': len(matching),
              'training_rows_modified': False, 'approved_for_release': False,
              'scope': 'Potential accounting-notation collisions and original MAUD row, not automatic errors or exclusions'}
    (output/'report.json').write_text(json.dumps(result, indent=2)); volume.commit()
    return result


@app.local_entrypoint()
def main():
    print(json.dumps(inspect.remote(), indent=2))
