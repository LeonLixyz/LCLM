"""Read-only ACORD query/grade coverage and nonzero-grade manual samples."""
import json
from collections import Counter
import modal
from data.stage3_full_modal import ROOT, image, volume

app = modal.App('lclm-acord-coverage-inspection')


@app.function(image=image, cpu=2, memory=16384, timeout=1800, volumes={'/data': volume})
def inspect():
    import subprocess
    from data.audit_full_expansion_modal import GENERATED
    from data.expansion_checkpoint_audit import audit_source
    from data.grounding_subset_accounting import digest
    from data.full_expansion_tasks import load
    from data.grounding_claim_review import primary_evidence, answer_sentences
    from data.acord_question_coverage import parse_question
    subprocess.run(['python', '-m', 'pytest', '-q', '/opt/lclm/tests/test_acord_question_coverage.py'], check=True)
    volume.reload(); task_path = GENERATED.parent/'full-20260906-v3/acord.tasks.jsonl'
    ledger = audit_source(task_path, GENERATED, 'acord')
    if not ledger['source_report_complete'] or ledger['remaining']: raise ValueError('Incomplete ACORD source')
    raw_queries = Counter(); raw_grades = Counter()
    for row in load('acord', 'materialized_raw/tasks/train'):
        raw_queries[row['query']] += 1; raw_grades[str(row['relevance'])] += 1
    prepared_queries = Counter(); prepared_grades = Counter(); prepared_wording = Counter()
    with task_path.open() as stream:
        for line in stream:
            row = json.loads(line)
            query, wording = parse_question(row['question'], allow_legacy=True)
            prepared_queries[query] += 1; prepared_wording[wording] += 1
            prepared_grades[row['gold_answer']] += 1
    if raw_queries != prepared_queries or raw_grades != prepared_grades: raise ValueError('ACORD task/query/grade mapping differs')
    selected = {}; accepted_grades = Counter(); accepted_queries = Counter()
    audit = json.loads((ROOT/'full-expansion-format-audit/acord.json').read_text())
    accepted_path = GENERATED/'acord.accepted.jsonl'
    import hashlib
    hasher = hashlib.sha256()
    with accepted_path.open('rb') as stream:
        for line in stream:
            hasher.update(line); row = json.loads(line); grade = row['verification']['parsed_final']
            query, _ = parse_question(row['task'])
            accepted_grades[grade] += 1; accepted_queries[query] += 1
            if grade == '0': continue
            score = digest(('acord-grade-review-v1:'+row['task_id']).encode())
            if grade not in selected or score < selected[grade][0]: selected[grade] = (score, row)
    if audit['status'] != 'passed' or hasher.hexdigest() != audit['accepted_file_sha256']:
        raise ValueError('Changed audited accepted rows')
    examples = [{'grade': grade, 'selection_hash': value[0], 'training_row': value[1]}
                for grade, value in sorted(selected.items())]
    samples = {'status': 'awaiting_manual_review', 'accepted_file_sha256': hasher.hexdigest(),
        'selection': 'Lowest SHA256(acord-grade-review-v1:task_id) per nonzero accepted relevance grade', 'examples': examples}
    samples_bytes = json.dumps(samples, ensure_ascii=False, indent=2).encode()
    report = {'status': 'coverage_checked', 'ledger': ledger,
        'prepared_queries': len(prepared_queries), 'accepted_queries': len(accepted_queries),
        'prepared_grades': dict(prepared_grades), 'accepted_grades': dict(accepted_grades),
        'prepared_question_wording': dict(prepared_wording), 'saved_questions_all_normalized_0_4': True,
        'top_prepared_queries': prepared_queries.most_common(10),
        'samples_sha256': digest(samples_bytes), 'training_rows_modified': False, 'approved_for_release': False}
    primary = [{'task_id': e['training_row']['task_id'], 'grade': e['grade'],
        'question': e['training_row']['task'], 'answer': answer_sentences(e['training_row'])[0],
        'primary_evidence': primary_evidence(e['training_row'])} for e in examples]
    destination = ROOT/'acord-coverage-v1'; destination.mkdir(exist_ok=True)
    for name, raw in {'report.json': json.dumps(report, indent=2).encode(), 'samples.json': samples_bytes,
                     'primary-evidence.json': json.dumps(primary, indent=2, ensure_ascii=False).encode()}.items():
        path = destination/name
        if path.exists() and path.read_bytes() != raw: raise ValueError('Changed coverage; use a new version')
        path.write_bytes(raw)
    volume.commit()
    return {k: v for k, v in report.items() if k != 'ledger'}


@app.local_entrypoint()
def main():
    print(json.dumps(inspect.remote(), indent=2))
