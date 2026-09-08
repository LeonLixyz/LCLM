"""CPU-only regression checks on Modal; never starts a generator or publishes."""
import json
import subprocess
import modal
from data.stage3_full_modal import image, volume, ROOT

app = modal.App('lclm-expansion-repair-regressions')


@app.function(image=image, cpu=2, memory=8192, timeout=600, volumes={'/data': volume})
def test():
    from data.run_grounding_calibration_modal import load_inputs
    calibration_rows, _, calibration_sha = load_inputs()
    from data.grounding_claim_review import primary_evidence, validate_sentence_vote
    old_decisions = ROOT/'grounding-calibration-claims-v1/decisions.jsonl'
    quote_replay = []
    if old_decisions.exists():
        by_id = {r['task_id']: r for r in calibration_rows}
        for line in old_decisions.read_text().splitlines():
            decision = json.loads(line)
            if 'error' in decision: continue
            evidence = primary_evidence(by_id[decision['task_id']])
            new = [validate_sentence_vote(s['sentence'], evidence,
                   {k: s[k] for k in ('supported', 'abstention_only', 'evidence_quotes')})
                   for s in decision['sentences']]
            if any(a['keep'] != b['keep'] for a, b in zip(decision['sentences'], new)):
                quote_replay.append({'task_id': decision['task_id'],
                    'old_all_sentences_pass': all(s['keep'] for s in decision['sentences']),
                    'new_all_sentences_pass': all(s['keep'] for s in new)})
    files = ['test_grounding_subset_accounting.py', 'test_question_grounding_structured.py', 'test_question_grounding_review.py', 'test_accounting_answer_review.py', 'test_grounding_full_review_gate.py', 'test_grounding_claim_review.py', 'test_expansion_release_selection.py', 'test_stage3_release_checks.py',
             'test_multidoc2dial_dialogue.py', 'test_expansion_corrective_paths.py', 'test_corrective_expansion_review.py',
             'test_expansion_judge_json.py', 'test_expansion_task_normalization.py',
             'test_harvest_expansion_trace.py']
    result = subprocess.run(['python', '-m', 'pytest', '-q'] + ['/opt/lclm/tests/' + f for f in files],
                            text=True, capture_output=True)
    report = {'exit_code': result.returncode, 'tests': files,
              'real_calibration_preflight': {'rows': len(calibration_rows), 'input_manifest_sha256': calibration_sha},
              'quote_normalization_replay_changes': quote_replay,
              'stdout': result.stdout, 'stderr': result.stderr,
              'scope': 'Corrective provenance/path isolation, release selection/accounting and harvest/prompt regressions; not GPU validation'}
    (ROOT / 'expansion-repair-regressions.json').write_text(json.dumps(report, indent=2))
    volume.commit()
    return report


@app.local_entrypoint()
def main():
    print(json.dumps(test.remote(), indent=2))
