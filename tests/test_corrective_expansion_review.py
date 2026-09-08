import pytest
from data.corrective_expansion_review import validate_corrective_review


def receipts():
    generation = {'status': 'complete', 'reasons': {'accepted:qwen_semantic': 21, 'wrong_answer:qwen_semantic': 11},
                  'manifest': {'pilot_limit': 32, 'model': 'Qwen/Qwen3-235B-A22B-Instruct-2507',
                               'model_revision': 'ac9c66cc9b46af7306746a9250f23d47083d689e',
                               'teacher_prompt_saved_in_training_messages': False,
                               'semantic_review': 'question-first-every-claim-v1',
                               'input_provenance': {'source': 'multidoc2dial', 'tasks_sha256': 'tasks',
                                                    'question_rendering_version': 'multidoc2dial-chronological-v1'}}}
    audit = {'status': 'passed', 'source': 'multidoc2dial', 'failed_rows': 0, 'rows': 21, 'expected_accepted': 21,
             'generation_manifest_file_sha256': 'manifest', 'accepted_file_sha256': 'accepted'}
    review = {'approved_for_full_generation': True, 'source': 'multidoc2dial', 'accepted_traces': 21,
              'generation_manifest_file_sha256': 'manifest', 'accepted_file_sha256': 'accepted',
              'reviewed_task_ids': ['task-0', 'task-1']}
    return generation, audit, review


def check(generation, audit, review):
    return validate_corrective_review(generation, audit, review, 'manifest', 'accepted',
                                      {f'task-{i}' for i in range(21)}, 'tasks')


def test_good_receipts():
    assert check(*receipts()) == {'accepted': 21, 'reviewed': 2}


@pytest.mark.parametrize('field,value', [('approved_for_full_generation', False),
    ('generation_manifest_file_sha256', 'stale'), ('accepted_file_sha256', 'stale'),
    ('accepted_traces', 20), ('reviewed_task_ids', ['unknown', 'task-1']), ('reviewed_task_ids', ['task-0'])])
def test_bad_manual_review(field, value):
    generation, audit, review = receipts(); review[field] = value
    with pytest.raises(ValueError):
        check(generation, audit, review)


def test_bad_audit_and_changed_teacher():
    generation, audit, review = receipts(); audit['failed_rows'] = 1
    with pytest.raises(ValueError): check(generation, audit, review)
    generation, audit, review = receipts(); generation['manifest']['model'] = 'wrong'
    with pytest.raises(ValueError): check(generation, audit, review)
    generation, audit, review = receipts(); generation['manifest']['input_provenance']['tasks_sha256'] = 'stale'
    with pytest.raises(ValueError): check(generation, audit, review)
