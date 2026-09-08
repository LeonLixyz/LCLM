from copy import deepcopy
import pytest
from data.grounding_full_review_gate import validate_scale_approval, PROTOCOL_SHA, INPUT_SHA, TARGETS


def fixture():
    decisions = [{'task_id': f'r{i}', 'keep': bool(i % 2)} for i in range(58)]
    controls = {f'r{i}': {'expected_keep': bool(i % 2), 'actual_keep': bool(i % 2), 'passed': True} for i in range(4)}
    report = {'status': 'complete', 'rows': 58, 'kept': 29, 'errors': 0, 'calibration_passed': True,
              'controls': controls, 'manifest': {'protocol_sha256': PROTOCOL_SHA, 'input_rows_sha256': INPUT_SHA,
              'model': 'Qwen/Qwen3-235B-A22B-Instruct-2507',
              'model_revision': 'ac9c66cc9b46af7306746a9250f23d47083d689e', 'enable_thinking': False,
              'training_rows_modified': False, 'control_labels_sent_to_judge': False}}
    samples = {'decisions_sha256': 'a'*64, 'protocol_sha256': PROTOCOL_SHA,
               'examples': [{'task_id': f'r{i}'} for i in range(4,12)]}
    review = {'approved_for_full_source_review': True, 'approved_for_release': False,
              'protocol_sha256': PROTOCOL_SHA, 'decisions_sha256': 'a'*64,
              'input_rows_sha256': INPUT_SHA, 'target_sources': TARGETS.copy(),
              'reviewed_task_ids': [f'r{i}' for i in range(12)]}
    return report, decisions, review, samples, 'a'*64, PROTOCOL_SHA


def test_review_only_scale_gate_binds_all_required_evidence():
    assert validate_scale_approval(*fixture()) == {'candidate_rows': 8506, 'reviewed_examples': 12}


@pytest.mark.parametrize('failure', ['partial', 'failed_controls', 'duplicate', 'keep_count', 'wrong_model',
    'thinking', 'prompt_leak', 'unapproved', 'release_approval', 'stale_hash', 'wrong_source_count',
    'missing_review', 'missing_sample', 'duplicate_review', 'wrong_protocol', 'error_control'])
def test_scale_gate_rejects_missing_or_changed_inputs(failure):
    r,d,v,s,h,p = deepcopy(fixture())
    if failure == 'partial': r['status'] = 'running'
    elif failure == 'failed_controls': r['controls']['r0']['passed'] = False
    elif failure == 'duplicate': d[-1] = d[0]
    elif failure == 'keep_count': r['kept'] += 1
    elif failure == 'wrong_model': r['manifest']['model_revision'] = 'main'
    elif failure == 'thinking': r['manifest']['enable_thinking'] = True
    elif failure == 'prompt_leak': r['manifest']['control_labels_sent_to_judge'] = True
    elif failure == 'unapproved': v['approved_for_full_source_review'] = False
    elif failure == 'release_approval': v['approved_for_release'] = True
    elif failure == 'stale_hash': v['decisions_sha256'] = 'b'*64
    elif failure == 'wrong_source_count': v['target_sources']['faithdial'] -= 1
    elif failure == 'missing_review': v['reviewed_task_ids'].pop()
    elif failure == 'missing_sample': s['examples'].pop()
    elif failure == 'duplicate_review': v['reviewed_task_ids'].append('r0')
    elif failure == 'wrong_protocol': p = 'b'*64
    else: d[0]['error'] = 'parse'; r['errors'] = 1
    with pytest.raises(ValueError): validate_scale_approval(r,d,v,s,h,p)
