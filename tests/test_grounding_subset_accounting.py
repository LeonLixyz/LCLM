import copy
import json
import pytest
from data.grounding_subset_accounting import digest, completed_decisions, partition_candidate


def encode(rows):
    return b''.join((json.dumps(row)+'\n').encode() for row in rows)


def fixture():
    original = b'{"task_id": "a", "verification":{"accepted":true}, "messages":[{"content":"literal memory & tools"}]}\r\n' + encode([
        {'task_id': k, 'verification': {'accepted': True}} for k in ('b', 'c')])
    decisions = [{'task_id': 'a', 'source': 's', 'keep': True},
                 {'task_id': 'b', 'source': 's', 'keep': False},
                 {'task_id': 'c', 'source': 's', 'keep': False, 'error': 'timeout'}]
    content = encode(decisions)
    report = {'status': 'complete', 'remaining': 0, 'reviewed': 3, 'training_rows_modified': False,
        'manifest': {'candidate_rows': 3, 'training_rows_modified': False, 'sources': [
            {'source': 's', 'rows': 3, 'accepted_file_sha256': digest(original)}]},
        'decisions_sha256': digest(content), 'counts': {'s': {'kept': 1, 'rejected': 1, 'error': 1}}}
    generation = {'status': 'complete', 'source': 's', 'reasons': {'accepted:qwen_semantic': 3, 'no_tool_call': 2}}
    return original, content, report, generation


def test_exact_bytes_disjoint_complete_accounting_and_no_approval():
    original, content, report, generation = fixture(); before = copy.deepcopy((report, generation))
    output = partition_candidate('s', original, content, report, generation)
    assert output['retained'] == original.splitlines(keepends=True)[0]
    assert output['excluded'] == b''.join(original.splitlines(keepends=True)[1:])
    assert (report, generation) == before
    manifest = output['manifest']
    assert manifest['total_original_attempts_accounted'] == 5
    assert manifest['counts'] == {'retained_candidate': 1, 'review_rejected': 1, 'review_error': 1}
    assert manifest['approved_for_release'] is False and manifest['original_rows_rewritten'] is False
    ledger = list(map(json.loads, output['ledger'].splitlines()))
    assert [r['task_id'] for r in ledger] == ['a', 'b', 'c']
    assert ledger[0]['original_row_sha256'] == digest(output['retained'])
    assert ledger[2]['decision']['error'] == 'timeout'


@pytest.mark.parametrize('key,value', [('status', 'running'), ('remaining', 1), ('reviewed', 2),
    ('decisions_sha256', 'wrong'), ('training_rows_modified', True), ('counts', {}),
    ('remaining', False), ('counts', {'s': {'kept': True, 'rejected': 1, 'error': 1}})])
def test_partial_or_changed_review_fails(key, value):
    original, content, report, generation = fixture(); report[key] = value
    with pytest.raises(ValueError): partition_candidate('s', original, content, report, generation)


@pytest.mark.parametrize('change', ['duplicate', 'missing', 'extra', 'source', 'nonboolean', 'kept_error', 'blank_error', 'truncated'])
def test_decision_integrity(change):
    _, content, report, _ = fixture(); rows = list(map(json.loads, content.splitlines()))
    if change == 'duplicate': rows[1] = rows[0]
    if change == 'missing': rows.pop()
    if change == 'extra': rows.append({'task_id': 'd', 'source': 's', 'keep': True})
    if change == 'source': rows[0]['source'] = 'other'
    if change == 'nonboolean': rows[0]['keep'] = 1
    if change == 'kept_error': rows[0]['error'] = 'bad'
    if change == 'blank_error': rows[2]['error'] = ''
    content = encode(rows)
    if change == 'truncated': content = content[:-1]
    report['decisions_sha256'] = digest(content)
    with pytest.raises(ValueError): completed_decisions(content, report)


@pytest.mark.parametrize('change', ['hash', 'duplicate', 'id', 'unverified', 'truncated'])
def test_original_row_integrity(change):
    original, content, report, generation = fixture(); rows = list(map(json.loads, original.splitlines()))
    if change == 'hash': original += b' '
    else:
        if change == 'duplicate': rows[1] = rows[0]
        if change == 'id': rows[0]['task_id'] = 'other'
        if change == 'unverified': rows[0]['verification']['accepted'] = False
        original = encode(rows)
        if change == 'truncated': original = original[:-1]
        report['manifest']['sources'][0]['accepted_file_sha256'] = digest(original)
    with pytest.raises(ValueError): partition_candidate('s', original, content, report, generation)


@pytest.mark.parametrize('change', ['incomplete', 'wrong_source', 'wrong_count', 'negative', 'bool'])
def test_original_generation_accounting(change):
    original, content, report, generation = fixture()
    if change == 'incomplete': generation['status'] = 'running'
    if change == 'wrong_source': generation['source'] = 'other'
    if change == 'wrong_count': generation['reasons']['accepted:qwen_semantic'] = 4
    if change == 'negative': generation['reasons']['no_tool_call'] = -1
    if change == 'bool': generation['reasons']['no_tool_call'] = True
    with pytest.raises(ValueError): partition_candidate('s', original, content, report, generation)


@pytest.mark.parametrize('keep', [True, False])
def test_empty_partition_is_valid_and_decision_order_does_not_reorder_rows(keep):
    original, content, report, generation = fixture()
    rows = [{'task_id': k, 'source': 's', 'keep': keep} for k in ('c', 'a', 'b')]
    content = encode(rows); report['decisions_sha256'] = digest(content)
    report['counts'] = {'s': {'kept' if keep else 'rejected': 3}}
    result = partition_candidate('s', original, content, report, generation)
    assert result['retained' if keep else 'excluded'] == original
    assert result['excluded' if keep else 'retained'] == b''


def test_complete_multi_source_review_and_source_identity_is_enforced():
    original, content, report, generation = fixture()
    rows = list(map(json.loads, content.splitlines()))
    rows.append({'task_id': 'd', 'source': 'other', 'keep': True})
    content = encode(rows); report['decisions_sha256'] = digest(content)
    report['reviewed'] = report['manifest']['candidate_rows'] = 4
    report['manifest']['sources'].append({'source': 'other', 'rows': 1, 'accepted_file_sha256': 'other'})
    report['counts']['other'] = {'kept': 1}
    result = partition_candidate('s', original, content, report, generation)
    assert result['manifest']['original_accepted_rows'] == 3
    rows[0]['task_id'], rows[3]['task_id'] = rows[3]['task_id'], rows[0]['task_id']
    content = encode(rows); report['decisions_sha256'] = digest(content)
    with pytest.raises(ValueError): partition_candidate('s', original, content, report, generation)
