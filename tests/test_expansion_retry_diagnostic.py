import pytest
from data.expansion_retry_diagnostic import fair_quotas, select_failures, summarize
from data.expansion_retry_diagnostic import (
    response_token_ids, ensure_complete_response, RawCaptureError,
    TruncatedResponseError, IncompleteResponseError)


def row(i, kind='retry_rejected', reason='wrong_answer'):
    return {'task_id': str(i), '_attempt_provenance': {'kind': kind, 'previous_reason': reason}}


def test_failed_only_order_independent_and_rare_reason_coverage():
    rows = [row(i) for i in range(100)] + [row(100, reason='invalid_tool_call')]
    rows += [row(101, 'previously_unattempted'), row(102, 'previously_accepted')]
    selected, counts, quotas = select_failures(iter(rows), 10)
    assert selected == select_failures(iter(reversed(rows)), 10)[0]
    assert counts == {'wrong_answer': 100, 'invalid_tool_call': 1}
    assert quotas == {'invalid_tool_call': 1, 'wrong_answer': 9}
    assert '100' in {r['task_id'] for r in selected}
    assert not {'101', '102'} & {r['task_id'] for r in selected}


def test_exact_budget_and_insufficient_failures():
    q = fair_quotas({'a': 1000, 'b': 1000, 'small': 14, 'unattempted': 0}, 1000)
    assert q == {'a': 493, 'b': 493, 'small': 14}
    with pytest.raises(ValueError):
        select_failures([row(1, 'previously_unattempted')], 1)


def test_judge_failure_not_counted_as_tool_failure_or_acceptance():
    result = summarize([{'source': 'clapnq', 'previous_reason': 'wrong_answer',
        'tool_call_count': 1, 'rule_verification': {'accepted': True},
        'verification': {'accepted': False, 'reason': 'diagnostic_error:judge:JSONDecodeError'},
        'error': {'phase': 'judge'}}])
    assert result['counts']['tasks_with_valid_expansion'] == 1
    assert result['counts']['automatic_accepted'] == 0
    assert result['counts']['error_phase:judge'] == 1


@pytest.mark.parametrize('ids', [None, '12', [True], [-1], [1.5]])
def test_raw_capture_rejects_invalid_tokens(ids):
    with pytest.raises(RawCaptureError):
        response_token_ids({'response_token_ids': ids})


def test_token_capture_and_incomplete_responses():
    assert response_token_ids({'response_token_ids': [1, 2]}) == [1, 2]
    ensure_complete_response('stop', 'rollout')
    ensure_complete_response('tool_calls', 'rollout')
    with pytest.raises(TruncatedResponseError):
        ensure_complete_response('length', 'rollout')
    for reason in ['abort', 'content_filter', None]:
        with pytest.raises(IncompleteResponseError):
            ensure_complete_response(reason, 'judge')
