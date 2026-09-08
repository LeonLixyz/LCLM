import copy
import json
import pytest
from data.question_grounding_review import claim_decision, fit_decision, review_question, messages, CLAIM_INSTRUCTION


def row():
    return {'task_id': 'x', 'task': 'Use these source documents: primary.\nGive 2018 then 2019.',
        'gold_answer': 'SECRET_REFERENCE', 'expected_keep': 'SECRET_CONTROL',
        'messages': [{'role': 'assistant', 'tool_calls': [{'id': 'c', 'function': {
            'name': 'expand', 'arguments': {'segment_id': 'seg_1'}}}]},
            {'role': 'tool', 'tool_call_id': 'c', 'content': 'SOURCE primary:part0\n2018: 2. 2019: 3.\nRELATED SOURCE other:\nSECRET_PADDING'},
            {'role': 'assistant', 'content': 'FINAL: 2, 3'}]}


def vote(**kwargs):
    return {'supported': True, 'abstention_only': False, 'issue': 'none',
            'evidence_quotes': [{'segment_id': 'seg_1', 'quote': '2018: 2. 2019: 3.'}], **kwargs}


def test_every_call_has_question_and_answer_without_reference_or_control():
    r = row(); before = copy.deepcopy(r); requests = []
    answers = iter([json.dumps(vote()), '{"correct":true,"grounded":true,"issue":"none"}'])
    def complete(m): requests.append(m); return next(answers)
    assert review_question(r, complete)['keep']
    assert r == before and len(requests) == 2
    for request in requests:
        payload = json.loads(request[1]['content'])
        assert payload['question'] == r['task'] and payload['answer'] == '2, 3'
    assert 'SECRET_' not in json.dumps(requests)


@pytest.mark.parametrize('issue', ['wrong_scope', 'wrong_period', 'wrong_order', 'wrong_value_or_unit', 'wrong_boundary'])
def test_supported_quote_does_not_override_wrong_question_condition(issue):
    result = claim_decision('3, 2', {'seg_1': '2018: 2. 2019: 3.'}, vote(supported=False, issue=issue))
    assert result['quotes_present'] and not result['keep']


@pytest.mark.parametrize('change', [{'supported': 'true'}, {'issue': 'invented'}, {'issue': []},
    {'issue': 'wrong_order'}, {'abstention_only': 1}, {'extra': True}, {'evidence_quotes': []}])
def test_malformed_or_unquoted_support_fails_closed(change):
    if change == {'evidence_quotes': []}:
        assert not claim_decision('2, 3', {}, vote(**change))['keep']
    else:
        with pytest.raises(ValueError): claim_decision('2, 3', {}, vote(**change))


@pytest.mark.parametrize('decision', [
    {'correct': True, 'grounded': True, 'issue': 'wrong_boundary'},
    {'correct': False, 'grounded': True, 'issue': 'none'},
    {'correct': 'true', 'grounded': True, 'issue': 'none'},
    {'correct': True, 'grounded': True, 'issue': []}])
def test_fit_schema_and_consistency(decision):
    with pytest.raises(ValueError): fit_decision(decision)


def test_claim_failure_skips_whole_answer_and_preserves_row():
    r = row(); before = copy.deepcopy(r)
    result = review_question(r, lambda _: json.dumps(vote(supported=False, issue='wrong_order')))
    assert not result['keep'] and result['answer_fit'] is None and r == before


def test_prompt_explicitly_covers_found_failure_modes_without_control_values():
    m = messages(CLAIM_INSTRUCTION, 'Question?', 'Answer', {'seg_1': 'Evidence'}, 'Answer')
    for phrase in ('roll-forward', 'respectively', 'inclusive', 'thousands', 'another condition'):
        assert phrase in m[0]['content']
