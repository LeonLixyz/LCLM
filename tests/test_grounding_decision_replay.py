import copy
import json
import pytest
from data.grounding_claim_review import review_claims
from data.grounding_decision_replay import replay_decision


def fixture(supported=True):
    row = {'task_id': 'x', 'task': 'Use these source documents: primary.\nWhat is the value?',
        'messages': [{'role': 'assistant', 'tool_calls': [{'id': 'c', 'function': {
            'name': 'expand', 'arguments': {'segment_id': 'seg_1'}}}]},
            {'role': 'tool', 'tool_call_id': 'c', 'content': 'SOURCE primary:part0\nThe value is 2.'},
            {'role': 'assistant', 'content': 'FINAL: The value is 2.'}]}
    answers = iter([{'supported': supported, 'abstention_only': False,
        'evidence_quotes': [{'segment_id': 'seg_1', 'quote': 'The value is 2.'}]},
        {'correct': True, 'grounded': True, 'issue': 'none'}])
    saved = {'source': 's', **review_claims(row, lambda _: json.dumps(next(answers)))}
    return row, saved


@pytest.mark.parametrize('supported', [True, False])
def test_replays_without_new_inference_or_mutation(supported):
    row, saved = fixture(supported); before = copy.deepcopy((row, saved))
    assert replay_decision(row, saved) == 'replayed'
    assert (row, saved) == before


@pytest.mark.parametrize('change', ['id', 'keep', 'sentence', 'missing', 'extra', 'quote_flag', 'quote', 'version', 'fit'])
def test_tampered_decision_fails(change):
    row, saved = fixture()
    if change == 'id': saved['task_id'] = 'other'
    if change == 'keep': saved['keep'] = False
    if change == 'sentence': saved['sentences'][0]['sentence'] = 'Other answer.'
    if change == 'missing': saved['sentences'] = []
    if change == 'extra': saved['sentences'].append(copy.deepcopy(saved['sentences'][0]))
    if change == 'quote_flag': saved['sentences'][0]['quotes_present'] = False
    if change == 'quote': saved['sentences'][0]['evidence_quotes'][0]['quote'] = 'The value is 3.'
    if change == 'version': saved['version'] = 'other'
    if change == 'fit': saved['answer_fit'] = {'correct': True, 'grounded': True, 'issue': 'wrong_request'}
    with pytest.raises(ValueError): replay_decision(row, saved)


def test_error_is_quarantined_not_replayed_as_rejection():
    row, _ = fixture(); saved = {'source': 's', 'task_id': 'x', 'keep': False, 'error': 'timeout'}
    assert replay_decision(row, saved) == 'error_quarantined'
    saved['keep'] = True
    with pytest.raises(ValueError): replay_decision(row, saved)
