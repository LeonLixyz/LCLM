import copy
import json
import pytest
from data.question_grounding_structured import CHECKS, SCHEMA, review_structured, validate_decision


def vote(**changes):
    return {'checks': dict.fromkeys(CHECKS, True), 'abstention_only': False, 'issue': 'none',
            'evidence_quotes': [{'segment_id': 'seg_1', 'quote': 'Go to the U.S. Embassy.'}], **changes}


def row():
    return {'task_id': 'x', 'task': 'Use these source documents: primary.\nWhere? Return FINAL.',
            'gold_answer': 'SECRET_REFERENCE', 'expected_keep': 'SECRET_CONTROL',
            'messages': [{'role': 'assistant', 'tool_calls': [{'id': 'c', 'function': {
                'name': 'expand', 'arguments': {'segment_id': 'seg_1'}}}]},
                {'role': 'tool', 'tool_call_id': 'c',
                 'content': 'SOURCE primary:part0\nGo to the U.S. Embassy.\nRELATED SOURCE other:\nSECRET_PADDING'},
                {'role': 'assistant', 'content': 'FINAL: Go to the U.S. Embassy.'}]}


def test_whole_answer_single_call_no_abbreviation_split_no_metadata_leak():
    r = row(); before = copy.deepcopy(r); calls = []
    def complete(messages, schema):
        calls.append((messages, schema)); return json.dumps(vote())
    assert review_structured(r, complete)['keep']
    assert r == before and len(calls) == 1 and calls[0][1] == SCHEMA
    payload = json.loads(calls[0][0][1]['content'])
    assert payload['answer'] == 'Go to the U.S. Embassy.' and 'sentence' not in payload
    assert 'SECRET_' not in json.dumps(calls)


@pytest.mark.parametrize('key', CHECKS)
def test_one_failed_check_rejects_despite_valid_quote(key):
    checks = dict.fromkeys(CHECKS, True); checks[key] = False
    result = validate_decision('Go to the U.S. Embassy.', {'seg_1': 'Go to the U.S. Embassy.'},
                               vote(checks=checks, issue='unsupported_claim'))
    assert result['quotes_present'] and not result['keep']


@pytest.mark.parametrize('changes', [
    {'checks': {}}, {'checks': dict.fromkeys(CHECKS, 1)}, {'checks': []},
    {'issue': 'wrong_scope'}, {'issue': []}, {'abstention_only': 'false'}, {'extra': True}])
def test_schema_or_inconsistent_decision_is_error(changes):
    with pytest.raises(ValueError): validate_decision('answer', {}, vote(**changes))


@pytest.mark.parametrize('quotes', [[], [{'segment_id': 'seg_2', 'quote': 'Go to the U.S. Embassy.'}],
    [{'segment_id': 'seg_1', 'quote': 'Go to the US Embassy.'}]])
def test_no_valid_quote_fails_closed(quotes):
    assert not validate_decision('answer', {'seg_1': 'Go to the U.S. Embassy.'}, vote(evidence_quotes=quotes))['keep']


def test_schema_requires_exact_keys_and_boolean_checks():
    assert SCHEMA['additionalProperties'] is False
    checks = SCHEMA['properties']['checks']
    assert checks['additionalProperties'] is False and set(checks['required']) == set(CHECKS)
    assert all(p == {'type': 'boolean'} for p in checks['properties'].values())
