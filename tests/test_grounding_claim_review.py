import copy
import json
import pytest
from data.grounding_claim_review import (
    primary_evidence, answer_sentences, sentence_messages, validate_sentence_vote, review_claims, parse_object, quote_normalize)


def row():
    return {'task_id': 'example', 'task': 'Use these source documents: primary.\nQuestion?',
            'gold_answer': 'DO NOT SEND THIS REFERENCE', 'manual_expected_keep': False,
            'messages': [
                {'role': 'assistant', 'tool_calls': [
                    {'id': 'c1', 'function': {'name': 'expand', 'arguments': {'segment_id': 'seg_1'}}},
                    {'id': 'c2', 'function': {'name': 'expand', 'arguments': {'segment_id': 'seg_2'}}}]},
                {'role': 'tool', 'tool_call_id': 'c1', 'content': 'SOURCE primary:part0\nThe value is 42.\nRELATED SOURCE other:\nPadding.'},
                {'role': 'tool', 'tool_call_id': 'c2', 'content': 'SOURCE other:part0\nDistractor.'},
                {'role': 'assistant', 'content': 'FINAL: The value is 42.'}]}


def vote(**kwargs):
    return {'supported': True, 'abstention_only': False,
            'evidence_quotes': [{'segment_id': 'seg_1', 'quote': 'value is 42'}], **kwargs}


def test_primary_evidence_excludes_padding_and_distractors():
    assert primary_evidence(row()) == {'seg_1': 'The value is 42.'}


@pytest.mark.parametrize('quote,segment,keep', [('value is 42', 'seg_1', True),
    ('invented', 'seg_1', False), ('value is 42', 'seg_2', False), ('Padding.', 'seg_1', False)])
def test_quotes_are_scoped_to_exact_primary_segment(quote, segment, keep):
    result = validate_sentence_vote('The value is 42.', primary_evidence(row()),
        vote(evidence_quotes=[{'quote': quote, 'segment_id': segment}]))
    assert result['keep'] is keep


@pytest.mark.parametrize('sentence,keep', [
    ('The date is not provided.', True), ('I cannot answer the requested date.', True),
    ('The provided context does not include its date and reviews.', True),
    ('The source does not specify the color of the apple.', True),
    ('The source does not specify the color, but it is red.', False),
    ('It applies to everyone.', False), ('I cannot tell, but it applies to everyone.', False),
    ('The date is not available and it applies to everyone.', False)])
def test_abstention_cannot_be_generic_quote_bypass(sentence, keep):
    result = validate_sentence_vote(sentence, primary_evidence(row()),
        vote(abstention_only=True, evidence_quotes=[]))
    assert result['keep'] is keep


@pytest.mark.parametrize('change', [{'supported': 'true'}, {'abstention_only': 0},
    {'evidence_quotes': 'quote'}, {'extra': True}, {'evidence_quotes': [{'quote': ''}]}])
def test_malformed_vote_fails_closed(change):
    with pytest.raises(ValueError): validate_sentence_vote('Answer.', {}, vote(**change))


def test_complete_review_is_read_only_and_never_sends_reference_or_control():
    candidate = row(); original = copy.deepcopy(candidate); requests = []
    answers = iter([json.dumps(vote()), '{"correct":true,"grounded":true,"issue":"none"}'])
    def complete(messages):
        requests.append(messages); return next(answers)
    result = review_claims(candidate, complete)
    assert result['keep'] and candidate == original
    assert 'DO NOT SEND' not in json.dumps(requests) and 'manual_expected_keep' not in json.dumps(requests)
    assert 'Padding.' not in json.dumps(requests) and 'Distractor.' not in json.dumps(requests)


def test_unsupported_sentence_overrides_other_supported_sentence():
    candidate = row(); candidate['messages'][-1]['content'] += ' Everyone benefits.'
    answers = iter([json.dumps(vote()), json.dumps(vote(supported=False, evidence_quotes=[]))])
    result = review_claims(candidate, lambda _: next(answers))
    assert not result['keep'] and result['answer_fit'] is None
    assert len(result['sentences']) == 2


def test_sentence_split_covers_multiline_answer():
    candidate = row(); candidate['messages'][-1]['content'] = 'FINAL: First claim.\nSecond claim! Third claim?'
    assert answer_sentences(candidate)[1] == ['First claim.', 'Second claim!', 'Third claim?']


def test_fenced_json_and_bad_types():
    assert parse_object('```json\n{"supported":false}\n```') == {'supported': False}
    with pytest.raises(ValueError): parse_object('[]')


def test_source_prefix_collision_not_accepted():
    candidate = row(); candidate['messages'][1]['content'] = 'SOURCE primary2:part0\nWrong source.'
    with pytest.raises(ValueError): primary_evidence(candidate)


@pytest.mark.parametrize('source,quote', [
    ("I ca n't describe it , but it 's personal .", "I can't describe it, but it's personal."),
    ("We do n't agree ; you 're mistaken !", "We don't agree; you're mistaken!"),
    ('The rate is 12.5 % .', 'The rate is 12.5%.')])
def test_source_tokenization_spacing_is_reversible(source, quote):
    assert quote_normalize(source) == quote_normalize(quote)


@pytest.mark.parametrize('source,quote', [('not able', 'notable'), ('-2', '2'),
    ('12.5', '125'), ('no evidence', 'evidence'), ('must not', 'must'), ('red, blue', 'red blue')])
def test_normalization_preserves_semantically_meaningful_characters(source, quote):
    assert quote_normalize(source) != quote_normalize(quote)


def test_inconsistent_fit_reason_fails_closed():
    answers = iter([json.dumps(vote()), '{"correct":true,"grounded":true,"issue":"unsupported_claim"}'])
    with pytest.raises(ValueError): review_claims(row(), lambda _: next(answers))


def test_date_typo_in_judge_quote_is_not_auto_repaired():
    result = validate_sentence_vote('Pickups were filmed in 1998.', {'seg_1': 'Pickups were filmed in 1998.'},
        vote(evidence_quotes=[{'segment_id': 'seg_1', 'quote': 'Pickups were filmed in 1988.'}]))
    assert result['keep'] is False
