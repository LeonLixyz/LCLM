import copy
import pytest
from data.billsum_quote_diagnostic import diagnose_quotes, evidence_runs


def fixture(parts=None, quote='Special Inspector General'):
    parts = parts or [('a:part0', 'Special '), ('a:part1', 'Inspector General')]
    task = {'family': 'billsum', 'task_id': 't', 'question': 'Use these source documents: a.\nSummarize.',
            'segments': [{'segment_id': f'seg_{i+1}', 'record_id': record, 'text': text}
                         for i, (record, text) in enumerate(parts)],
            'support_segment_ids': [f'seg_{i+1}' for i in range(len(parts))]}
    messages = []
    for item in reversed(task['segments']):  # Expansion order is not source order.
        call = item['segment_id']
        messages.extend([{'role': 'assistant', 'tool_calls': [{'id': call, 'function': {
            'name': 'expand', 'arguments': {'segment_id': call}}}]},
            {'role': 'tool', 'tool_call_id': call, 'content': 'SOURCE '+item['record_id']+'\n'+item['text']}])
    row = {'task_id': 't', 'task': task['question'], 'messages': messages,
           'verification': {'accepted': False, 'semantic_review': {'summary_claims': {'claims': [
               {'supported': True, 'quotes_present': False, 'evidence_quotes': [quote]}]}}}}
    return row, task


def test_cross_chunk_match_is_read_only_and_not_approval():
    row, task = fixture(); before = copy.deepcopy((row, task))
    result = diagnose_quotes(row, task)
    assert result['claims'][0]['contiguous_primary_quotes_present']
    assert result['claims'][0]['quote_matches'][0][0]['parts'] == [0, 1]
    assert not result['approved_for_release'] and not result['training_rows_modified']
    assert (row, task) == before


@pytest.mark.parametrize('parts', [
    [('a:part0', 'Special'), ('a:part2', 'Inspector General')],
    [('a:part0', 'Special\nRELATED SOURCE b:\nInspector General')],
    [('a:part0', 'Special\nRELATED SOURCE b:\nPadding'), ('a:part1', 'Inspector General')],
    [('a:part1', 'Special'), ('a:part0', 'Inspector General')]])
def test_no_join_across_gaps_padding_or_reversed_source(parts):
    row, task = fixture(parts)
    assert not diagnose_quotes(row, task)['claims'][0]['contiguous_primary_quotes_present']


def test_no_join_across_documents():
    row, task = fixture([('a:part0', 'Special'), ('b:part0', 'Inspector General')])
    row['task'] = task['question'] = 'Use these source documents: a, b.\nSummarize.'
    assert not diagnose_quotes(row, task)['claims'][0]['contiguous_primary_quotes_present']


def test_unexpanded_primary_is_not_evidence():
    row, task = fixture(); row['messages'] = row['messages'][2:]
    assert not diagnose_quotes(row, task)['claims'][0]['contiguous_primary_quotes_present']


def test_distractor_cannot_supply_missing_words():
    row, task = fixture([('a:part0', 'Special'), ('b:part0', 'Inspector General')])
    task['support_segment_ids'] = ['seg_1']
    assert not diagnose_quotes(row, task)['claims'][0]['contiguous_primary_quotes_present']


@pytest.mark.parametrize('quote', ['special Inspector General', 'Special Inspector-General',
                                  'Special Inspector general', 'Special General'])
def test_no_fuzzy_word_or_punctuation_matching(quote):
    row, task = fixture(quote=quote)
    assert not diagnose_quotes(row, task)['claims'][0]['contiguous_primary_quotes_present']


@pytest.mark.parametrize('change', ['task', 'body', 'call', 'orphan', 'duplicate', 'pending', 'part', 'support'])
def test_invalid_provenance_or_transport_fails(change):
    row, task = fixture()
    if change == 'task': row['task_id'] = 'wrong'
    if change == 'body': row['messages'][1]['content'] += 'injected'
    if change == 'call': row['messages'][0]['tool_calls'][0]['function']['name'] = 'search'
    if change == 'orphan': row['messages'][1]['tool_call_id'] = 'wrong'
    if change == 'duplicate': row['messages'] += copy.deepcopy(row['messages'][:2])
    if change == 'pending': row['messages'].pop()
    if change == 'part':
        task['segments'][1]['record_id'] = 'a:part0'
        row['messages'][1]['content'] = 'SOURCE a:part0\nInspector General'
    if change == 'support': task['support_segment_ids'] = ['seg_99']
    with pytest.raises(ValueError): evidence_runs(row, task)


def test_false_supported_vote_and_empty_quotes_stay_unapproved():
    row, task = fixture()
    claim = row['verification']['semantic_review']['summary_claims']['claims'][0]
    claim['supported'] = False
    assert diagnose_quotes(row, task)['claims'][0]['supported_vote'] is False
    claim['evidence_quotes'] = []
    result = diagnose_quotes(row, task)
    assert not result['claims'][0]['contiguous_primary_quotes_present']
    assert not result['approved_for_release']
