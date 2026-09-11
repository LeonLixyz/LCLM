import copy
import pytest
from data.expansion_training_quality import inspect_expansion_efficiency


def message(*segments):
    return {'role': 'assistant', 'content': '', 'tool_calls': [
        {'id': str(i), 'type': 'function', 'function': {'name': 'expand', 'arguments': {'segment_id': segment}}}
        for i, segment in enumerate(segments)]}


def test_distinct_calls_including_multiple_calls_in_one_turn_are_retained():
    messages = [message('seg_1', 'seg_2'), {'role': 'tool', 'content': 'seg_1 seg_1'}, message('seg_3')]
    original = copy.deepcopy(messages)
    result = inspect_expansion_efficiency(messages)
    assert result['eligible'] and result['tool_calls'] == 3 and result['redundant_calls'] == 0
    assert messages == original


def test_repeats_across_turns_and_parallel_calls_are_counted_without_editing():
    messages = [message('seg_2', 'seg_2'), message('seg_1', 'seg_2')]
    messages[1]['tool_calls'][0]['function']['arguments'] = '{"segment_id": "seg_1"}'
    original = copy.deepcopy(messages)
    result = inspect_expansion_efficiency(messages)
    assert not result['eligible'] and result['repeated_segments'] == {'seg_2': 3}
    assert result['redundant_calls'] == 2 and messages == original


def test_plain_native_agents_or_malformed_calls_cannot_be_misclassified():
    with pytest.raises(ValueError, match='no expansion'):
        inspect_expansion_efficiency([{'role': 'assistant', 'content': 'FINAL: 2'}])
    other = message('seg_1')
    other['tool_calls'][0]['function']['name'] = 'terminal_step'
    with pytest.raises(ValueError, match='canonical expansion'):
        inspect_expansion_efficiency([other])
