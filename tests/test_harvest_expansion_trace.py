import copy
import pytest
from data.harvest_expansion_trace import harvest_training_messages


def trace(final='Computing the sum.\nFINAL: 42'):
    return [{'role':'system','content':'task instruction'},
        {'role':'user','content':'seg_1 <|memory_start|>document<|memory_end|>'},
        {'role':'assistant','content':'I need the exact number.','tool_calls':[
            {'id':'call1','type':'function','function':{'name':'expand','arguments':{'segment_id':'seg_1'}}}]},
        {'role':'tool','tool_call_id':'call1','content':'<analysis> is a document string; result is 42'},
        {'role':'assistant','content':final,'reasoning_content':'teacher-only'}]


def test_only_assistant_prose_changes_and_original_trace_is_immutable():
    original=trace();before=copy.deepcopy(original)
    messages,metadata=harvest_training_messages(original)
    assert original==before
    assert messages[:2]==original[:2]
    assert messages[2]['tool_calls']==original[2]['tool_calls']
    assert messages[2]['content']==''
    assert messages[3]==original[3]
    assert messages[-1]=={'role':'assistant','content':'FINAL: 42'}
    assert metadata['removed_assistant_characters']>0


@pytest.mark.parametrize('answer',['42','FINAL: 1\nFINAL: 2','FINAL: 42\nmore prose',
    'FINAL: <think>secret</think>','FINAL: <|memory_start|>42<|memory_end|>'])
def test_ambiguous_or_unsafe_answer_boundaries_are_not_guessed(answer):
    with pytest.raises(ValueError):harvest_training_messages(trace(answer))


def test_multiple_calls_and_turns_are_not_dropped():
    messages=trace('FINAL: 42')
    second=copy.deepcopy(messages[2:4])
    second[0]['tool_calls'][0]['id']='call2';second[1]['tool_call_id']='call2'
    messages[-1:-1]=second
    result,_=harvest_training_messages(messages)
    assert len(result)==len(messages)
    assert [m.get('tool_calls') for m in result]==[m.get('tool_calls') for m in messages]
