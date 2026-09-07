import json
import pytest
from data.clean_agent_trajectories import clean_native,clean_openthoughts,clean_nemotron

def test_cot_removed_without_editing_command_strings():
    commands=[{'keystrokes':'echo "<think>literal</think>"\n','duration':1}]
    row={'conversations':[{'role':'user','content':'protocol\nTask Description:\nfix code'},
        {'role':'assistant','content':'<think>private</think>'+json.dumps({'analysis':'private','plan':'private','commands':commands})},
        {'role':'user','content':'terminal output'},
        {'role':'assistant','content':json.dumps({'analysis':'private','commands':[],'task_complete':True})}]}
    out=clean_openthoughts(row)
    assert out['messages'][2]['tool_calls'][0]['function']['arguments']['commands']==commands
    assert out['messages'][3]['role']=='tool'
    assert 'private' not in json.dumps(out)
    assert len([m for m in out['messages'] if m['role']=='assistant'])==2

def test_native_tool_arguments_preserved():
    call={'id':'x','type':'function','function':{'name':'run','arguments':'{"analysis":"code field"}'}}
    out=clean_native({'messages':[{'role':'assistant','content':'<think>hidden</think>','reasoning_content':'hidden','tool_calls':[call]}],
        'tools':[{'type':'function','function':{'name':'run','parameters':{}}}]},'test')
    assert out['messages'][0]['tool_calls'][0]['function']['arguments']=={'analysis':'code field'}
    assert call['function']['arguments']=='{"analysis":"code field"}'
    assert 'reasoning_content' not in out['messages'][0]

def test_unclosed_thinking_rejected():
    with pytest.raises(ValueError):clean_native({'messages':[{'role':'assistant','content':'<think>unfinished'}]},'test')


def native_fixture():
    return {'messages':[
        {'role':'system','content':'task instructions'},
        {'role':'user','content':'lookup two things'},
        {'role':'assistant','reasoning_content':'private','tool_calls':[
            {'id':'a','type':'function','function':{'name':'lookup','arguments':'{"query":"a"}'}},
            {'id':'b','type':'function','function':{'name':'lookup','arguments':{'query':'b'}}}]},
        {'role':'tool','tool_call_id':'b','content':'second result'},
        {'role':'tool','tool_call_id':'a','content':'first result'},
        {'role':'assistant','content':'<analysis>private</analysis>answer'}],
        'tools':[{'type':'function','function':{'name':'lookup','parameters':'{"type":"object"}'}}]}


def test_transport_json_and_parallel_calls():
    row=native_fixture()
    row['messages']=[json.dumps(m) for m in row['messages']]
    result=clean_native(row,'test')
    assert isinstance(result['tools'][0]['function']['parameters'],dict)
    assert result['messages'][3]['name']=='lookup'
    assert result['messages'][-1]['content']=='answer'
    assert 'private' not in json.dumps(result)


def test_missing_tool_response_rejected():
    row=native_fixture();del row['messages'][3]
    with pytest.raises(ValueError,match='missing tool response'):clean_native(row,'test')


def test_structured_tool_result_and_identical_definitions_preserved():
    row=native_fixture()
    row['messages'][3]['content']={'value':[1,'literal <think>field</think>']}
    row['tools']=row['tools']*2
    result=clean_native(row,'test')
    assert json.loads(result['messages'][3]['content'])==row['messages'][3]['content']
    assert len(result['tools'])==1


def test_missing_response_id_only_inferred_when_unambiguous():
    row=native_fixture()
    del row['messages'][2]['tool_calls'][1]
    del row['messages'][3]
    del row['messages'][3]['tool_call_id']
    result=clean_native(row,'test')
    assert result['messages'][3]['tool_call_id']=='a'
    row=native_fixture();del row['messages'][3]['tool_call_id']
    with pytest.raises(ValueError,match='ambiguous tool response'):
        clean_native(row,'test')


def test_undefined_tool_rejected():
    row=native_fixture();row['tools']=[]
    with pytest.raises(ValueError,match='undefined tool'):clean_native(row,'test')


def test_search_generation_format_removed_but_answer_contract_preserved():
    row=native_fixture()
    row['messages'][0]['content']='Search task\n2. ** Planning ** - Plan visibly.\n6. ** Output Format ** - Final Answer: <Entity>\n7. ** Response Format ** - Use:\n - Thought: <reasoning>\n - Search: <call>'
    result=clean_nemotron(row,'nvidia/Nemotron-SFT-Agentic-v2','search')
    text=result['messages'][0]['content']
    assert 'Thought:' not in text and '** Planning **' not in text
    assert 'Final Answer: <Entity>' in text
