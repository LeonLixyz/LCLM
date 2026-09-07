from copy import deepcopy
from types import SimpleNamespace
import pytest
from data.expansion_trace_audit import audit_trace
from data.synthetic_expansion_agent import EXPAND_TOOL
from data.expansion_task_normalization import TASK_SYSTEM_PROMPT


class TinyTokenizer:
    rendered = '<|im_start|>user\ncontext<|im_end|>\n<|im_start|>assistant\n<tool_call>\ncall<|im_end|>\n'
    def encode(self, text, **kwargs):return list(range(len(text)))
    def apply_chat_template(self, *args, **kwargs):return self.rendered
    def __call__(self, text, **kwargs):
        return {'input_ids':self.encode(text),'offset_mapping':[(i,i+1) for i in range(len(text))]}


def fixture():
    body='evidence '*100
    messages=[{'role':'system','content':TASK_SYSTEM_PROMPT},
        {'role':'user','content':f'seg_1\n<|memory_start|>{body}<|memory_end|>\nseg_2\n<|memory_start|>{body}<|memory_end|>'},
        {'role':'assistant','content':'','tool_calls':[{'id':'call1','type':'function',
            'function':{'name':'expand','arguments':{'segment_id':'seg_1'}}}]},
        {'role':'tool','tool_call_id':'call1','content':body},
        {'role':'assistant','content':'FINAL: 42'}]
    row={'task_id':'task1','messages':messages,'tools':deepcopy([EXPAND_TOOL]),
         'verification':{'accepted':True},'segment_count':2,'tool_call_count':1,
         'support_segment_ids':['seg_1'],'gold_answer':'42','source_dataset':'example/qa'}
    tokenizer=TinyTokenizer()
    start=tokenizer.rendered.index('<tool_call>')
    compact={'base_input_ids':tokenizer.encode(tokenizer.rendered),
             'base_labels':[-100]*start+list(range(start,len(tokenizer.rendered))),
             'memory_strings':[body,body]}
    prep=SimpleNamespace(worker_process_example=lambda _:compact,
        _compact_agent_memory_regions=lambda _:(messages,[body,body]))
    return row,tokenizer,prep,compact


def test_trace_audit_valid_native_trace():
    row,tokenizer,prep,_=fixture()
    metrics=audit_trace(row,'maud',tokenizer,prep)
    assert metrics['accepted']==1 and metrics['calls']==1


@pytest.mark.parametrize('failure',['system','tools','body','duplicate_result','extra_argument','answer','labels','preamble'])
def test_trace_audit_rejects_wrong_training_contract(failure):
    row,tokenizer,prep,compact=fixture()
    if failure=='system':row['messages'][0]['content']='teacher instructions'
    elif failure=='tools':row['tools']=[]
    elif failure=='body':row['messages'][3]['content']='wrong'
    elif failure=='duplicate_result':row['messages'].insert(4,deepcopy(row['messages'][3]))
    elif failure=='extra_argument':row['messages'][2]['tool_calls'][0]['function']['arguments']['extra']=1
    elif failure=='answer':row['messages'][-1]['content']='FINAL: 99'
    elif failure=='labels':compact['base_labels'][0]=0
    else:row['messages'][2]['content']='reasoning'
    with pytest.raises(ValueError):audit_trace(row,'maud',tokenizer,prep)
