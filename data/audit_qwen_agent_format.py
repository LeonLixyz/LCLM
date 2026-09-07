"""Integration audit against Qwen's real tokenizer (no model/GPU required)."""
import json
import re
from pathlib import Path
from data.chat_utils import tokenize_qwen_agent_conversation
from data.clean_agent_trajectories import clean_nemotron


def audit_row(row,tokenizer):
    rendered=tokenizer.apply_chat_template(row['messages'],tools=row['tools'],
        tokenize=False,add_generation_prompt=False,enable_thinking=False)
    packed=tokenize_qwen_agent_conversation(row['messages'],tokenizer=tokenizer,
        tools=row['tools'],chat_template_kwargs={'enable_thinking':False})
    reference=tokenizer(rendered,add_special_tokens=False,return_offsets_mapping=True)
    assert reference['input_ids']==packed['input_ids']
    turns=list(re.finditer(r'<\|im_start\|>(system|user|assistant)\n(.*?)<\|im_end\|>\n',rendered,re.S))
    spans=[(m.start(2),m.end()) for m in turns if m[1]=='assistant']
    expected=[any(start<=a and b<=end and a<b for start,end in spans)
              for a,b in reference['offset_mapping']]
    actual=[v!=-100 for v in packed['labels']]
    assert actual==expected,'assistant-only label mask differs from independent ChatML spans'
    assistants=sum(m['role']=='assistant' for m in row['messages'])
    assert len(spans)==assistants
    calls=sum(len(m.get('tool_calls') or []) for m in row['messages'])
    assert sum(rendered[a:b].count('<tool_call>\n') for a,b in spans)==calls
    assert '<|memory_start|>' not in rendered and '<|memory_end|>' not in rendered
    assert '<think>' not in rendered and '</think>' not in rendered
    from data import preprocess_for_dynamic_packing as preprocessing
    preprocessing._worker_tokenizer=tokenizer
    preprocessing._worker_embed_tokenizer=None
    preprocessing._worker_memory_start_id=tokenizer.convert_tokens_to_ids('<|memory_start|>')
    preprocessing._worker_memory_end_id=tokenizer.convert_tokens_to_ids('<|memory_end|>')
    preprocessing._worker_memory_id=tokenizer.convert_tokens_to_ids('<|memory|>')
    transport={**row,'messages':json.dumps(row['messages']),'tools':json.dumps(row['tools'])}
    prepared=preprocessing.worker_process_example((transport,16,'compression_prompt',None,None))
    assert prepared is not None,'packing worker dropped a valid native row'
    assert prepared['base_input_ids']==packed['input_ids']
    assert prepared['base_labels']==packed['labels']
    assert prepared['memory_strings']==[] and prepared['memory_positions']==[]
    if calls:
        assert '<tools>' in rendered
        assert '<tool_response>' in rendered or row['messages'][-1].get('tool_calls')
    return {'tokens':len(actual),'labeled_tokens':sum(actual),'assistant_turns':assistants,'tool_calls':calls}


def audit_sources(root,output,limit=50):
    from collections import Counter
    from transformers import AutoTokenizer
    from huggingface_hub import HfApi
    model='Qwen/Qwen3-4B-Instruct-2507'
    revision=HfApi().model_info(model).sha
    tokenizer=AutoTokenizer.from_pretrained(model,revision=revision)
    tokenizer.add_special_tokens({'additional_special_tokens':['<|memory_start|>','<|memory_end|>','<|memory|>']})
    report={'tokenizer':model,'revision':revision,'sources':{}}
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    for source in sorted(Path(root).glob('nvidia--*')):
        for path in sorted(source.glob('data/*.jsonl')):
            counts=Counter();rejections=Counter();examples=[]
            with path.open() as stream:
                for i,line in enumerate(stream):
                    if i>=limit:break
                    counts['input']+=1
                    try:row=clean_nemotron(json.loads(line),source.name.replace('--','/'),path.stem)
                    except (ValueError,KeyError,TypeError) as exc:
                        rejections[str(exc)]+=1;continue
                    metrics=audit_row(row,tokenizer)
                    counts['passed']+=1
                    counts.update(metrics)
                    if len(examples)<2:
                        examples.append({'raw_training_json':row,'applied_chat_template':tokenizer.apply_chat_template(
                            row['messages'],tools=row['tools'],tokenize=False,add_generation_prompt=False,enable_thinking=False)})
            key=source.name+'/'+path.stem
            report['sources'][key]={'counts':dict(counts),'rejections':dict(rejections)}
            (output/(source.name+'-'+path.stem+'.examples.json')).write_text(json.dumps(examples,ensure_ascii=False,indent=2))
    (output/'report.json').write_text(json.dumps(report,indent=2))
    return report
