"""Audit every accepted v5 pilot trace before a separately recorded review."""
import json
from pathlib import Path
import modal
from data.stage3_full_modal import image,volume

ROOT=Path('/data/stage3-agent/real-expansion/pilots/full-20260906-v6-audit')
app=modal.App('lclm-expansion-pilot-review-v6')

@app.function(image=image,cpu=8,memory=32768,timeout=1800,volumes={'/data':volume})
def audit():
    import re
    import hashlib
    from collections import Counter
    from transformers import AutoTokenizer
    from data.synthetic_expansion_agent import EXPAND_TOOL,TEACHER_SYSTEM_PROMPT,verify_trace
    from data.real_expansion_agent import verify_real_trace
    from data.expansion_task_normalization import TASK_SYSTEM_PROMPT
    from data.harvest_expansion_trace import harvest_training_messages
    from data import preprocess_for_dynamic_packing as prep
    from data.build_full_expansion_tasks_modal import SOURCES
    generation=json.loads((ROOT/'pilot-generation-report.json').read_text())
    assert generation['status']=='complete'
    tokenizer=AutoTokenizer.from_pretrained('Qwen/Qwen3-4B-Instruct-2507',
        revision='cdbee75f17c01a7cc42f958dc650907174af0554')
    tokenizer.add_special_tokens({'additional_special_tokens':['<|memory_start|>','<|memory_end|>','<|memory|>']})
    prep._worker_tokenizer=tokenizer;prep._worker_embed_tokenizer=None
    prep._worker_memory_start_id=tokenizer.convert_tokens_to_ids('<|memory_start|>')
    prep._worker_memory_id=tokenizer.convert_tokens_to_ids('<|memory|>')
    prep._worker_memory_end_id=tokenizer.convert_tokens_to_ids('<|memory_end|>')
    counts={};samples={};rejections={};seen=set();minimum=10**9
    for source in SOURCES:
        totals=Counter();examples=[];errors=[]
        source_report=json.loads((ROOT/(source+'.generation.json')).read_text())
        assert source_report['status']=='complete'
        assert sum(source_report['reasons'].values())==32
        for line in (ROOT/(source+'.accepted.jsonl')).read_text().splitlines():
            row=json.loads(line);messages=row['messages']
            assert row['task_id'] not in seen;seen.add(row['task_id'])
            assert row['verification']['accepted'] is True
            assert row['tools']==[EXPAND_TOOL]
            assert messages[0]=={'role':'system','content':TASK_SYSTEM_PROMPT}
            assert messages[1]['role']=='user'
            assert TEACHER_SYSTEM_PROMPT not in json.dumps(messages)
            assert harvest_training_messages(messages)[0]==messages
            matches=re.findall(r'(?m)^(seg_[1-9][0-9]*)\n<\|memory_start\|>(.*?)<\|memory_end\|>',messages[1]['content'],re.S)
            segments=dict(matches)
            assert len(matches)==len(segments)==row['segment_count']
            assert len(segments)>=2
            for body in segments.values():
                size=len(tokenizer.encode(body,add_special_tokens=False));minimum=min(minimum,size)
                assert size>=512,f'Short segment in {row["task_id"]}: {size}'
            pending={};expanded=[];calls=0
            for message in messages[2:]:
                if message['role']=='assistant':
                    for call in message.get('tool_calls',[]):
                        assert call['id'] not in pending
                        pending[call['id']]=call['function']['arguments']['segment_id'];calls+=1
                elif message['role']=='tool':
                    segment=pending.pop(message['tool_call_id'])
                    assert message['content']==segments[segment]
                    expanded.append(segment)
            assert not pending and calls==row['tool_call_count']
            assert set(row['support_segment_ids'])<=set(expanded)
            task={'segments':[{'segment_id':k,'text':v} for k,v in matches],
                'support_segment_ids':row['support_segment_ids'],'gold_answer':row['gold_answer'],
                'expected_final':'FINAL: '+str(row['gold_answer']),'source_dataset':row['source_dataset']}
            if source=='synthetic':assert verify_trace(task,messages).accepted
            elif row['verification'].get('answer_metric')=='qwen_question_evidence_dual_judge':
                semantic=row['verification']['semantic_review']
                assert semantic['keep'] is True
                assert all(semantic[k]['correct'] is True and semantic[k]['grounded'] is True
                           for k in ('blind_vote','reference_vote'))
                if source=='billsum':
                    assert semantic['summary_claims']['keep'] is True
                    assert all(v['supported'] and v['quotes_present'] for v in semantic['summary_claims']['claims'])
                if source=='pubmedqa_labeled':
                    from data.pubmedqa_split import training_ids
                    allowed=training_ids(json.loads(Path('/data/stage3-agent/real-expansion/sources/pubmedqa_labeled/official-splits/split-manifest.json').read_text()))
                    assert row['source_row_id'] in allowed
                    assert verify_real_trace(task,messages)['accepted']
                else:
                    assert verify_real_trace(task,messages)['reason'].startswith(('accepted:','wrong_answer:'))
            elif row['verification'].get('answer_metric')!='qwen_reference_and_evidence_judge':
                assert verify_real_trace(task,messages)['accepted']
            else:
                assert row['verification']['judge']=={'correct':True,'grounded':True}
                assert verify_real_trace(task,messages)['reason'].startswith(('accepted:','wrong_answer:'))
            compact=prep.worker_process_example((row,16,'compression_prompt',None,None))
            assert compact is not None
            compact_messages,memories=prep._compact_agent_memory_regions(messages)
            assert memories==list(segments.values())==compact['memory_strings']
            rendered=tokenizer.apply_chat_template(compact_messages,tools=row['tools'],tokenize=False,add_generation_prompt=False)
            encoded=tokenizer(rendered,add_special_tokens=False,return_offsets_mapping=True)
            spans=[(m.start(1),m.end()) for m in re.finditer(r'<\|im_start\|>assistant\n(.*?)<\|im_end\|>\n',rendered,re.S)]
            expected=[any(s<=a<b<=e for s,e in spans) for a,b in encoded['offset_mapping']]
            assert encoded['input_ids']==compact['base_input_ids']
            assert expected==[v!=-100 for v in compact['base_labels']]
            assert sum(rendered[a:b].count('<tool_call>\n') for a,b in spans)==calls
            totals.update(accepted=1,calls=calls,multi_expansion=int(len(set(expanded))>1),
                labeled_tokens=sum(expected),segments=len(segments))
            if len(examples)<2:
                examples.append({'task_id':row['task_id'],'task':row['task'],'gold':row['gold_answer'],
                    'answer':messages[-1]['content'],'expanded':expanded,
                    'evidence_excerpts':[m['content'][:500] for m in messages if m['role']=='tool'],
                    'verification':row['verification']})
        assert totals['accepted']==sum(v for k,v in source_report['reasons'].items() if k.startswith('accepted'))
        for line in (ROOT/(source+'.rejected.jsonl')).read_text().splitlines():
            r=json.loads(line)
            if r.get('error') and len(errors)<3:
                errors.append({'task_id':r['task_id'],'error':r['error'],'verification':r['verification']})
        counts[source]=dict(totals);samples[source]=examples;rejections[source]=errors
    report={'status':'passed','accepted_traces':len(seen),'minimum_segment_tokens':minimum,'sources':counts,
        'generation_manifest_sha256':hashlib.sha256(json.dumps(generation['manifest'],sort_keys=True).encode()).hexdigest(),
        'checks':['tool schema and exact expand bodies','explicit task system without teacher system',
            'final-only assistant harvesting','all segments >=512 real Qwen tokens',
            'source verification repeated','all assistant calls and only assistant spans labeled'],
        'limits':'Free-form answer correctness still relies on the source reference/evidence LLM judge; this audit does not prove semantics.'}
    (ROOT/'format-audit.json').write_text(json.dumps(report,indent=2))
    (ROOT/'review-samples.json').write_text(json.dumps(samples,indent=2))
    (ROOT/'review-errors.json').write_text(json.dumps(rejections,indent=2));volume.commit()
    return {'audit':report,'samples':samples,'errors':rejections}

@app.local_entrypoint()
def main():
    result = audit.remote()
    print(json.dumps(result['audit'],indent=2))
    print('Full samples/errors saved as review-samples.json and review-errors.json on the data volume.')
