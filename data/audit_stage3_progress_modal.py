"""Bounded diagnostic reads of persistent Stage-3 job artifacts."""
import json
from pathlib import Path
import modal
from data.stage3_full_modal import image,volume,ROOT

app=modal.App('lclm-stage3-progress-audit')

@app.function(image=image,cpu=2,memory=8192,timeout=600,volumes={'/data':volume})
def audit():
    out={}
    for name in ['agents-transport/report.json','publication-report.json']:
        p=ROOT/name
        if p.exists():out[name]={k:v for k,v in json.loads(p.read_text()).items()
            if k!='parquet_files'}
    for kind in ['base','agents','expansion']:
        paths=sorted((ROOT/f'packed-{kind}-cs16-32768').glob('part-*/report.json'))
        reports=[json.loads(p.read_text()) for p in paths]
        out[kind]={'completed_partitions':len(paths),'input_rows':sum(r['counts']['input_rows'] for r in reports)}
        out[kind]['counts']={key:sum(r['counts'].get(key,0) for r in reports)
            for key in set().union(*(r['counts'] for r in reports))}
    paths=sorted((ROOT/'packed-base-prefix-recovery').glob('part-*/report.json'))
    reports=[json.loads(p.read_text()) for p in paths]
    out['recovery']={'completed_partitions':len(paths),'counts':{
        key:sum(r['counts'].get(key,0) for r in reports)
        for key in set().union(*(r['counts'] for r in reports))}}
    if out['base']['completed_partitions']==64 and len(reports)==64:
        from data.stage3_release_checks import validate_base_recovery
        baseline=[json.loads(p.read_text()) for p in
            sorted((ROOT/'packed-base-cs16-32768').glob('part-*/report.json'))]
        out['base_combined_counts']=validate_base_recovery(baseline,reports)
    p=ROOT/'agents-qwen-v2/report.json'
    if p.exists():
        out['cleaning']={key:{k:v for k,v in counts.items() if not k.startswith('rejected:')}
                         for key,counts in json.loads(p.read_text()).items()}
    source=Path('/data/stage3-agent/real-expansion/pilots/full-20260906-v3')
    out['tasks']=[json.loads(p.read_text()) for p in sorted(source.glob('*.build.json'))]
    for version in ['v4','v5','v6']:
        pilot=source.parent/f'full-20260906-{version}-audit'
        out['pilot_'+version]={p.name:json.loads(p.read_text()) for p in sorted(pilot.glob('*.json'))
            if 'manifest' not in p.name and p.name not in {'review-samples.json','review-errors.json'}
            and not p.name.startswith('semantic-review-')}
    (ROOT/'progress-audit.json').write_text(json.dumps(out,indent=2));volume.commit()
    return out

@app.local_entrypoint()
def main(tests:bool=False, samples:bool=False, version:str='v6', sources:str='finqa,convfinqa,clapnq,lex_glue'):
    print(json.dumps(test.remote() if tests else inspect_pilot.remote(version,sources) if samples else audit.remote(),indent=2))

@app.function(image=image,timeout=600,volumes={'/data':volume})
def test():
    import subprocess
    result=subprocess.run(['python','-m','pytest','tests/test_expansion_task_normalization.py',
        'tests/test_clean_agent_trajectories.py','tests/test_chat_utils.py','tests/test_preprocess_for_dynamic_packing.py',
        'tests/test_stage3_release_checks.py','tests/test_expansion_judge_json.py',
        'tests/test_expansion_numeric_normalization.py','tests/test_real_expansion_agent.py',
        'tests/test_harvest_expansion_trace.py','tests/test_packed_file_discovery.py',
        'tests/test_expansion_semantic_review.py','tests/test_pubmedqa_split.py','-q'],
        cwd='/opt/lclm',capture_output=True,text=True)
    report={'exit_code':result.returncode,'output':result.stdout+result.stderr}
    if result.returncode==0:
        from transformers import AutoTokenizer
        from data.preprocess_for_dynamic_packing import process_sft_example
        tokenizer=AutoTokenizer.from_pretrained('Qwen/Qwen3-4B-Instruct-2507',
            revision='cdbee75f17c01a7cc42f958dc650907174af0554')
        tokenizer.add_special_tokens({'additional_special_tokens':['<|memory_start|>','<|memory_end|>','<|memory|>']})
        tested=[]
        for prefix in ['', '\n', '\n\n', '\t', ' ', '  ', '\r\n', '\u2003']:
            row={'compression_prompt':[{'role':'user','content':'Question'}], 'target':prefix+'Final answer.'}
            ids,labels,_,boundary=process_sft_example(row,tokenizer)
            prompt=tokenizer.apply_chat_template(row['compression_prompt'],tokenize=False,add_generation_prompt=True)
            full=tokenizer.apply_chat_template(row['compression_prompt']+[{'role':'assistant','content':row['target']}],tokenize=False,add_generation_prompt=False)
            offsets=tokenizer(full,add_special_tokens=False,return_offsets_mapping=True)['offset_mapping']
            assert [v!=-100 for v in labels]==[a>=len(prompt) and b>a for a,b in offsets]
            old_ids=tokenizer.apply_chat_template(row['compression_prompt'],tokenize=True,add_generation_prompt=True)
            mismatch=ids[:len(old_ids)]!=old_ids
            recovery=process_sft_example({**row,'_recover_prefix_only':True},tokenizer)
            strict=process_sft_example({**row,'_legacy_sft_prefix_strict':True},tokenizer)
            assert (recovery is not None)==mismatch
            assert (strict is None)==mismatch
            tested.append({'prefix':repr(prefix),'recovery_needed':mismatch})
        report['real_tokenizer_boundary_cases']=tested
    (ROOT/'validation/prefix-recovery-tests.json').write_text(json.dumps(report,indent=2))
    volume.commit()
    return report

@app.function(image=image,timeout=600,volumes={'/data':volume})
def inspect_pilot(version:str,sources:str):
    if version not in ['v4','v5','v6']:raise ValueError('Unknown pilot version')
    root=Path('/data/stage3-agent/real-expansion/pilots')/f'full-20260906-{version}-audit'
    out={}
    for path in sorted(root.glob('*.jsonl')):
        if path.name.split('.')[0] not in sources.split(','):continue
        examples=[]
        with path.open() as stream:
            for line in stream:
                r=json.loads(line)
                item={k:r.get(k) for k in ['task_id','task','gold_answer','verification','error']}
                item['assistants']=[m for m in r.get('messages',[]) if m['role']=='assistant']
                item['tool_excerpts']=[m.get('content','')[:350] for m in r.get('messages',[]) if m['role']=='tool']
                examples.append(item)
                if len(examples)>=2:break
        out[path.name]=examples
    return out
