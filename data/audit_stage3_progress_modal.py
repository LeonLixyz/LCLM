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
    full=source.parent/'full-20260906-v6'
    out['full_v6']={}
    from data.build_full_expansion_tasks_modal import SOURCES
    for key in SOURCES:
        for suffix in ('generation','progress'):
            path=full/(key+'.'+suffix+'.json')
            if path.exists():
                out['full_v6'][key]=json.loads(path.read_text())
                break
    completion=full/'full-generation-report.json'
    if completion.exists():
        out['full_v6_completion']=json.loads(completion.read_text())
    out['full_expansion_format']={p.stem:{k:v for k,v in json.loads(p.read_text()).items()
        if k in ('status','rows','counts','failed_rows','minimum_segment_tokens')}
        for p in sorted((ROOT/'full-expansion-format-audit').glob('*.json'))}
    (ROOT/'progress-audit.json').write_text(json.dumps(out,indent=2));volume.commit()
    return out

@app.local_entrypoint()
def main(tests:bool=False, samples:bool=False, version:str='v6', sources:str='finqa,convfinqa,clapnq,lex_glue', running_source:str=''):
    print(json.dumps(inspect_running_source.remote(running_source) if running_source else
        test.remote() if tests else inspect_pilot.remote(version,sources) if samples else audit.remote(),indent=2))


@app.function(image=image,cpu=2,memory=16384,timeout=1800,volumes={'/data':volume})
def inspect_running_source(source:str):
    """Committed prefix diagnostics only; never full-source approval or recovery."""
    import hashlib
    from collections import Counter
    from datetime import datetime, timezone
    from data.build_full_expansion_tasks_modal import SOURCES
    from data.grounding_claim_review import primary_evidence
    if source not in SOURCES: raise ValueError('Unknown running source')
    volume.reload()
    generated=Path('/data/stage3-agent/real-expansion/pilots/full-20260906-v6')
    progress=json.loads((generated/(source+'.progress.json')).read_text())
    seen=set(); counts=Counter(); details=Counter(); selected={}; files=[]
    for accepted in (True,False):
        path=generated/(source+('.accepted.jsonl' if accepted else '.rejected.jsonl'))
        digest=hashlib.sha256(); rows=0
        with path.open('rb') as stream:
            for line in stream:
                if not line.endswith(b'\n'): raise ValueError('Incomplete diagnostic snapshot line')
                digest.update(line); row=json.loads(line); rows+=1
                task_id=row['task_id']; verdict=row['verification']
                if task_id in seen or verdict['accepted'] is not accepted: raise ValueError('Invalid snapshot row')
                seen.add(task_id); reason=verdict['reason']; counts[reason]+=1
                semantic=verdict.get('semantic_review',{})
                if semantic:
                    claims=semantic.get('summary_claims')
                    if accepted: bucket='accepted'
                    elif claims is not None: bucket='summary_claim_rejected'
                    else: bucket='dual_judge_rejected'
                    if claims is not None:
                        for claim in claims['claims']:
                            details['claim_supported_'+str(claim['supported'])+'_quotes_'+str(claim['quotes_present'])]+=1
                    else:
                        for name in ('blind_vote','reference_vote'):
                            if name in semantic: details[name+':'+json.dumps(semantic[name],sort_keys=True)]+=1
                else: bucket=reason
                score=hashlib.sha256(('running-source-diagnostic-v1:'+task_id).encode()).hexdigest()
                if bucket not in selected or score<selected[bucket][0]: selected[bucket]=(score,row)
        files.append({'path':str(path),'rows':rows,'sha256':digest.hexdigest()})
    snapshot_sha=hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest()
    examples=[{'bucket':k,'selection_hash':v[0],'training_row':v[1]} for k,v in sorted(selected.items())]
    primary=[]
    for example in examples:
        row=example['training_row']
        view={'bucket':example['bucket'],'task_id':row['task_id'],'question':row.get('task'),
              'reference':row.get('gold_answer'),'verification':row['verification'],
              'assistant_messages':[m for m in row.get('messages',[]) if m['role']=='assistant']}
        try: view['primary_evidence']=primary_evidence(row)
        except Exception as exc: view['evidence_error']=type(exc).__name__+': '+str(exc)
        primary.append(view)
    report={'status':'partial_snapshot_diagnostic_only','source':source,'at':datetime.now(timezone.utc).isoformat(),
        'files':files,'snapshot_sha256':snapshot_sha,'rows_observed':len(seen),'reason_counts':dict(counts),
        'review_details':dict(details),'progress_checkpoint':progress,'samples':len(examples),
        'training_rows_modified':False,'approved_for_release':False,
        'limits':'Committed file snapshot, not completed-source accounting. Progress checkpoint can lag observed files. Every reason bucket sampled by minimum hash; no new inference.'}
    destination=ROOT/'running-source-diagnostics'/source/snapshot_sha
    destination.mkdir(parents=True,exist_ok=True)
    for name,value in [('report.json',report),('samples.json',examples),('primary-evidence.json',primary)]:
        path=destination/name
        if path.exists(): raise ValueError('This snapshot already has diagnostics; read existing artifacts')
        path.write_text(json.dumps(value,ensure_ascii=False,indent=2))
    volume.commit()
    return {**report,'output':str(destination)}

@app.function(image=image,timeout=600,volumes={'/data':volume})
def test():
    import subprocess
    result=subprocess.run(['python','-m','pytest','tests/test_expansion_task_normalization.py',
        'tests/test_clean_agent_trajectories.py','tests/test_chat_utils.py','tests/test_preprocess_for_dynamic_packing.py',
        'tests/test_stage3_release_checks.py','tests/test_expansion_judge_json.py',
        'tests/test_expansion_numeric_normalization.py','tests/test_real_expansion_agent.py',
        'tests/test_harvest_expansion_trace.py','tests/test_packed_file_discovery.py',
        'tests/test_expansion_semantic_review.py','tests/test_pubmedqa_split.py',
        'tests/test_techqa_adapter.py','tests/test_expansion_checkpoint_audit.py',
        'tests/test_expansion_trace_audit.py','tests/test_source_notices.py','-q'],
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
