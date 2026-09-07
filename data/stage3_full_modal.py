"""Resumable acquisition and audits for the September Stage-3 build."""
import json
import os
from pathlib import Path
import modal

ROOT = Path('/data/stage3-build-20260906')
volume = modal.Volume.from_name('lclm-stage3-data')
image = modal.Image.debian_slim(python_version='3.11').pip_install(
    'datasets==3.6.0', 'huggingface_hub>=0.36,<2', 'pyarrow>=18,<22',
    'transformers==4.57.1', 'pytest', 'jinja2>=3.1',
).env({'PYTHONPATH':'/opt/lclm'}).add_local_dir('.', '/opt/lclm', ignore=['.git', '.venv', '__pycache__', '_modal_run'])
app = modal.App('lclm-stage3-full-20260906-' + os.environ.get('LCLM_STAGE3_JOB', 'agents'))

@app.function(image=image, cpu=8, memory=32768, timeout=86400,
              volumes={'/data':volume}, secrets=[modal.Secret.from_name('huggingface')])
def acquire_agents():
    from huggingface_hub import HfApi, snapshot_download
    api = HfApi()
    reports=[]
    for repo in ['open-thoughts/OpenThoughts-Agent-SFT-100K', 'nvidia/Nemotron-Agentic-v1',
                 'nvidia/Nemotron-SFT-Agentic-v2']:
        info=api.dataset_info(repo)
        destination=ROOT/'sources'/repo.replace('/','--')
        snapshot_download(repo, repo_type='dataset', revision=info.sha, local_dir=destination)
        reports.append({'repo':repo,'revision':info.sha,'path':str(destination)})
        volume.commit()
    (ROOT/'agent-source-manifest.json').write_text(json.dumps(reports,indent=2))
    volume.commit()
    return reports

@app.function(image=image, cpu=4, memory=16384, timeout=3600, volumes={'/data':volume})
def inventory():
    from datasets import load_from_disk
    import pyarrow.parquet as pq
    results=[]
    source_root=Path('/data/stage3-agent/real-expansion/sources')
    for source in sorted(source_root.iterdir()):
        item={'source':source.name,'tables':[]}
        for marker in source.rglob('dataset_info.json'):
            try:
                ds=load_from_disk(str(marker.parent))
                row=ds[0] if len(ds) else {}
                item['tables'].append({'path':str(marker.parent),'rows':len(ds),
                    'schema':str(ds.features),'sample':{k:str(v)[:700] for k,v in row.items()}})
            except Exception as e:
                item['tables'].append({'path':str(marker.parent),'error':str(e)})
        for name in ['snapshot-manifest.json','materialization-report.json']:
            p=source/name
            if p.exists():item[name]=json.loads(p.read_text())
        results.append(item)
    ROOT.mkdir(parents=True,exist_ok=True)
    (ROOT/'inventory.json').write_text(json.dumps(results,indent=2))
    volume.commit()
    return [{'source':r['source'],'tables':[{k:t[k] for k in ['path','rows'] if k in t} for t in r['tables']]} for r in results]

@app.local_entrypoint()
def main(action: str='inventory'):
    function={'inventory':inventory,'acquire-agents':acquire_agents,'agent-samples':agent_samples,'clean-agents':clean_agents,'build-tasks':build_tasks,'audit-qwen':audit_qwen}[action]
    print(json.dumps(function.remote(),indent=2))

@app.function(image=image, timeout=600, volumes={'/data':volume})
def agent_samples():
    import pyarrow.parquet as pq
    results=[]
    for source in sorted((ROOT/'sources').iterdir()):
        files=list(source.rglob('*.parquet'))+list(source.rglob('*.jsonl'))
        if not files:continue
        p=files[0]
        if p.suffix=='.parquet':row=next(pq.ParquetFile(p).iter_batches(batch_size=1)).to_pylist()[0]
        else:
            with p.open() as f:row=json.loads(next(f))
        results.append({'source':source.name,'files':[str(x.relative_to(source)) for x in files], 'sample':row})
    (ROOT/'agent-samples.json').write_text(json.dumps(results,indent=2))
    volume.commit()
    return [{**r,'sample':str(r['sample'])[:14000]} for r in results]

@app.function(image=image,cpu=8,memory=16384,timeout=3600,volumes={'/data':volume},
              secrets=[modal.Secret.from_name('huggingface')])
def audit_qwen():
    import subprocess
    from data.audit_qwen_agent_format import audit_sources
    subprocess.run(['python','-m','pytest','/opt/lclm/tests/test_clean_agent_trajectories.py','-q'],check=True)
    result=audit_sources(ROOT/'sources',ROOT/'qwen-format-audit')
    volume.commit()
    return result

@app.function(image=image,cpu=8,memory=32768,timeout=86400,volumes={'/data':volume})
def clean_agents():
    import hashlib
    from collections import Counter
    import pyarrow.parquet as pq
    from data.clean_agent_trajectories import clean_nemotron,clean_openthoughts
    from data.build_stage3_agent_mixture import agent_row_is_successful,agent_row_is_primary_trace
    output=ROOT/'agents-qwen-v2'
    output.mkdir(parents=True,exist_ok=True)
    report={}
    for source in sorted((ROOT/'sources').iterdir()):
        counts=Counter()
        seen=set()
        destination=output/(source.name+'.jsonl')
        temp=destination.with_suffix('.tmp')
        with temp.open('w') as target,(output/(source.name+'.rejected.jsonl')).open('w') as rejected:
            files=sorted(source.rglob('*.parquet'))+sorted(source.rglob('*.jsonl'))
            for path in files:
                if path.suffix=='.parquet':
                    stream=(r for b in pq.ParquetFile(path).iter_batches(batch_size=64) for r in b.to_pylist())
                else:
                    stream=(json.loads(line) for line in path.open() if line.strip())
                for index,row in enumerate(stream):
                    counts['input']+=1
                    try:
                        if source.name.startswith('open-thoughts'):
                            if not agent_row_is_successful(row):raise ValueError('source_failed_rollout')
                            if not agent_row_is_primary_trace(row):raise ValueError('derived_duplicate')
                            result=clean_openthoughts(row)
                        else:result=clean_nemotron(row,source.name.replace('--','/'),path.stem)
                        result['source_row_id']=str(path.relative_to(source))+':'+str(index)
                        fingerprint=hashlib.sha256(json.dumps([result['messages'],result['tools']],sort_keys=True).encode()).hexdigest()
                        if fingerprint in seen:raise ValueError('duplicate_trajectory')
                        seen.add(fingerprint)
                        target.write(json.dumps(result,ensure_ascii=False)+'\n')
                        counts['accepted']+=1
                        counts['assistant_turns']+=sum(m['role']=='assistant' for m in result['messages'])
                        counts['tool_calls']+=sum(len(m.get('tool_calls') or []) for m in result['messages'])
                        counts['accepted_subset:'+path.stem]+=1
                    except (ValueError,TypeError,KeyError) as exc:
                        counts['rejected:'+str(exc)[:100]]+=1
                        rejected.write(json.dumps({'file':str(path),'index':index,'reason':str(exc)})+'\n')
                    if counts['input']%10000==0:
                        print(source.name,dict(counts),flush=True)
        os.replace(temp,destination)
        report[source.name]=dict(counts)
        (output/'report.json').write_text(json.dumps(report,indent=2))
        volume.commit()
    return report

@app.function(image=image,cpu=16,memory=65536,timeout=86400,volumes={'/data':volume})
def build_tasks():
    import random
    from collections import Counter
    from data.full_expansion_tasks import candidates,build_task
    from data.real_expansion_sources import SOURCE_SPECS
    from data.synthetic_expansion_agent import generate_task
    destination=Path('/data/stage3-agent/real-expansion/pilots/full-20260906-v2')
    destination.mkdir(parents=True,exist_ok=True)
    counts={}
    selected=['maud','finqa','pubmedqa_labeled','clapnq','contract_nli','tatqa','convfinqa',
        'multihiertt','multidoc2dial','faithdial','watsonx_docs_qa','cuad','acord','billsum','lex_glue']
    source_ids={s.key:s.source_id for s in SOURCE_SPECS}
    source_ids['contract_nli']='stanfordnlp/contract-nli'
    source_ids['pubmedqa_labeled']='qiaojin/PubMedQA:pqa_labeled'
    for key in selected:
        output=destination/(key+'.tasks.jsonl')
        report_path=destination/(key+'.build.json')
        if report_path.exists():
            counts[key]=json.loads(report_path.read_text());continue
        stats=Counter();pool={};rng=random.Random(20260906)
        for i,(_,_,_,docs) in enumerate(candidates(key)):
            for d in docs:
                if len(pool)<256:pool[d['document_id']]=d
                elif rng.random()<256/(i+1):
                    pool.pop(next(iter(pool)));pool[d['document_id']]=d
        with output.with_suffix('.tmp').open('w') as f:
            seen=set()
            for identifier,question,answer,docs in candidates(key):
                stats['candidates']+=1
                if identifier in seen:stats['duplicate']+=1;continue
                seen.add(identifier)
                try:
                    task=build_task(key,source_ids[key],identifier,question,answer,docs,list(pool.values()))
                    f.write(json.dumps(task,ensure_ascii=False)+'\n')
                    stats['tasks']+=1
                    stats['multi_segment_support']+=len(task['support_segment_ids'])>1
                except ValueError as exc:stats[str(exc)]+=1
        os.replace(output.with_suffix('.tmp'),output)
        counts[key]=dict(stats)
        report_path.write_text(json.dumps(dict(stats),indent=2))
        volume.commit()
        print(key,dict(stats),flush=True)
    synthetic=destination/'synthetic.tasks.jsonl'
    if not synthetic.exists():
        with synthetic.with_suffix('.tmp').open('w') as f:
            for i in range(10000):
                f.write(json.dumps(generate_task(i,seed=20260906,distractors=6),ensure_ascii=False)+'\n')
        os.replace(synthetic.with_suffix('.tmp'),synthetic)
    counts['synthetic']={'tasks':10000,'families':5}
    (destination/'manifest.json').write_text(json.dumps(counts,indent=2))
    volume.commit()
    return counts
