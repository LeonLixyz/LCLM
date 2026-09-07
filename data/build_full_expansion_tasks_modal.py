"""Parallel source build with explicit QA/split gates before teacher rollout."""
import json
import os
from pathlib import Path
import modal
from data.stage3_full_modal import image,volume

app=modal.App('lclm-full-expansion-task-build-v3')
ROOT=Path('/data/stage3-agent/real-expansion/pilots/full-20260906-v3')
SOURCES=['maud','finqa','pubmedqa_labeled','clapnq','contract_nli','tatqa','convfinqa',
    'multihiertt','multidoc2dial','faithdial','watsonx_docs_qa','acord','billsum','lex_glue','synthetic']

@app.function(image=image,cpu=8,memory=32768,timeout=86400,max_containers=8,volumes={'/data':volume})
def build_source(key):
    import random
    from collections import Counter
    from data.full_expansion_tasks import candidates,build_task
    from data.real_expansion_sources import SOURCE_SPECS
    from data.synthetic_expansion_agent import generate_task
    ROOT.mkdir(parents=True,exist_ok=True)
    report=ROOT/(key+'.build.json')
    if report.exists():return json.loads(report.read_text())
    output=ROOT/(key+'.tasks.jsonl')
    counts=Counter();pool={};rng=random.Random(20260906)
    source_ids={s.key:s.source_id for s in SOURCE_SPECS}
    source_ids.update(contract_nli='stanfordnlp/contract-nli',pubmedqa_labeled='qiaojin/PubMedQA:pqa_labeled')
    if key!='synthetic':
        for i,(_,_,_,documents) in enumerate(candidates(key)):
            for d in documents:
                if d['document_id'] in pool:continue
                if len(pool)<512:pool[d['document_id']]=d
                elif rng.random()<512/(i+1):
                    pool.pop(next(iter(pool)));pool[d['document_id']]=d
    with output.with_suffix('.tmp').open('w') as stream:
        seen=set()
        rows=range(10000) if key=='synthetic' else candidates(key)
        for row in rows:
            counts['candidates']+=1
            try:
                if key=='synthetic':task=generate_task(row,seed=20260906,distractors=6)
                else:
                    identifier,question,answer,documents=row
                    if identifier in seen:counts['duplicate']+=1;continue
                    seen.add(identifier)
                    task=build_task(key,source_ids[key],identifier,question,str(answer),documents,list(pool.values()))
                stream.write(json.dumps(task,ensure_ascii=False)+'\n')
                counts['tasks']+=1
                counts['multi_segment_support']+=len(task['support_segment_ids'])>1
            except ValueError as exc:counts[str(exc)]+=1
            if counts['candidates']%10000==0:print(key,dict(counts),flush=True)
    os.replace(output.with_suffix('.tmp'),output)
    result={'source':key,'counts':dict(counts),'builder':'v3'}
    report.write_text(json.dumps(result,indent=2));volume.commit()
    return result

@app.function(image=image,volumes={'/data':volume})
def finalize(reports):
    exclusions={'cuad':'The materialized file contains all 510 contracts, not a verified official training split. Quarantined pending split resolution.',
        'fa_v2':'Exact source identifier still needed.',
        'financebench':'Non-commercial license requires mixture-license decision.'}
    result={'sources':reports,'exclusions':exclusions,'builder':'v3'}
    (ROOT/'manifest.json').write_text(json.dumps(result,indent=2));volume.commit()
    return result

@app.local_entrypoint()
def main():
    reports=[]
    for report in build_source.map(SOURCES):
        reports.append(report);print(json.dumps(report),flush=True)
    print(json.dumps(finalize.remote(reports),indent=2))
