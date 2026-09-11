"""Finite CPU corrected-Lex generation→audited private readiness; no auto launch."""
from __future__ import annotations
import json
from pathlib import Path
import modal
from data import lexglue_completion as completed
from data import expansion_completion_chain as chain
from data.sglang_backlog import atomic_json

APP_NAME='lclm-corrected-lexglue-completion-20260909-v2'
app=modal.App(APP_NAME)
inputs=modal.Volume.from_name('lclm-stage3-data',create_if_missing=False)
outputs=modal.Volume.from_name('lclm-stage3-agent-outputs-v2-20260909',create_if_missing=False,version=2)
cache=modal.Volume.from_name('lclm-hf-cache')
image=(modal.Image.debian_slim(python_version='3.11')
    .pip_install('torch==2.8.0',index_url='https://download.pytorch.org/whl/cpu')
    .pip_install('transformers==4.57.1','pyarrow==21.0.0','datasets==3.6.0','pytest==8.4.2')
    .env({'PYTHONPATH':'/opt/lclm','TOKENIZERS_PARALLELISM':'false','OMP_NUM_THREADS':'1','HF_HOME':'/cache/huggingface'})
    .add_local_dir('.','/opt/lclm',copy=True,ignore=['.git','__pycache__','**/__pycache__/**','**/*.pyc','_modal_run']))
options=dict(image=image,volumes={'/data':inputs,'/runs':outputs,'/cache':cache},secrets=[modal.Secret.from_name('huggingface')])


def reload():inputs.reload();outputs.reload()
def identity():return {'function_call_id':modal.current_function_call_id(),'input_id':modal.current_input_id()}
def root_for(policy_sha):
    chain.core.digest(policy_sha)
    return completed.GENERATION_ROOT/'private-completion'/policy_sha


@app.function(**options,cpu=4,memory=16384,timeout=1800)
def tests():
    import subprocess
    result=subprocess.run(['python','-m','pytest','-q','tests/test_lexglue_completion.py',
        'tests/test_export_reviewed_sglang27.py','tests/test_expansion_training_quality.py'],
        cwd='/opt/lclm',capture_output=True,text=True)
    print(result.stdout,result.stderr,flush=True)
    if result.returncode:raise RuntimeError('Corrected completion CPU tests failed')
    return {'status':'passed','output':result.stdout,'configuration':completed.configuration()}


@app.function(**options,cpu=2,memory=8192,timeout=1800)
def check_inputs(policy_path:str,policy_sha:str,review_spec:dict,manifest_spec:dict):
    reload()
    _,_,manifest,_,dependency=completed.governing(policy_path,policy_sha,review_spec,manifest_spec)
    return {'status':'valid_reviewed_corrected_completion','task_version':manifest['task_version'],
        'readiness_path':dependency['readiness_path'],'configuration':completed.configuration(),
        'approved_for_release':False}


@app.function(**options,cpu=4,memory=32768,timeout=21600,max_containers=1,retries=0)
def selection_and_export(policy_path:str,policy_sha:str,review_spec:dict,manifest_spec:dict,terminal_spec:dict,base:dict):
    import uuid
    reload();policy,*_=completed.governing(policy_path,policy_sha,review_spec,manifest_spec)
    chain.core.require(chain.base_snapshot(policy)==base,'Both exact finished base manifests required')
    root=root_for(policy_sha)/'selection-export';root.mkdir(parents=True,exist_ok=True)
    binding={'policy_sha256':policy_sha,'review':review_spec,'manifest':manifest_spec,'terminal':terminal_spec,
             'base':base,'configuration':completed.configuration()}
    final=root/'report.json'
    if final.exists():
        result=json.loads(final.read_text());chain.core.require(result['binding']==binding,'Completed selection/export inputs changed')
        chain.core.frozen_bytes(result['selection']);chain.core.frozen_bytes(result['transport']);return result
    owner=root/'owner.json';current=identity()
    if owner.exists():
        chain.core.require(json.loads(owner.read_text())=={'identity':current,'binding':binding},'Unrelated incomplete export owner')
    else:atomic_json(owner,{'identity':current,'binding':binding});outputs.commit()
    selected=root/'selection-stage.json'
    if selected.exists():selection=json.loads(selected.read_text());chain.core.frozen_bytes(selection['selection'])
    else:
        prior=[]
        for path in (root/'selection-attempts').glob('*/selection.json'):
            try:value=json.loads(path.read_text())
            except (ValueError,UnicodeDecodeError):continue
            if value.get('status')=='reviewed_complete':prior.append(path)
        chain.core.require(len(prior)<=1,'Multiple complete corrected selection attempts')
        if prior:selection={'status':'complete','selection':{'path':str(prior[0]),'sha256':chain.file_sha(prior[0])}}
        else:selection=completed.freeze_selection(policy_path,policy_sha,review_spec,manifest_spec,terminal_spec,base,
                                                   root/'selection-attempts'/uuid.uuid4().hex)
        atomic_json(selected,selection);outputs.commit()
    chosen=chain.document(selection['selection'])
    transports=[]
    for path in (root/'export-attempts').glob('*/manifest.json'):
        try:value=json.loads(path.read_text())
        except (ValueError,UnicodeDecodeError):continue
        if value.get('status')=='reviewed_complete':transports.append(path)
    chain.core.require(len(transports)<=1,'Multiple complete corrected transports')
    if transports:
        path=transports[0];transport=chain.document({'path':str(path),'sha256':chain.file_sha(path)})
        chain.core.require(transport['selection_sha256']==selection['selection']['sha256'],'Recovered transport selection changed')
        for spec in transport['files']:
            chain.core.require(chain.file_sha(Path(transport['transport_root'])/spec['name'])==spec['sha256'],
                               'Recovered completed transport shard changed')
    else:
        destination=root/'export-attempts'/uuid.uuid4().hex
        transport=chain.core.export_reviewed(selection['selection']['path'],selection['selection']['sha256'],destination,
            reviewed_sources=['lex_glue'],excluded_task_ids=chosen['excluded_task_ids'])
        path=destination/'manifest.json'
    result={'status':'complete','binding':binding,'selection':selection['selection'],
            'transport':{'path':str(path),'sha256':chain.file_sha(path)},'rows':transport['rows']}
    atomic_json(final,result);outputs.commit();return result


@app.function(**options,cpu=8,memory=32768,timeout=7200,max_containers=4,retries=0)
def audit_shard(policy_path:str,policy_sha:str,review_spec:dict,manifest_spec:dict,selection_spec:dict,transport_spec:dict,index:int):
    import multiprocessing as mp
    import pyarrow.parquet as pq
    from collections import Counter
    from data.expansion_trace_audit import init_worker,audit_worker
    from data.grouped_stage3_repair_modal import bounded_imap
    reload();completed.governing(policy_path,policy_sha,review_spec,manifest_spec)
    transport=chain.document(transport_spec);entry=transport['files'][index]
    chain.core.require(transport['selection_sha256']==selection_spec['sha256'],'Audit transport selection changed')
    path=Path(transport['transport_root'])/entry['name']
    chain.core.require(chain.file_sha(path)==entry['sha256'],'Audit shard bytes changed')
    binding={'selection':selection_spec,'transport':transport_spec,'index':index,'file':entry,
             'configuration':completed.configuration()}
    report_path=root_for(policy_sha)/'audits'/f'part-{index:06d}.json'
    if report_path.exists():
        report=json.loads(report_path.read_text());chain.core.require(report['binding']==binding,'Completed audit binding changed')
        return {'status':report['status'],'report':{'path':str(report_path),'sha256':chain.file_sha(report_path)},'rows':report['rows']}
    ids=[];counts=Counter();minimum=None;errors=[]
    def payloads():
        for batch in pq.ParquetFile(path).iter_batches(batch_size=32):
            for row in batch.to_pylist():
                chain.core.require(row['selection_sha256']==selection_spec['sha256'],'Audit row selection changed')
                ids.append(row['task_id']);yield completed.transport_payload(row)
    with mp.get_context('spawn').Pool(8,initializer=init_worker) as pool:
        for result in bounded_imap(pool,audit_worker,payloads(),8):
            if result.get('error'):errors.append(result)
            else:
                metrics=dict(result['metrics']);size=metrics.pop('minimum_segment_tokens')
                minimum=size if minimum is None else min(size,minimum);counts.update(metrics)
    chain.core.require(len(ids)==len(set(ids))==entry['rows'],'Audited shard task IDs/count changed')
    report={'status':'passed' if not errors else 'failed','binding':binding,'rows':len(ids),
        'task_ids_sha256':chain.core.sha(('\n'.join(sorted(ids))+'\n').encode()),
        'counts':dict(counts),'minimum_segment_tokens':minimum,'failed_rows':len(errors),'errors':errors,
        'approved_for_release':False,'scope':'Every selected canonical native call/evidence body, pinned decoder tokenization and independent assistant-only labels.'}
    atomic_json(report_path,report);outputs.commit()
    return {'status':report['status'],'report':{'path':str(report_path),'sha256':chain.file_sha(report_path)},'rows':len(ids)}


@app.function(**options,cpu=4,memory=16384,timeout=1800)
def finalize_readiness(policy_path:str,policy_sha:str,review_spec:dict,manifest_spec:dict,terminal_spec:dict,
                       selection_spec:dict,transport_spec:dict,audit_specs:list):
    from collections import Counter
    reload();policy,_,manifest,_,dependency=completed.governing(policy_path,policy_sha,review_spec,manifest_spec)
    completed.terminal_binding(terminal_spec,manifest_spec,review_spec,manifest)
    transport=chain.document(transport_spec);selection=chain.document(selection_spec)
    chain.core.require(transport['base_snapshot']==chain.base_snapshot(policy)==selection['base_snapshot'],
                       'Readiness bases differ from completed primary chain snapshot')
    chain.core.require(len(audit_specs)==len(transport['files']),'Missing corrected audit shards')
    counts=Counter();rows=0;seen=set()
    for spec in audit_specs:
        audit=chain.document(spec);index=audit['binding']['index']
        chain.core.require(index not in seen and 0<=index<len(transport['files']) and audit['status']=='passed'
            and audit['failed_rows']==0 and audit['binding']['transport']==transport_spec
            and audit['binding']['selection']==selection_spec and audit['binding']['file']==transport['files'][index]
            and audit['binding']['configuration']==completed.configuration(),'Incomplete/changed corrected audit')
        seen.add(index);rows+=audit['rows'];counts.update(audit['counts'])
    chain.core.require(rows==transport['rows'] and counts['accepted']==rows,'Corrected audit coverage differs')
    audit_spec=chain.write_document(root_for(policy_sha)/'format-audit.json',{'status':'passed','failed_rows':0,'rows':rows,
        'counts':dict(counts),'selection_sha256':selection_spec['sha256'],'transport_manifest_sha256':transport_spec['sha256'],
        'audit_parts':audit_specs,'configuration':completed.configuration(),'approved_for_release':False})
    ready={'schema':'reviewed-corrected-expansion-ready-v1','status':'complete','chain_policy_sha256':policy_sha,
        'task_version':manifest['task_version'],'ontology_artifact':manifest['ontology_artifact'],'source_allowlist':['lex_glue'],
        'training_quality_code_sha256':chain.file_sha(Path(chain.__file__).parent/'expansion_training_quality.py'),
        'manual_exclusions_sha256':policy['manual_exclusions']['sha256'],'source_review':review_spec,
        'generation_manifest':manifest_spec,'generation_terminal':terminal_spec,'selection':selection_spec,
        'transport':transport_spec,'format_audit':audit_spec,'completion_configuration':completed.configuration(),
        'approved_for_release':False}
    ready_spec=completed.publish_readiness(policy,policy_sha,transport['base_snapshot'],dependency,ready,
        root_for(policy_sha)/'ready-candidate.json');outputs.commit()
    return {'status':'complete','readiness':ready_spec,'format_audit':audit_spec,'rows':rows,'approved_for_release':False}


def graph_ids(call_id,suffix):
    def walk(nodes):
        for node in nodes:yield node;yield from walk(node.children)
    return {n.function_call_id for n in walk(modal.FunctionCall.from_id(call_id).get_call_graph()) if n.function_name.endswith(suffix)}


class Handoff(Exception):
    def __init__(self,result):self.result=result


@app.function(**options,cpu=2,memory=8192,timeout=86400,max_containers=1,retries=0)
def drive(policy_path:str,policy_sha:str,review_spec:dict,manifest_spec:dict,generation_call_id:str):
    import time
    reload();policy,_,manifest,_,_=completed.governing(policy_path,policy_sha,review_spec,manifest_spec)
    root=root_for(policy_sha);path=root/'state.json';current=identity()
    binding={'policy_sha256':policy_sha,'review':review_spec,'manifest':manifest_spec,'initial_generation_call_id':generation_call_id,
             'configuration':completed.configuration()}
    state=json.loads(path.read_text()) if path.exists() else {'binding':binding,'children':{},'handoffs':[],
        'generation_call_id':generation_call_id,'generation_handoffs':[],'status':'waiting_generation'}
    chain.core.require(state['binding']==binding,'Corrected completion chain changed')
    if state.get('completion'):
        chain.core.frozen_bytes(state['completion']['readiness']);return state['completion']
    def persist():atomic_json(path,state);outputs.commit()
    if state['handoffs'] and state['handoffs'][-1].get('status')!='successor_started':
        last=state['handoffs'][-1];child_id=last.get('successor_call_id')
        if not child_id:
            candidates=graph_ids(last['parent_identity']['function_call_id'],'drive')-set(last['known'])
            chain.core.require(len(candidates)==1,'Uncertain corrected completion handoff cannot be duplicated')
            child_id=candidates.pop();last['successor_call_id']=child_id
        if current==last['parent_identity']:persist();return {'status':'handed_off','successor_call_id':child_id}
        chain.core.require(current['function_call_id']==child_id,'Unrelated completion coordinator')
        last['status']='successor_started'
    state['identity']=current;persist();started=time.monotonic()
    args=[policy_path,policy_sha,review_spec,manifest_spec,generation_call_id]
    def maybe_handoff():
        if time.monotonic()-started<17*3600:return
        chain.core.require(len(state['handoffs'])<128,'Finite corrected completion handoff limit exceeded')
        last={'status':'dispatching','parent_identity':current,'known':sorted(graph_ids(current['function_call_id'],'drive'))}
        state['handoffs'].append(last);persist();child=drive.spawn(*args)
        last.update(status='spawned',successor_call_id=child.object_id);state['status']='handed_off';persist()
        raise Handoff({'status':'handed_off','successor_call_id':child.object_id})
    def spawn(key,function,arguments,suffix):
        entry=state['children'].get(key)
        if entry and entry.get('result'):return entry
        if entry and not entry.get('call_id'):
            candidates=graph_ids(entry['identity']['function_call_id'],suffix)-{e.get('call_id') for e in state['children'].values()}
            chain.core.require(len(candidates)==1,'Unknown corrected CPU dispatch needs reconciliation')
            entry['call_id']=candidates.pop();persist()
        if not entry:
            entry={'status':'dispatching','identity':current,'arguments':arguments};state['children'][key]=entry;persist()
            child=function.spawn(*arguments);entry.update(status='running',call_id=child.object_id);persist()
        chain.core.require(entry['arguments']==arguments,'Corrected stage inputs changed')
        return entry
    def await_call(entry):
        if entry.get('result'):return entry['result']
        while True:
            maybe_handoff()
            try:result=modal.FunctionCall.from_id(entry['call_id']).get(timeout=55)
            except TimeoutError:continue
            reload();entry.update(status='complete',result=result);persist();return result
    try:
        if 'terminal' not in state:
            while True:
                maybe_handoff()
                try:result=modal.FunctionCall.from_id(state['generation_call_id']).get(timeout=55)
                except TimeoutError:continue
                if result.get('status')=='handed_off':
                    new=result['successor_call_id'];chain.core.require(new not in state['generation_handoffs'] and new!=state['generation_call_id'],
                        'Corrected generation handoff loop')
                    state['generation_handoffs'].append(state['generation_call_id']);state['generation_call_id']=new;persist();continue
                reload();terminal={'path':result['completion_report_path'],'sha256':result['completion_report_sha256']}
                completed.terminal_binding(terminal,manifest_spec,review_spec,manifest);state['terminal']=terminal;persist();break
        if 'base' not in state:
            while True:
                maybe_handoff();reload();base=chain.base_snapshot(policy)
                if base:state['base']=base;persist();break
                time.sleep(45)
        first=await_call(spawn('selection_export',selection_and_export,
            [policy_path,policy_sha,review_spec,manifest_spec,state['terminal'],state['base']],'selection_and_export'))
        transport=chain.document(first['transport']);state['status']='auditing_every_selected_trace';persist()
        for index in range(len(transport['files'])):
            maybe_handoff();spawn(f'audit-{index:06d}',audit_shard,
                [policy_path,policy_sha,review_spec,manifest_spec,first['selection'],first['transport'],index],'audit_shard')
        audits=[]
        for index in range(len(transport['files'])):
            report=await_call(state['children'][f'audit-{index:06d}'])
            chain.core.require(report['status']=='passed','Corrected native/token/label audit failed')
            audits.append(report['report'])
        result=await_call(spawn('readiness',finalize_readiness,
            [policy_path,policy_sha,review_spec,manifest_spec,state['terminal'],first['selection'],first['transport'],audits],
            'finalize_readiness'))
        state.update(status='complete',completion=result);persist();return result
    except Handoff as exc:return exc.result
    except Exception as exc:
        state.update(status='needs_attention',error={'type':type(exc).__name__,'message':str(exc)});persist();raise
