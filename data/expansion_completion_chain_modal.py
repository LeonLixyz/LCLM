"""Finite CPU generation-to-packing chain; deployment/tests never dispatch it."""
from __future__ import annotations
import json
from pathlib import Path
import modal
from data import expansion_completion_chain as chain
from data.sglang_backlog import atomic_json

APP_NAME='lclm-expansion-completion-chain-20260909-v1'
app=modal.App(APP_NAME)
inputs=modal.Volume.from_name('lclm-stage3-data',create_if_missing=False)
outputs=modal.Volume.from_name('lclm-stage3-agent-outputs-v2-20260909',create_if_missing=False,version=2)
cache=modal.Volume.from_name('lclm-hf-cache')
ignore=['.git','__pycache__','**/__pycache__/**','**/*.pyc','_modal_run']
image=(modal.Image.debian_slim(python_version='3.11')
    .pip_install('torch==2.8.0',index_url='https://download.pytorch.org/whl/cpu')
    .pip_install('transformers==4.57.1','pyarrow==21.0.0','datasets==3.6.0','pytest==8.4.2')
    .env({'PYTHONPATH':'/opt/lclm','TOKENIZERS_PARALLELISM':'false','OMP_NUM_THREADS':'1','HF_HOME':'/cache/huggingface'})
    .add_local_dir('.','/opt/lclm',copy=True,ignore=ignore))
options=dict(image=image,volumes={'/data':inputs,'/runs':outputs,'/cache':cache},secrets=[modal.Secret.from_name('huggingface')])


def reload():
    inputs.reload();outputs.reload()


def commit():
    inputs.commit();outputs.commit()


def identity():
    return {'function_call_id':modal.current_function_call_id(),'input_id':modal.current_input_id()}


def root_for(policy_sha):
    chain.core.digest(policy_sha)
    return Path(chain.CHAIN_ROOT)/policy_sha


def verify_saved_stage(report,input_binding):
    chain.core.require(report['binding']==input_binding,'Completed CPU stage input changed')
    for key in ('selection','transport','append_summary'):
        if key in report:chain.core.frozen_bytes(report[key])
    return report


@app.function(**options,cpu=4,memory=16384,timeout=1800)
def tests():
    import subprocess
    result=subprocess.run(['python','-m','pytest','-q','tests/test_expansion_completion_chain.py',
        'tests/test_export_reviewed_sglang27.py','tests/test_reviewed_expansion_adapters.py',
        'tests/test_expansion_training_quality.py'],cwd='/opt/lclm',capture_output=True,text=True)
    print(result.stdout,result.stderr,flush=True)
    if result.returncode:raise RuntimeError('Finite packing chain CPU tests failed')
    return {'status':'passed','output':result.stdout,'configuration':chain.config(),
            'scope':'CPU fixtures only; no selection/approval/export/append of production data'}


@app.function(**options,cpu=4,memory=16384,timeout=1800)
def check_policy(policy_path:str,policy_sha:str):
    reload();policy,_=chain.load_policy(policy_path,policy_sha)
    return {'status':'valid_root_policy','reviewed_sources':policy['reviewed_sources'],
            'sglang27_sources':policy['sglang27_sources'],'configuration':chain.config(),
            'approved_for_release':False}


@app.function(**options,cpu=4,memory=32768,timeout=21600,max_containers=1,retries=0)
def stage_worker(policy_path:str,policy_sha:str,stage:str,arguments:dict):
    """A completed stage survives later retries; incomplete attempts remain separate."""
    import uuid,time
    reload();policy,_=chain.load_policy(policy_path,policy_sha)
    if stage in ('freeze_sglang','freeze_union'):
        chain.core.require(chain.base_snapshot(policy)==arguments['base'],
                           'CPU selection requires both actual completed base manifests')
    if stage in ('export_sglang','export_union'):
        selected=chain.document(arguments['selection'])
        expected=policy['sglang27_sources'] if stage=='export_sglang' else policy['reviewed_sources']
        chain.core.require(set(selected['reviewed_sources'])==set(expected) and
            any(spec.get('sha256')==policy_sha for spec in selected['approval']['artifacts']),
            'Export selection does not match this root source policy')
    stage_root=root_for(policy_sha)/'stages'/stage
    binding={'policy_sha256':policy_sha,'stage':stage,'arguments':arguments,'configuration':chain.config()}
    final=stage_root/'report.json'
    if final.exists():return verify_saved_stage(json.loads(final.read_text()),binding)
    owner=stage_root/'owner.json';current=identity()
    if owner.exists():
        saved_owner=json.loads(owner.read_text())
        chain.core.require(saved_owner['identity']==current and saved_owner['binding']==binding,
                           'Another logical call/input owns unfinished CPU stage')
    else:
        atomic_json(owner,{'identity':current,'binding':binding});outputs.commit()
    # Manifest/selection is the operation's last write; reuse it if a preemption followed completion.
    marker_name='selection.json' if stage.startswith('freeze_') else 'manifest.json'
    candidates=[]
    for attempt in (stage_root/'attempts').glob('*'):
        marker=attempt/'artifact'/marker_name
        if marker.exists():
            try:
                value=json.loads(marker.read_text())
            except (ValueError,UnicodeDecodeError):
                continue  # Interrupted last-file write; keep partial bytes and use a fresh attempt.
            if value.get('status')=='reviewed_complete':candidates.append(marker)
    if len(candidates)>1:raise ValueError('Multiple completed CPU attempts require reconciliation')
    if candidates:
        marker=candidates[0];value=json.loads(marker.read_text())
        if stage.startswith('freeze_'):
            chain.core.require(any(a.get('sha256')==policy_sha for a in value['approval']['artifacts']),
                'Recovered frozen selection belongs to another policy')
            result={'status':'complete','selection':{'path':str(marker),'sha256':chain.file_sha(marker)}}
        else:
            chain.core.require(value['selection_sha256']==arguments['selection']['sha256'] and
                value['status']=='reviewed_complete','Recovered transport belongs to another selection')
            for spec in value['files']:
                chain.core.require(chain.file_sha(Path(value['transport_root'])/spec['name'])==spec['sha256'],
                    'Recovered completed transport shard changed')
            result={'status':'complete','transport':{'path':str(marker),'sha256':chain.file_sha(marker)},
                    'rows':value['rows'],'task_ids_sha256':value['task_ids_sha256']}
    else:
        attempt=stage_root/'attempts'/uuid.uuid4().hex
        attempt.mkdir(parents=True)
        atomic_json(attempt/'started.json',{'identity':current,'binding':binding,'started_at':time.time()})
        outputs.commit();destination=attempt/'artifact'
        if stage=='freeze_sglang':
            result=chain.freeze_sglang_selection(policy_path,policy_sha,arguments['terminal'],arguments['base'],destination)
        elif stage=='export_sglang':
            from data.export_reviewed_sglang27 import export_reviewed
            selection=chain.document(arguments['selection'])
            value=export_reviewed(arguments['selection']['path'],arguments['selection']['sha256'],destination,
                reviewed_sources=selection['reviewed_sources'],excluded_task_ids=selection['excluded_task_ids'])
            result={'status':'complete','transport':{'path':str(destination/'manifest.json'),
                'sha256':chain.file_sha(destination/'manifest.json')},'rows':value['rows'],'task_ids_sha256':value['task_ids_sha256']}
        elif stage=='freeze_union':
            result=chain.freeze_union_selection(policy_path,policy_sha,arguments['selection'],arguments['transport'],
                                                arguments['base'],destination,arguments.get('additional_components'))
        elif stage=='export_union':
            from data.reviewed_expansion_adapters import export_union
            selection=chain.document(arguments['selection'])
            value=export_union(arguments['selection']['path'],arguments['selection']['sha256'],destination,
                reviewed_sources=selection['reviewed_sources'],excluded_task_ids=selection['excluded_task_ids'])
            result={'status':'complete','transport':{'path':str(destination/'manifest.json'),
                'sha256':chain.file_sha(destination/'manifest.json')},'rows':value['rows'],'task_ids_sha256':value['task_ids_sha256']}
        else:raise ValueError('Unknown finite CPU stage')
    result.update(binding=binding,worker_identity=current)
    atomic_json(final,result);outputs.commit()
    return result


@app.function(**options,cpu=8,memory=32768,timeout=86400,max_containers=4,retries=0)
def append_partition(policy_path:str,policy_sha:str,transport:dict,partition:int):
    reload();policy,_=chain.load_policy(policy_path,policy_sha)
    from data.grouped_stage3_expansion_append import pack_partition
    result=pack_partition(transport['path'],transport['sha256'],partition,policy['append_partitions'],
                          modal.current_function_call_id(),checkpoint=commit)
    path=Path(policy['base']['root'])/'expansion-append'/transport['sha256']/f'part-{partition:03d}'/'report.json'
    return {'status':result['status'],'partition':partition,'report':{'path':str(path),'sha256':chain.file_sha(path)},
            'counts':{length:value['counts'] for length,value in result['outputs'].items()}}


@app.function(**options,cpu=8,memory=65536,timeout=21600,max_containers=1,retries=0)
def finalize_append(policy_path:str,policy_sha:str,transport:dict):
    reload();policy,_=chain.load_policy(policy_path,policy_sha)
    from data.grouped_stage3_expansion_append import finalize
    result=finalize(transport['path'],transport['sha256'],policy['append_partitions'],checkpoint=commit)
    root=Path(policy['base']['root'])
    summary=root/'expansion-append'/transport['sha256']/'summary.json'
    manifests={str(n):{'path':str(root/f'packed-cs16-{n}'/'manifest.json'),
        'sha256':chain.file_sha(root/f'packed-cs16-{n}'/'manifest.json')} for n in (16384,32768)}
    report={'status':'complete','append_summary':{'path':str(summary),'sha256':chain.file_sha(summary)},
        'packed_manifests':manifests,'lengths':result,'transport':transport,'approved_for_release':False}
    atomic_json(root_for(policy_sha)/'append-final.json',report);outputs.commit()
    return report


def child_graph_ids(call_id,suffix):
    def walk(nodes):
        for node in nodes:
            yield node;yield from walk(node.children)
    return {n.function_call_id for n in walk(modal.FunctionCall.from_id(call_id).get_call_graph())
            if n.function_name.endswith(suffix)}


def coordinate(policy_path,policy_sha):
    import time
    reload();policy,policy_spec=chain.load_policy(policy_path,policy_sha)
    root=root_for(policy_sha);path=root/'state.json';current=identity()
    state=json.loads(path.read_text()) if path.exists() else {
        'status':'waiting_generation','policy':policy_spec,'configuration':chain.config(),'children':{},'handoffs':[],
        'generation_call_id':policy['generation']['initial_call_id'],'generation_handoffs':[]}
    chain.core.require(state['policy']==policy_spec and state['configuration']==chain.config(),'Chain state policy/code changed')
    if state.get('completion'):
        chain.core.frozen_bytes(state['completion']);return chain.document(state['completion'])
    def persist():
        state['updated_at']=time.time();atomic_json(path,state);outputs.commit()
    if state['handoffs'] and state['handoffs'][-1].get('status')!='successor_started':
        item=state['handoffs'][-1]
        call_id=item.get('successor_call_id')
        if not call_id:
            candidates=child_graph_ids(item['parent_identity']['function_call_id'],'drive')-set(item['known_successor_ids'])
            chain.core.require(len(candidates)==1,'Uncertain chain handoff cannot be duplicated')
            call_id=candidates.pop();item['successor_call_id']=call_id
        if current==item['parent_identity']:
            persist();return {'status':'handed_off','successor_call_id':call_id,'state_path':str(path)}
        chain.core.require(current['function_call_id']==call_id,'Unrelated coordinator cannot take over finite chain')
        item.update(status='successor_started',successor_identity=current)
    state['coordinator_identity']=current;persist();begin=time.monotonic()
    def handoff():
        if time.monotonic()-begin<17*3600:return None
        chain.core.require(len(state['handoffs'])<128,'Finite chain handoff limit reached')
        item={'parent_identity':current,'known_successor_ids':sorted(child_graph_ids(current['function_call_id'],'drive')),
              'status':'dispatching'}
        state['handoffs'].append(item);persist()
        child=drive.spawn(policy_path,policy_sha)
        item.update(successor_call_id=child.object_id,status='spawned');state['status']='handed_off';persist()
        return {'status':'handed_off','successor_call_id':child.object_id,'state_path':str(path)}
    def maybe_handoff():
        value=handoff()
        if value:raise _Handoff(value)
    def spawn_known(key,function,args,suffix):
        item=state['children'].get(key)
        if item and item.get('result'):return item
        if item and not item.get('call_id'):
            candidates=child_graph_ids(item['coordinator_identity']['function_call_id'],suffix)-{
                value.get('call_id') for value in state['children'].values()}
            chain.core.require(len(candidates)==1,'Uncertain child dispatch requires reconciliation: '+key)
            item['call_id']=candidates.pop();persist()
        if not item:
            item={'status':'dispatching','coordinator_identity':current,'arguments':args}
            state['children'][key]=item;persist()
            child=function.spawn(*args)
            item.update(call_id=child.object_id,status='running');persist()
        chain.core.require(item['arguments']==args,'Child stage arguments changed')
        return item
    def await_child(key,item):
        if item.get('result'):return item['result']
        call=modal.FunctionCall.from_id(item['call_id'])
        while True:
            maybe_handoff()
            try:result=call.get(timeout=55)
            except TimeoutError:continue
            reload();item.update(status='complete',result=result);persist()
            print(json.dumps({'completed_stage':key}),flush=True)
            return result
    try:
        if not state.get('terminal'):
            while True:
                maybe_handoff()
                call=modal.FunctionCall.from_id(state['generation_call_id'])
                try:result=call.get(timeout=55)
                except TimeoutError:continue
                if result.get('status')=='handed_off':
                    next_id=result['successor_call_id']
                    chain.core.require(next_id!=state['generation_call_id'] and next_id not in state['generation_handoffs'],
                        'Generation handoff loop detected')
                    state['generation_handoffs'].append(state['generation_call_id']);state['generation_call_id']=next_id;persist()
                    continue
                reload()
                terminal={'path':result['completion_report_path'],'sha256':result['completion_report_sha256']}
                chain.validate_terminal(policy,terminal);state.update(terminal=terminal,status='waiting_base_manifests');persist();break
        if not state.get('base'):
            while True:
                maybe_handoff();reload();base=chain.base_snapshot(policy)
                if base:state['base']=base;persist();break
                time.sleep(45)
        stages=[('freeze_sglang',{'terminal':state['terminal'],'base':state['base']})]
        first=await_child('freeze_sglang',spawn_known('freeze_sglang',stage_worker,
            [policy_path,policy_sha,'freeze_sglang',stages[0][1]],'stage_worker'))
        core_export=await_child('export_sglang',spawn_known('export_sglang',stage_worker,
            [policy_path,policy_sha,'export_sglang',{'selection':first['selection']}],'stage_worker'))
        if 'additional_components' not in state:
            state['status']='waiting_reviewed_additional_sources';persist()
            while True:
                maybe_handoff();reload()
                ready=chain.ready_additional_components(policy,policy_sha,state['base'])
                if ready is not None:state['additional_components']=ready;persist();break
                time.sleep(45)
        union=await_child('freeze_union',spawn_known('freeze_union',stage_worker,
            [policy_path,policy_sha,'freeze_union',{'selection':first['selection'],'transport':core_export['transport'],'base':state['base'],
                'additional_components':state['additional_components']}],'stage_worker'))
        exported=await_child('export_union',spawn_known('export_union',stage_worker,
            [policy_path,policy_sha,'export_union',{'selection':union['selection']}],'stage_worker'))
        state['status']='appending_both_lengths';persist()
        # Spawn bounded independent CPU partitions; no map cancellation on a sibling preemption.
        for partition in range(policy['append_partitions']):
            maybe_handoff();spawn_known(f'append-{partition:03d}',append_partition,
                [policy_path,policy_sha,exported['transport'],partition],'append_partition')
        for partition in range(policy['append_partitions']):
            key=f'append-{partition:03d}';await_child(key,state['children'][key])
        final=await_child('finalize_append',spawn_known('finalize_append',finalize_append,
            [policy_path,policy_sha,exported['transport']],'finalize_append'))
        complete={'status':'complete','policy':policy_spec,'configuration':chain.config(),
            'generation_terminal':state['terminal'],'generation_final_call_id':state['generation_call_id'],
            'generation_handoffs':state['generation_handoffs'],'chain_handoffs':state['handoffs'],
            'base_snapshot':state['base'],'sglang_selection':first['selection'],'sglang_transport':core_export['transport'],
            'union_selection':union['selection'],'union_transport':exported['transport'],
            'additional_components':state['additional_components'],
            'packing':final,'approved_for_release':False,'hf_upload_started':False,'model_training_started':False}
        state['completion']=chain.write_document(root/'completion.json',complete);state['status']='complete';persist()
        return complete
    except _Handoff as exc:return exc.result
    except Exception as exc:
        state.update(status='needs_attention',error={'type':type(exc).__name__,'message':str(exc)})
        persist();raise


class _Handoff(Exception):
    def __init__(self,result):self.result=result


@app.function(**options,cpu=2,memory=8192,timeout=86400,max_containers=1,retries=0)
def drive(policy_path:str,policy_sha:str):
    return coordinate(policy_path,policy_sha)


@app.function(**options,cpu=2,memory=8192,timeout=1800)
def status(policy_sha:str):
    outputs.reload();path=root_for(policy_sha)/'state.json'
    return json.loads(path.read_text()) if path.exists() else {'status':'not_started','root':str(root_for(policy_sha))}
