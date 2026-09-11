"""Pure frozen-selection stages for an explicitly reviewed private packing chain.

This code never chooses sources, approves data, dispatches jobs, or publishes.
Root's policy authorizes deterministic selection; generation verdicts/raw traces
remain unchanged. All generated selections bind that policy and actual artifacts.
"""
from __future__ import annotations
import copy
import hashlib
import json
import sqlite3
import tempfile
from collections import Counter
from pathlib import Path

from data import export_reviewed_sglang27 as core
from data import reviewed_expansion_adapters as adapters
from data.expansion_training_quality import inspect_expansion_efficiency

SCHEMA = 'finite-reviewed-expansion-packing-v1'
BASE_ROOT = '/data/stage3-build-20260906/grouped-packing-20260909-v3'
CHAIN_ROOT = '/runs/stage3-build-20260906/expansion-completion-chain-20260909-v1'
CODE_FILES = ('expansion_completion_chain.py', 'expansion_completion_chain_modal.py',
    'expansion_training_quality.py', 'export_reviewed_sglang27.py', 'reviewed_expansion_adapters.py',
    'grouped_stage3_expansion_append.py', 'grouped_stage3_packing.py', 'grouped_stage3_packing_modal.py',
    'grouped_stage3_repair_modal.py', 'stage3_tokenizers.py',
    'preprocess_for_dynamic_packing.py', 'dynamic_packing_dataset.py',
    'clean_agent_trajectories.py', 'harvest_expansion_trace.py', 'expansion_task_normalization.py')


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8*1024*1024), b''): h.update(block)
    return h.hexdigest()


def config():
    return {'version': SCHEMA, 'cpu_only': True, 'approved_for_release': False,
        'generation': 'wait for actual frozen v3 terminal report; follow recorded coordinator handoffs',
        'efficiency_policy': 'no-repeated-immutable-expansion-v1', 'lengths': [16384,32768],
        'code_sha256': {name:file_sha(Path(__file__).parent/name) for name in CODE_FILES}}


def document(spec):
    return json.loads(core.frozen_bytes(spec))


def write_document(path, value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    raw=(json.dumps(value,ensure_ascii=False,indent=2)+'\n').encode()
    if path.exists():
        core.require(path.read_bytes()==raw,'Refusing to overwrite a frozen selection artifact')
    else:
        with path.open('xb') as handle: handle.write(raw)
    return {'path':str(path),'sha256':core.sha(raw)}


def load_policy(path, expected_sha):
    spec={'path':str(core.absolute(path)), 'sha256':expected_sha}
    policy=document(spec)
    core.require(policy.get('schema')==SCHEMA and policy.get('reviewed_by')=='root' and
        policy.get('approved_for_internal_packing') is True and policy.get('approved_for_release') is False,
        'Missing root-reviewed internal-packing policy')
    core.require(policy.get('configuration')==config(),'Chain/quality/adapter code differs from reviewed policy')
    allowed=policy.get('reviewed_sources',[])
    sg=policy.get('sglang27_sources',[])
    core.require('lex_glue' not in sg,'Original opaque-label LexGLUE cannot enter the core selection')
    dependencies=policy.get('additional_readiness_dependencies',[])
    additional_sources={source for dep in dependencies for source in dep['source_allowlist']}
    core.require(allowed and len(set(allowed))==len(allowed) and sg and len(set(sg))==len(sg) and set(sg)<=set(allowed),
        'Explicit unique source allowlists required')
    reviews=policy.get('source_reviews',{})
    core.require(set(reviews)==set(allowed)-additional_sources,
        'Original selected sources need root review; additional sources require their separate readiness review')
    artifacts=policy.get('approval_artifacts',[])
    core.require(artifacts,'Frozen root review artifacts required')
    for name,review in reviews.items():
        core.require(review.get('approved_for_internal_packing') is True and str(review.get('evidence_review','')).strip()
            and review.get('artifacts'),'Missing source-specific internal quality approval')
        artifacts=[*artifacts,*review['artifacts']]
    for artifact in artifacts: core.frozen_bytes(artifact)
    manual=document(policy['manual_exclusions'])
    ids=[core.valid_id(row['task_id']) for row in manual['entries']]
    core.require(manual['reviewed_by']=='root' and manual['count']==len(ids)==len(set(ids)),
        'Documented manual exclusions are inconsistent')
    core.require(policy['base']['root']==BASE_ROOT,'Unexpected base pack target')
    core.require(type(policy.get('append_partitions')) is int and 1<=policy['append_partitions']<=64,
        'Bounded append partition count required')
    generation=policy['generation']
    core.require(generation.get('initial_call_id') and generation.get('app')=='lclm-sglang27-continuation-storage-20260909-v3',
        'Actual v3 continuation call required')
    manifest=document(generation['manifest'])
    decision=document(generation['continuation_decision'])
    resume=document(generation['resume_probe_decision'])
    core.require(decision['backlog_manifest_sha256']==generation['manifest']['sha256'] and
        decision['resume_probe_decision_sha256']==generation['resume_probe_decision']['sha256'] and
        decision['approved_for_generation'] is True and decision['reviewed_by']=='root' and
        set(sg)<=set(decision['source_reviews']),'Internal source policy is not bound to reviewed generation')
    core.require(resume['backlog_manifest_sha256']==generation['manifest']['sha256'] and
        manifest['pending_rows']==272287,'Unexpected generation lineage')
    stream_names=[]; stream_sources=set(sg)
    for dependency in dependencies:
        core.require(dependency['stream_id'] not in ('sglang27','legacy','pilot') and
            dependency['source_allowlist'] and len(set(dependency['source_allowlist']))==len(dependency['source_allowlist']) and
            not set(dependency['source_allowlist']) & stream_sources and
            set(dependency['source_allowlist'])<=set(allowed) and dependency['task_version'],
            'Additional source dependency overlaps an original teacher stream')
        core.absolute(dependency['readiness_path']);core.frozen_bytes(dependency['ontology_artifact'])
        stream_names.append(dependency['stream_id']);stream_sources.update(dependency['source_allowlist'])
    for stream in policy.get('retained_streams',[]):
        core.require(stream['kind'] in ('legacy_accepted_jsonl','frozen_pilot_jsonl'), 'Unknown retained teacher type')
        core.require('lex_glue' not in stream['source_allowlist'],'Opaque legacy/pilot LexGLUE is blocked')
        core.require(not set(stream['source_allowlist']) & additional_sources,'Additional source also present in an original teacher stream')
        core.require(stream['stream_id']!='sglang27' and stream['source_allowlist'] and
            set(stream['source_allowlist'])<=set(allowed),'Unreviewed retained stream source')
        stream_names.append(stream['stream_id']); stream_sources.update(stream['source_allowlist'])
    core.require(len(stream_names)==len(set(stream_names)) and stream_sources==set(allowed),
        'Source policy/typed stream union differs')
    return policy, spec


def base_snapshot(policy):
    root=Path(policy['base']['root'])
    summary=root/'partitions'/'summary.json'
    if not summary.exists(): return None
    summary_value=json.loads(summary.read_text())
    result={'root':str(root),'manifest_sha256':{}}
    for length in (16384,32768):
        path=root/f'packed-cs16-{length}'/'base-native-manifest.json'
        if not path.exists(): path=path.with_name('manifest.json')
        if not path.exists(): return None
        result['manifest_sha256'][str(length)]=file_sha(path)
    # The append adapter enforces status, full output-byte audit, revisions, sizes and no expansion.
    from data.grouped_stage3_expansion_append import verify_base
    verify_base({'base_snapshot':result})
    if policy['base'].get('manifest_sha256'):
        core.require(result['manifest_sha256']==policy['base']['manifest_sha256'],'Root-reviewed base bytes changed')
    result['summary']={'path':str(summary),'sha256':file_sha(summary)}
    return result


def validate_terminal(policy, terminal_spec):
    terminal=document(terminal_spec)
    manifest=document(policy['generation']['manifest'])
    core.require(terminal.get('successor_terminal_status')=='terminal' and terminal.get('stage')=='continuation'
        and terminal.get('manifest_sha256')==policy['generation']['manifest']['sha256']
        and terminal.get('decision_sha256')==policy['generation']['continuation_decision']['sha256'],
        'Generation terminal is absent or differently bound')
    for source in policy['sglang27_sources']:
        expected,=[s for s in manifest['sources'] if s['source']==source]
        actual=terminal['sources'][source]
        core.require(actual['approved_for_stage'] is True and not actual['held'] and actual['unresolved']==0
            and actual['unattempted']==0 and actual['completed']==actual['expected_rows']==expected['rows'],
            'Selected generation source is held/incomplete/unresolved: '+source)
        specs={c['index']:c for c in expected['chunks'] if c['index']>0}
        chunks=actual['chunks']
        core.require(len(chunks)==len(specs) and {c['index'] for c in chunks}==set(specs),
            'Missing/duplicate completed generation chunks')
        for chunk in chunks:
            spec=specs[chunk['index']]
            core.require(chunk['status']=='complete' and chunk['unresolved']==0 and chunk['completed']==spec['rows']
                and chunk['input_sha256']==spec['sha256']
                and chunk['effective_task_ids_sha256']==spec['task_ids_sha256'],
                'Terminal chunk coverage changed')
            report=document({'path':chunk['report_path'],'sha256':chunk['report_sha256']})
            core.require(report['backlog_manifest_sha256']==policy['generation']['manifest']['sha256']
                and report['decision_sha256']==policy['generation']['continuation_decision']['sha256'],
                'Completed source report has different generation binding')
    return terminal,manifest


def approvals(policy, policy_spec, *artifacts):
    all_artifacts=[policy_spec,policy['manual_exclusions'],*policy['approval_artifacts'],*artifacts]
    for review in policy['source_reviews'].values(): all_artifacts.extend(review['artifacts'])
    seen=set(); distinct=[]
    for spec in all_artifacts:
        key=(spec['path'],spec['sha256'])
        if key not in seen: distinct.append(spec);seen.add(key)
    return {'status':'approved','scope':'internal_private_packing','public_release_approved':False,
        'authority':'Deterministic application of root policy; no new approval inferred.', 'artifacts':distinct}


def evaluate_training_exclusion(messages, task_id, manual_ids):
    report=inspect_expansion_efficiency(messages)
    dynamic=report['eligible'] is False
    manual=task_id in manual_ids
    return manual or dynamic, {'task_id':task_id,'manual':manual,'repeated_immutable_expansion':dynamic,
        'overlap':manual and dynamic,'efficiency':report}


def freeze_sglang_selection(policy_path, policy_sha, terminal_spec, base, destination):
    policy,policy_spec=load_policy(policy_path,policy_sha)
    terminal,manifest=validate_terminal(policy,terminal_spec)
    output=core.absolute(destination);output.mkdir(parents=True,exist_ok=False)
    manual={row['task_id'] for row in document(policy['manual_exclusions'])['entries']}
    excluded=set(manual);exclusion_counts=Counter();quality_sources={};sources=[]
    resume=document(policy['generation']['resume_probe_decision'])
    decision=document(policy['generation']['continuation_decision'])
    root=Path(policy['generation']['manifest']['path']).parent
    origin=Path(manifest['origin_probe_manifest_path']).parent
    with tempfile.TemporaryDirectory(prefix='chain-sg-ids-') as temp:
        db=sqlite3.connect(str(Path(temp)/'ids.sqlite'))
        db.execute('CREATE TABLE ids (task_id TEXT PRIMARY KEY, source TEXT, exported INTEGER)')
        db.execute('PRAGMA cache_size=-1024')
        with (output/'training-exclusions.jsonl').open('x') as ex_handle:
            for name in policy['sglang27_sources']:
                source,=[s for s in manifest['sources'] if s['source']==name]
                counts=dict.fromkeys(('results','accepted','excluded_accepted','exported'),0)
                source_quality=Counter();quality_sources[name]=source_quality
                chunks=[]
                for spec in source['chunks']:
                    directory=((origin if name in resume['reused_probe_reports'] else root)/'outputs'/name/'chunk-00000'
                               if spec['index']==0 else root/'outputs'/name/f"chunk-{spec['index']:05d}")
                    report_spec={'path':str(directory/'report.json'),'sha256':file_sha(directory/'report.json')}
                    if spec['index']==0:
                        core.require(report_spec['sha256']==decision['source_reviews'][name]['probe_report_sha256'],
                            'Source probe bytes differ from root continuation review')
                    else:
                        frozen,=[c for c in terminal['sources'][name]['chunks'] if c['index']==spec['index']]
                        core.require(report_spec['path']==frozen['report_path'] and report_spec['sha256']==frozen['report_sha256'],
                            'Generation result report changed after terminal accounting')
                    report=document(report_spec);tasks=core.frozen_tasks(spec)
                    core.require(report['status']=='complete' and report['rows']==report['completed']==len(tasks)
                        and report['source']==name and report['input_sha256']==spec['sha256']
                        and set(report['result_hashes'])==set(tasks),'Incomplete actual source chunk')
                    accepted_count=0
                    for task_id in sorted(tasks):
                        result=document({'path':str(directory/'results'/(task_id+'.json')),
                            'sha256':report['result_hashes'][task_id]})
                        core.require(result['task_id']==task_id and result['source']==name and
                            result['task_sha256']==core.sha(json.dumps(tasks[task_id],sort_keys=True).encode()),
                            'Frozen result/task identity changed')
                        accepted=result['verification']['accepted'];core.require(type(accepted) is bool,'Nonboolean verdict')
                        exclude=False
                        if accepted:
                            exclude,info=evaluate_training_exclusion(result['trace']['messages'],task_id,manual)
                            values=dict(accepted=1,manual=info['manual'],dynamic=info['repeated_immutable_expansion'],
                                overlap=info['overlap'],excluded_union=exclude,eligible=not exclude)
                            exclusion_counts.update(values);source_quality.update(values)
                            if exclude:
                                excluded.add(task_id);info.update(source=name,result_sha256=report['result_hashes'][task_id])
                                ex_handle.write(json.dumps(info)+'\n')
                        selected=accepted and not exclude
                        try: db.execute('INSERT INTO ids VALUES (?,?,?)',(task_id,name,int(selected)))
                        except sqlite3.IntegrityError as exc: raise ValueError('Duplicate task across frozen27B chunks') from exc
                        counts['results']+=1;counts['accepted']+=accepted;accepted_count+=accepted
                        counts['excluded_accepted']+=accepted and exclude;counts['exported']+=selected
                    core.require(accepted_count==report['counts']['automatic_accepted'],'Actual accepted count changed')
                    chunks.append({'tasks':spec,'report':report_spec,'results_root':str(directory/'results'),
                                   'attempts_root':str(directory/'attempts')})
                    db.commit()
                sources.append({'source':name,'counts':counts,'task_ids_sha256':core.ids_digest(db,name),'chunks':chunks})
        db.close()
    exclusion_spec={'path':str(output/'training-exclusions.jsonl'),'sha256':file_sha(output/'training-exclusions.jsonl')}
    quality_spec=write_document(output/'training-quality-accounting.json',{'policy':inspect_expansion_efficiency.__module__,
        'code_sha256':file_sha(Path(__file__).parent/'expansion_training_quality.py'),
        'counts':dict(exclusion_counts),'by_source':{name:dict(value) for name,value in quality_sources.items()},
        'manual_policy_ids_count':len(manual),'count_scope':'Accepted27B trace candidates; exclusions are training decisions.',
        'exclusions':exclusion_spec,'generation_verdicts_unchanged':True})
    selection={'schema':core.SELECTION_SCHEMA,'status':'reviewed_complete','model':core.MODEL,
        'model_revision':core.REVISION,'reviewed_sources':policy['sglang27_sources'],
        'excluded_task_ids':sorted(excluded),'base_snapshot':base,'sources':sources,
        'normalization':{'module_sha256':file_sha(Path(__file__).parent/'expansion_task_normalization.py'),
            'maud_choices':{'path':manifest['maud_ontology_path'],'sha256':manifest['maud_ontology_sha256']}},
        'approval':approvals(policy,policy_spec,terminal_spec,base['summary'],quality_spec,exclusion_spec),
        'training_quality_accounting':quality_spec,'generation_terminal':terminal_spec,
        'generation_manifest':policy['generation']['manifest']}
    selection_spec=write_document(output/'selection.json',selection)
    return {'status':'complete','selection':selection_spec,'quality_accounting':quality_spec,
            'source_counts':{s['source']:s['counts'] for s in sources}}


def freeze_union_selection(policy_path, policy_sha, core_selection_spec, transport_spec, base, destination, additional_components=None):
    policy,policy_spec=load_policy(policy_path,policy_sha)
    core_selection=document(core_selection_spec);transport=document(transport_spec)
    core.require(transport['selection_sha256']==core_selection_spec['sha256'] and
        transport['status']=='reviewed_complete','Strict27B export has not completed')
    output=core.absolute(destination);output.mkdir(parents=True,exist_ok=False)
    manual={row['task_id'] for row in document(policy['manual_exclusions'])['entries']}
    excluded=set(core_selection['excluded_task_ids'])|manual
    allowed=policy['reviewed_sources'];streams=[];counts={name:dict.fromkeys(('results','accepted','excluded_accepted','exported'),0) for name in allowed}
    dynamic_counts=Counter();quality_streams={}
    with tempfile.TemporaryDirectory(prefix='chain-union-ids-') as temp:
        db=sqlite3.connect(str(Path(temp)/'ids.sqlite'))
        db.execute('PRAGMA cache_size=-1024')
        db.execute('CREATE TABLE ids (task_id TEXT PRIMARY KEY, source TEXT, exported INTEGER)')
        db.execute('CREATE TABLE stream_ids (stream TEXT, task_id TEXT, exported INTEGER, PRIMARY KEY(stream,task_id))')
        with (output/'retained-training-exclusions.jsonl').open('x') as ex_handle:
            for original in policy.get('retained_streams',[]):
                stream=copy.deepcopy(original);name=stream['stream_id'];stream_quality=Counter();quality_streams[name]=stream_quality
                iterator=(adapters.iter_legacy_accepted(stream,selection_sha=policy_sha) if stream['kind']=='legacy_accepted_jsonl'
                          else adapters.iter_frozen_pilot(stream,selection_sha=policy_sha))
                scounts={s:dict.fromkeys(('results','accepted','excluded_accepted','exported'),0) for s in stream['source_allowlist']}
                for item in iterator:
                    key,source,accepted=item['task_id'],item['source'],item['accepted']
                    core.require(source in scounts,'Retained stream yielded unapproved source')
                    exclude=False
                    if accepted:
                        messages=item['row']['messages']
                        if isinstance(messages,str):messages=json.loads(messages)
                        exclude,info=evaluate_training_exclusion(messages,key,manual)
                        values=dict(accepted=1,manual=info['manual'],dynamic=info['repeated_immutable_expansion'],
                            overlap=info['overlap'],excluded_union=exclude,eligible=not exclude)
                        dynamic_counts.update(values);stream_quality.update(values)
                        if exclude:
                            excluded.add(key);info.update(source=source,stream_id=name)
                            ex_handle.write(json.dumps(info)+'\n')
                    selected=accepted and not exclude
                    try:
                        db.execute('INSERT INTO stream_ids VALUES (?,?,?)',(name,key,int(selected)))
                        if selected: db.execute('INSERT INTO ids VALUES (?,?,1)',(key,source))
                    except sqlite3.IntegrityError as exc: raise ValueError('Duplicate retained attempt or exported teacher task ID') from exc
                    scounts[source]['results']+=1;scounts[source]['accepted']+=accepted
                    scounts[source]['excluded_accepted']+=accepted and exclude;scounts[source]['exported']+=selected
                h=hashlib.sha256();n=0
                for key, in db.execute('SELECT task_id FROM stream_ids WHERE stream=? AND exported=1 ORDER BY task_id',(name,)):
                    h.update((key+'\n').encode());n+=1
                if not n:h.update(b'\n')
                stream.update(counts=scounts,task_ids_sha256=h.hexdigest());streams.append(stream)
                for source,values in scounts.items():
                    for k,v in values.items():counts[source][k]+=v
                db.commit()
        # Every separately committed component is checked again under the same current quality policy.
        import pyarrow.parquet as pq
        components=[{'stream_id':'sglang27','selection':core_selection_spec,'transport':transport_spec,
            'source_allowlist':policy['sglang27_sources']},*(additional_components or [])]
        core.require({c['stream_id'] for c in additional_components or []}==
            {d['stream_id'] for d in policy.get('additional_readiness_dependencies',[])},
            'Required corrected-source readiness is missing')
        for component in components:
            current= document(component['transport'])
            selected=document(component['selection'])
            core.require(current['selection_sha256']==component['selection']['sha256'] and
                current['base_snapshot']==base,'Additional transport/selection/base binding changed')
            excluded.update(selected['excluded_task_ids'])
            component_counts={s:dict.fromkeys(('results','accepted','excluded_accepted','exported'),0)
                              for s in component['source_allowlist']}
            for entry in current['files']:
                path=Path(current['transport_root'])/entry['name']
                core.require(file_sha(path)==entry['sha256'],'Completed strict transport changed')
                for batch in pq.ParquetFile(path).iter_batches(batch_size=32,columns=['task_id','source','messages']):
                    for row in batch.to_pylist():
                        key,source=row['task_id'],row['source']
                        core.require(source in component_counts and key not in excluded,
                            'Strict transport has unreviewed/excluded task')
                        messages=json.loads(row['messages']) if isinstance(row['messages'],str) else row['messages']
                        core.require(inspect_expansion_efficiency(messages)['eligible'],
                            'Additional transport violates the uniform no-repeat training policy')
                        try:
                            db.execute('INSERT INTO ids VALUES (?,?,1)',(key,source))
                            db.execute('INSERT INTO stream_ids VALUES (?,?,1)',(component['stream_id'],key))
                        except sqlite3.IntegrityError as exc: raise ValueError('Duplicate exported task across teacher streams') from exc
                        for k in ('results','accepted','exported'):component_counts[source][k]+=1
            core.require(sum(v['exported'] for v in component_counts.values())==current['rows'],
                'Strict transport row count changed')
            streams.append({**component,'kind':'sglang27_task_results','counts':component_counts,
                            'task_ids_sha256':current['task_ids_sha256']})
            for source,values in component_counts.items():
                for k,v in values.items():counts[source][k]+=v
        total_digest=core.ids_digest(db);db.close()
    exclusion_spec={'path':str(output/'retained-training-exclusions.jsonl'),'sha256':file_sha(output/'retained-training-exclusions.jsonl')}
    combined_quality=Counter(document(core_selection['training_quality_accounting'])['counts']);combined_quality.update(dynamic_counts)
    for component in additional_components or []:
        upstream=document(component['selection'])
        extra_quality=document(upstream['training_quality_accounting'])
        combined_quality.update(extra_quality['counts'])
    quality_spec=write_document(output/'retained-training-quality-accounting.json',{'counts':dict(dynamic_counts),
        'by_stream':{name:dict(value) for name,value in quality_streams.items()},
        'combined_accepted_trace_candidate_counts':dict(combined_quality),
        'manual_policy_ids_count':len(manual),'count_scope':'Accepted trace candidates per typed teacher stream; generation verdicts are unchanged.',
        'exclusions':exclusion_spec,'code_sha256':file_sha(Path(__file__).parent/'expansion_training_quality.py'),
        'strict27b_quality_accounting':core_selection['training_quality_accounting'],'generation_verdicts_unchanged':True})
    selection={'schema':adapters.SCHEMA,'status':'reviewed_complete','reviewed_sources':allowed,
        'excluded_task_ids':sorted(excluded),'exclusions':policy['manual_exclusions'],
        'approval':approvals(policy,policy_spec,core_selection_spec,transport_spec,quality_spec,exclusion_spec,
            *[c['readiness'] for c in additional_components or []]),
        'base_snapshot':base,'streams':streams,'source_counts':counts,'task_ids_sha256':total_digest,
        'training_quality_accounting':quality_spec}
    spec=write_document(output/'selection.json',selection)
    return {'status':'complete','selection':spec,'quality_accounting':quality_spec,'source_counts':counts}


def ready_additional_components(policy, policy_sha, base):
    """Wait on explicit corrected-source readiness; old opaque Lex is never a fallback."""
    components=[]
    for dependency in policy.get('additional_readiness_dependencies',[]):
        path=Path(dependency['readiness_path'])
        if not path.exists():return None
        spec={'path':str(path),'sha256':file_sha(path)};ready=document(spec)
        core.require(ready.get('schema')=='reviewed-corrected-expansion-ready-v1' and ready.get('status')=='complete'
            and ready.get('chain_policy_sha256')==policy_sha and ready.get('task_version')==dependency['task_version']
            and ready.get('ontology_artifact')==dependency['ontology_artifact']
            and ready.get('source_allowlist')==dependency['source_allowlist']
            and ready.get('training_quality_code_sha256')==file_sha(Path(__file__).parent/'expansion_training_quality.py')
            and ready.get('manual_exclusions_sha256')==policy['manual_exclusions']['sha256'],
            'Corrected-source readiness has different policy/task/ontology/quality bindings')
        review=document(ready['source_review']);generation=document(ready['generation_manifest'])
        terminal=document(ready['generation_terminal']);selection=document(ready['selection'])
        transport=document(ready['transport']);audit=document(ready['format_audit'])
        core.require(review.get('reviewed_by')=='root' and review.get('approved_for_internal_packing') is True
            and review.get('approved_for_release') is False and review.get('task_version')==dependency['task_version']
            and review.get('ontology_artifact')==dependency['ontology_artifact']
            and set(review.get('source_allowlist',[]))==set(dependency['source_allowlist'])
            and review.get('generation_manifest_sha256')==ready['generation_manifest']['sha256'],
            'Corrected source lacks its separate explicit root review')
        core.require(generation.get('task_version')==dependency['task_version'] and
            generation.get('ontology_artifact')==dependency['ontology_artifact'],
            'Actual generation manifest is not the reviewed corrected task version')
        core.require(terminal.get('manifest_sha256')==ready['generation_manifest']['sha256'] and
            terminal.get('successor_terminal_status')=='terminal','Corrected generation has not reached a bound terminal report')
        for source in dependency['source_allowlist']:
            done=terminal['sources'][source]
            core.require(done['approved_for_stage'] is True and not done['held'] and done['unresolved']==0 and
                done['unattempted']==0 and done['completed']==done['expected_rows']>0,
                'Corrected source has incomplete or unresolved attempted IDs')
        core.require(selection.get('generation_manifest')==ready['generation_manifest'] and
            selection.get('generation_terminal')==ready['generation_terminal'] and
            set(selection['reviewed_sources'])==set(dependency['source_allowlist']) and
            selection['base_snapshot']==base and transport['base_snapshot']==base and
            transport['selection_sha256']==ready['selection']['sha256'] and
            transport['status']=='reviewed_complete','Corrected frozen selection/transport lineage changed')
        core.require(audit.get('status')=='passed' and audit.get('selection_sha256')==ready['selection']['sha256'] and
            audit.get('transport_manifest_sha256')==ready['transport']['sha256'] and audit.get('failed_rows')==0,
            'Corrected transport lacks a bound completed format audit')
        quality=document(selection['training_quality_accounting'])
        core.require(quality.get('code_sha256')==ready['training_quality_code_sha256'],
            'Corrected selection used a different training quality policy')
        components.append({'stream_id':dependency['stream_id'],'source_allowlist':dependency['source_allowlist'],
            'selection':ready['selection'],'transport':ready['transport'],'readiness':spec})
    return components
