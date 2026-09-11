import copy
import json
from pathlib import Path
import pyarrow.parquet as pq
import pytest
from data import lexglue_completion as completed
from data import expansion_completion_chain as chain
from data import export_reviewed_sglang27 as core
from test_expansion_completion_chain import setup as original_setup,corrected_ready_fixture
from test_export_reviewed_sglang27 import fixture as sg_fixture,write_json,id_digest


def setup(tmp_path,monkeypatch):
    policy,_,base=original_setup(tmp_path/'original')
    ontology=write_json(tmp_path/'ontology.json',{'fixture':'corrected primary label definitions'})
    monkeypatch.setattr(completed,'ONTOLOGY_PATH',ontology['path'])
    monkeypatch.setattr(completed,'DEFINITIONS_SHA256',ontology['sha256'])
    dependency={'stream_id':'corrected_lexglue','source_allowlist':['lex_glue'],'task_version':completed.VERSION,
        'ontology_artifact':ontology,'readiness_path':str(tmp_path/'ready.json')}
    policy['reviewed_sources'].append('lex_glue');policy['additional_readiness_dependencies']=[dependency]
    policy_spec=write_json(tmp_path/'policy.json',policy)
    root=tmp_path/'corrected';chunks=[];report_specs=[]
    for index,rows in enumerate([[('probe',True)],[('good',True),('repeat',True),('both',True),('extra',True),('bad',False)]]):
        directory=root/'outputs/lex_glue'/f'chunk-{index:05d}'
        def mutate(key,result,raw):
            if key in ('repeat','both'):
                assistant=copy.deepcopy(result['trace']['messages'][2]);assistant['tool_calls'][0]['id']='call-2'
                tool=copy.deepcopy(result['trace']['messages'][3]);tool['tool_call_id']='call-2'
                result['trace']['messages'][4:4]=[assistant,tool]
        selected=sg_fixture(directory,rows=rows,mutate=mutate)
        (directory/'raw').rename(directory/'attempts')
        tasks=[json.loads(line) for line in (directory/'tasks.jsonl').read_text().splitlines()]
        hashes={}
        for ordinal,task in enumerate(tasks):
            task['source_row_id']=f'scotus-{index*10+ordinal}'
            result=json.loads((directory/'results'/(task['task_id']+'.json')).read_text())
            result.update(source='lex_glue',source_row_id=task['source_row_id'],
                task_sha256=core.sha(json.dumps(task,sort_keys=True).encode()))
            result['trace'].update(source_row_id=task['source_row_id'],sub_dataset='lex_glue')
            hashes[task['task_id']]=write_json(directory/'results'/(task['task_id']+'.json'),result)['sha256']
        raw=b''.join((json.dumps(task)+'\n').encode() for task in tasks)
        (directory/'tasks.jsonl').write_bytes(raw)
        spec={'path':str(directory/'tasks.jsonl'),'sha256':core.sha(raw),'rows':len(rows),'index':index,
              'task_ids_sha256':core.sha(json.dumps(sorted(key for key,_ in rows)).encode())}
        chunks.append(spec)
        report=json.loads((directory/'report.json').read_text())
        report.update(source='lex_glue',chunk=index,input_sha256=spec['sha256'],result_hashes=hashes,
            task_version=completed.VERSION,ontology_artifact=ontology)
        report_specs.append((directory/'report.json',report))
    manifest=write_json(root/'manifest.json',{'task_version':completed.VERSION,'ontology_artifact':ontology,
        'generation_config':completed.generator_config(),'original_pending_rows':6,
        'sources':[{'source':'lex_glue','rows':5,'probe_rows':1,'chunks':chunks}]})
    probe_path,probe_report=report_specs[0]
    probe_report.update(backlog_manifest_sha256=manifest['sha256'],decision_sha256='a'*64)
    probe_spec=write_json(probe_path,probe_report)
    extra=write_json(tmp_path/'extra-exclusions.json',{'reviewed_by':'root','count':1,'entries':[{'task_id':'extra'}]})
    review=write_json(root/'continuation-decision.json',{'reviewed_by':'root','decision':'continue_corrected_lexglue',
        'approved_for_generation':True,'approved_for_internal_packing':True,'approved_for_release':False,
        'source_allowlist':['lex_glue'],'generation_manifest_sha256':manifest['sha256'],
        'backlog_manifest_sha256':manifest['sha256'],'ontology_artifact':ontology,'task_version':completed.VERSION,
        'source_reviews':{'lex_glue':{'probe_report_sha256':probe_spec['sha256']}},'additional_manual_exclusions':extra,
        'completion_configuration':completed.configuration()})
    report_path,report=report_specs[1]
    report.update(backlog_manifest_sha256=manifest['sha256'],decision_sha256=review['sha256'])
    report_spec=write_json(report_path,report)
    terminal=write_json(root/'terminal.json',{'successor_terminal_status':'terminal','stage':'continuation',
        'manifest_sha256':manifest['sha256'],'decision_sha256':review['sha256'],
        'task_version':completed.VERSION,'ontology_artifact':ontology,'sources':{'lex_glue':{
            'approved_for_stage':True,'held':False,'unresolved':0,'unattempted':0,'completed':5,'expected_rows':5,
            'chunks':[{'index':1,'status':'complete','completed':5,'unresolved':0,'input_sha256':chunks[1]['sha256'],
                'effective_task_ids_sha256':chunks[1]['task_ids_sha256'],'report_path':report_spec['path'],
                'report_sha256':report_spec['sha256']}]}}})
    return policy_spec,review,manifest,terminal,base


def freeze(tmp_path,monkeypatch):
    policy,review,manifest,terminal,base=setup(tmp_path,monkeypatch)
    selected=completed.freeze_selection(policy['path'],policy['sha256'],review,manifest,terminal,base,tmp_path/'selection')
    return policy,review,manifest,terminal,base,selected


def test_frozen_corrected_selection_has_exact_quality_counts_and_extra_manual_binding(tmp_path,monkeypatch):
    policy,review,manifest,terminal,base=setup(tmp_path,monkeypatch)
    unchanged={p:chain.file_sha(p) for p in (tmp_path/'corrected').rglob('*') if p.is_file()}
    result=completed.freeze_selection(policy['path'],policy['sha256'],review,manifest,terminal,base,tmp_path/'selection')
    selection=chain.document(result['selection']);quality=chain.document(result['quality_accounting'])
    assert selection['sources'][0]['counts']=={'results':6,'accepted':5,'excluded_accepted':3,'exported':2}
    assert selection['sources'][0]['task_ids_sha256']==id_digest(['probe','good'])
    assert quality['counts']==dict(accepted=5,manual=2,dynamic=2,overlap=1,excluded_union=3,eligible=2)
    assert quality['by_config']=={'scotus':quality['counts']}
    assert quality['manual_exclusions']==chain.document(policy)['manual_exclusions']
    assert quality['additional_manual_exclusions']==chain.document(review)['additional_manual_exclusions']
    assert {p:chain.file_sha(p) for p in unchanged}==unchanged
    transport=core.export_reviewed(result['selection']['path'],result['selection']['sha256'],tmp_path/'transport',
        reviewed_sources=['lex_glue'],excluded_task_ids=selection['excluded_task_ids'])
    assert transport['rows']==2 and transport['base_snapshot']==base
    for entry in transport['files']:
        for row in pq.read_table(Path(transport['transport_root'])/entry['name']).to_pylist():
            trace,source,allowed=completed.transport_payload(row)
            assert trace['task_id']==row['task_id'] and source=='lex_glue' and allowed is None
            assert trace['messages']==json.loads(row['messages'])


@pytest.mark.parametrize('field,value',[('approved_for_internal_packing',False),('generation_manifest_sha256','0'*64),
    ('source_allowlist',['lex_glue','fixture']),('task_version','opaque'),('completion_configuration',{})])
def test_actual_source_decision_must_authorize_corrected_internal_packing(tmp_path,monkeypatch,field,value):
    policy,review,manifest,_,_=setup(tmp_path,monkeypatch)
    changed=chain.document(review);changed[field]=value;review=write_json(tmp_path/'changed.json',changed)
    with pytest.raises(ValueError):completed.governing(policy['path'],policy['sha256'],review,manifest)


@pytest.mark.parametrize('field,value',[('held',True),('unresolved',1),('unattempted',1),('completed',4)])
def test_held_or_unknown_corrected_ids_do_not_open_readiness(tmp_path,monkeypatch,field,value):
    _,review,manifest,terminal,_=setup(tmp_path,monkeypatch)
    changed=chain.document(terminal);changed['sources']['lex_glue'][field]=value
    spec=write_json(tmp_path/'bad-terminal.json',changed)
    with pytest.raises(ValueError):completed.terminal_binding(spec,manifest,review,chain.document(manifest))


def test_source_report_must_bind_actual_continuation_decision(tmp_path,monkeypatch):
    _,review,manifest,terminal,_=setup(tmp_path,monkeypatch)
    changed=chain.document(terminal);item=changed['sources']['lex_glue']['chunks'][0]
    report=json.loads(Path(item['report_path']).read_text());report['decision_sha256']='f'*64
    item['report_sha256']=write_json(Path(item['report_path']),report)['sha256']
    spec=write_json(tmp_path/'bad-terminal.json',changed)
    with pytest.raises(ValueError,match='different reviewed decision'):
        completed.terminal_binding(spec,manifest,review,chain.document(manifest))


def test_partial_selection_does_not_publish_last_marker(tmp_path,monkeypatch):
    policy,review,manifest,terminal,base=setup(tmp_path,monkeypatch)
    task_path=Path(chain.document(manifest)['sources'][0]['chunks'][0]['path'])
    with task_path.open('ab') as handle:handle.write(b' ')
    with pytest.raises(ValueError):
        completed.freeze_selection(policy['path'],policy['sha256'],review,manifest,terminal,base,tmp_path/'partial')
    assert not (tmp_path/'partial/selection.json').exists()


def test_candidate_passes_frozen_consumer_before_published_and_is_idempotent(tmp_path):
    policy,spec,base,dependency,ready=corrected_ready_fixture(tmp_path)
    candidate=tmp_path/'candidate.json'
    ready_spec=completed.publish_readiness(policy,spec['sha256'],base,dependency,ready,candidate)
    assert ready_spec['path']==dependency['readiness_path']
    assert completed.publish_readiness(policy,spec['sha256'],base,dependency,ready,candidate)==ready_spec
    assert chain.ready_additional_components(policy,spec['sha256'],base)[0]['readiness']==ready_spec


def test_invalid_candidate_never_publishes_watched_ready_path(tmp_path):
    policy,spec,base,dependency,ready=corrected_ready_fixture(tmp_path)
    ready['training_quality_code_sha256']='0'*64
    with pytest.raises(ValueError):
        completed.publish_readiness(policy,spec['sha256'],base,dependency,ready,tmp_path/'candidate.json')
    assert not Path(dependency['readiness_path']).exists()


def test_transport_payload_uses_exact_messages_and_native_audit(tmp_path):
    from test_expansion_trace_audit import fixture
    from data.expansion_trace_audit import audit_trace
    trace,tokenizer,prep,_=fixture()
    row={'task_id':trace['task_id'],'source':'lex_glue','messages':json.dumps(trace['messages']),
        'tools':json.dumps(trace['tools']),'provenance':json.dumps({'trace':{
            k:v for k,v in trace.items() if k not in ('messages','tools')}})}
    recovered,source,_=completed.transport_payload(row)
    assert audit_trace(recovered,source,tokenizer,prep)['accepted']==1
    row['source']='unreviewed'
    with pytest.raises(ValueError):completed.transport_payload(row)
