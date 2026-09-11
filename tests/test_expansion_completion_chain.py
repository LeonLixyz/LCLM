import copy
import json
from pathlib import Path
import pytest
from data import expansion_completion_chain as chain
from data import export_reviewed_sglang27 as core
from data import reviewed_expansion_adapters as adapters
from test_export_reviewed_sglang27 import fixture as sg_fixture,write_json,id_digest
from test_reviewed_expansion_adapters import legacy,pilot


def setup(tmp_path):
    generation_root=tmp_path/'generation'
    sources=[];chunks=[]
    def repeat(key,result,records):
        if key in ('repeat','both'):
            msg=copy.deepcopy(result['trace']['messages'][2]);msg['tool_calls'][0]['id']='call-2'
            response=copy.deepcopy(result['trace']['messages'][3]);response['tool_call_id']='call-2'
            result['trace']['messages'][4:4]=[msg,response]
    for index,rows in enumerate([[('probe',True)],[('good',True),('repeat',True),('both',True),('rejected',False)]]):
        directory=generation_root/'outputs'/'fixture'/f'chunk-{index:05d}'
        selected=sg_fixture(directory,rows=rows,mutate=repeat)
        (directory/'raw').rename(directory/'attempts')
        spec=selected['sources'][0]['chunks'][0]['tasks']
        spec.update(index=index,task_ids_sha256=core.sha(json.dumps(sorted(k for k,_ in rows)).encode()))
        chunks.append(spec)
    source={'source':'fixture','rows':4,'probe_rows':1,'probe_ids':['probe'],'chunks':chunks}
    choices=write_json(tmp_path/'choices.json',{})
    origin=write_json(tmp_path/'original/manifest.json',{'fixture':True})
    manifest=write_json(generation_root/'manifest.json',{'pending_rows':272287,'sources':[source],
        'origin_probe_manifest_path':origin['path'],'maud_ontology_path':choices['path'],'maud_ontology_sha256':choices['sha256']})
    resume=write_json(generation_root/'resume.json',{'backlog_manifest_sha256':manifest['sha256'],'reused_probe_reports':{}})
    probe=generation_root/'outputs/fixture/chunk-00000/report.json'
    decision=write_json(generation_root/'decision.json',{'backlog_manifest_sha256':manifest['sha256'],
        'resume_probe_decision_sha256':resume['sha256'],'approved_for_generation':True,'reviewed_by':'root',
        'source_reviews':{'fixture':{'probe_report_sha256':chain.file_sha(probe)}}})
    report_path=generation_root/'outputs/fixture/chunk-00001/report.json'
    report=json.loads(report_path.read_text());report.update(backlog_manifest_sha256=manifest['sha256'],decision_sha256=decision['sha256'])
    report_spec=write_json(report_path,report)
    terminal=write_json(tmp_path/'terminal.json',{'successor_terminal_status':'terminal','stage':'continuation',
        'manifest_sha256':manifest['sha256'],'decision_sha256':decision['sha256'],'sources':{'fixture':{
            'approved_for_stage':True,'held':False,'unresolved':0,'unattempted':0,'completed':4,'expected_rows':4,
            'chunks':[{'index':1,'status':'complete','completed':4,'unresolved':0,
                'input_sha256':chunks[1]['sha256'],'effective_task_ids_sha256':chunks[1]['task_ids_sha256'],
                'report_path':report_spec['path'],'report_sha256':report_spec['sha256']}]}}})
    manual=write_json(tmp_path/'manual.json',{'reviewed_by':'root','count':2,'entries':[{'task_id':'both'},{'task_id':'legacy-bad'}]})
    review=write_json(tmp_path/'review.json',{'reviewed_by':'root','evidence':'Fixture source review'})
    base={'root':str(tmp_path/'base'),'manifest_sha256':{'16384':'a'*64,'32768':'b'*64},
          'summary':write_json(tmp_path/'base/summary.json',{'fixture':True})}
    policy={'schema':chain.SCHEMA,'reviewed_by':'root','approved_for_internal_packing':True,'approved_for_release':False,
        'configuration':chain.config(),'reviewed_sources':['fixture'],'sglang27_sources':['fixture'],
        'source_reviews':{'fixture':{'approved_for_internal_packing':True,'evidence_review':'Fixture checked','artifacts':[review]}},
        'approval_artifacts':[review],'manual_exclusions':manual,'append_partitions':2,'base':{'root':chain.BASE_ROOT},
        'generation':{'initial_call_id':'fc-fixture','app':'lclm-sglang27-continuation-storage-20260909-v3',
            'manifest':manifest,'continuation_decision':decision,'resume_probe_decision':resume},'retained_streams':[]}
    return policy,terminal,base


def test_policy_requires_actual_generation_root_review_and_code_binding(tmp_path):
    policy,_,_=setup(tmp_path)
    spec=write_json(tmp_path/'policy.json',policy)
    assert chain.load_policy(spec['path'],spec['sha256'])[0]==policy
    for mutation in [lambda p:p.update(approved_for_internal_packing=False),
                     lambda p:p['configuration'].update(cpu_only=False),
                     lambda p:p['source_reviews']['fixture'].update(evidence_review=''),
                     lambda p:p['sglang27_sources'].append('unreviewed')]:
        changed=copy.deepcopy(policy);mutation(changed);spec=write_json(tmp_path/'changed.json',changed)
        with pytest.raises(ValueError):chain.load_policy(spec['path'],spec['sha256'])


@pytest.mark.parametrize('field,value',[('held',True),('unresolved',1),('unattempted',1),('completed',3)])
def test_selected_source_cannot_freeze_with_held_or_unknown_attempts(tmp_path,field,value):
    policy,spec,_=setup(tmp_path);terminal=chain.document(spec)
    terminal['sources']['fixture'][field]=value;spec=write_json(tmp_path/'bad-terminal.json',terminal)
    with pytest.raises(ValueError):chain.validate_terminal(policy,spec)


def test_terminal_rejects_missing_chunk_or_changed_report_bytes(tmp_path):
    policy,spec,_=setup(tmp_path);terminal=chain.document(spec)
    report=terminal['sources']['fixture']['chunks'][0]['report_path']
    with Path(report).open('ab') as handle:handle.write(b' ')
    with pytest.raises(ValueError):chain.validate_terminal(policy,spec)


def test_selection_freezes_actual_ids_exclusion_union_and_unchanged_verdicts(tmp_path):
    policy,terminal,base=setup(tmp_path);spec=write_json(tmp_path/'policy.json',policy)
    original={p:chain.file_sha(p) for p in (tmp_path/'generation').rglob('*.json')}
    result=chain.freeze_sglang_selection(spec['path'],spec['sha256'],terminal,base,tmp_path/'selection-stage')
    selection=chain.document(result['selection'])
    assert selection['sources'][0]['counts']=={'results':5,'accepted':4,'excluded_accepted':2,'exported':2}
    assert selection['sources'][0]['task_ids_sha256']==id_digest(['probe','good'])
    counts=chain.document(result['quality_accounting'])['counts']
    assert counts=={'accepted':4,'manual':1,'dynamic':2,'overlap':1,'excluded_union':2,'eligible':2}
    assert set(selection['excluded_task_ids'])=={'both','repeat','legacy-bad'}
    assert {p:chain.file_sha(p) for p in original}==original
    exported=core.export_reviewed(result['selection']['path'],result['selection']['sha256'],tmp_path/'strict-transport',
        reviewed_sources=['fixture'],excluded_task_ids=selection['excluded_task_ids'])
    assert exported['rows']==2


def test_all_three_teachers_export_once_with_manual_and_policy_counts(tmp_path):
    policy,terminal,base=setup(tmp_path)
    old=legacy(tmp_path/'old',keys=('legacy-good','legacy-bad'))
    pil=pilot(tmp_path/'pilot')
    policy['retained_streams']=[old,pil]
    spec=write_json(tmp_path/'policy.json',policy)
    sg=chain.freeze_sglang_selection(spec['path'],spec['sha256'],terminal,base,tmp_path/'sg-selection')
    selection=chain.document(sg['selection'])
    transport=core.export_reviewed(sg['selection']['path'],sg['selection']['sha256'],tmp_path/'core-export',
        reviewed_sources=['fixture'],excluded_task_ids=selection['excluded_task_ids'])
    transport_spec={'path':str(tmp_path/'core-export/manifest.json'),'sha256':chain.file_sha(tmp_path/'core-export/manifest.json')}
    frozen=chain.freeze_union_selection(spec['path'],spec['sha256'],sg['selection'],transport_spec,base,tmp_path/'union-selection')
    union=chain.document(frozen['selection'])
    assert union['task_ids_sha256']==id_digest(['legacy-good','pilot-1','probe','good'])
    assert union['source_counts']['fixture']=={'results':6,'accepted':5,'excluded_accepted':1,'exported':4}
    assert chain.document(frozen['quality_accounting'])['counts']['manual']==1
    manifest=adapters.export_union(frozen['selection']['path'],frozen['selection']['sha256'],tmp_path/'union-export',
        reviewed_sources=['fixture'],excluded_task_ids=union['excluded_task_ids'])
    assert manifest['rows']==4 and not manifest['approved_for_release']
    assert not (tmp_path/'union-export/components').exists()


def test_duplicate_exported_id_across_teachers_fails_in_selection(tmp_path):
    policy,terminal,base=setup(tmp_path)
    policy['retained_streams']=[legacy(tmp_path/'old',keys=('good',))]
    spec=write_json(tmp_path/'policy.json',policy)
    sg=chain.freeze_sglang_selection(spec['path'],spec['sha256'],terminal,base,tmp_path/'sg-selection')
    selection=chain.document(sg['selection'])
    core.export_reviewed(sg['selection']['path'],sg['selection']['sha256'],tmp_path/'core-export',
        reviewed_sources=['fixture'],excluded_task_ids=selection['excluded_task_ids'])
    transport={'path':str(tmp_path/'core-export/manifest.json'),'sha256':chain.file_sha(tmp_path/'core-export/manifest.json')}
    with pytest.raises(ValueError,match='Duplicate exported task'):
        chain.freeze_union_selection(spec['path'],spec['sha256'],sg['selection'],transport,base,tmp_path/'union-selection')


def test_old_lexglue_never_enters_original_teacher_streams(tmp_path):
    policy,_,_=setup(tmp_path)
    policy['sglang27_sources']=['lex_glue']
    spec=write_json(tmp_path/'policy.json',policy)
    with pytest.raises(ValueError,match='opaque-label LexGLUE'):
        chain.load_policy(spec['path'],spec['sha256'])


def corrected_ready_fixture(tmp_path):
    policy,_,base=setup(tmp_path)
    ontology=write_json(tmp_path/'ontology.json',{'definitions':'Primary source fixture'})
    dependency={'stream_id':'corrected_lex','source_allowlist':['lex_glue'],'task_version':'defined-labels-v1',
        'ontology_artifact':ontology,'readiness_path':str(tmp_path/'additional-ready.json')}
    policy['reviewed_sources'].append('lex_glue');policy['additional_readiness_dependencies']=[dependency]
    spec=write_json(tmp_path/'policy.json',policy)
    generation=write_json(tmp_path/'corrected-manifest.json',{'task_version':dependency['task_version'],'ontology_artifact':ontology})
    review=write_json(tmp_path/'corrected-source-review.json',{'reviewed_by':'root','approved_for_internal_packing':True,
        'approved_for_release':False,'task_version':dependency['task_version'],'ontology_artifact':ontology,
        'source_allowlist':['lex_glue'],'generation_manifest_sha256':generation['sha256']})
    terminal=write_json(tmp_path/'corrected-terminal.json',{'manifest_sha256':generation['sha256'],
        'successor_terminal_status':'terminal','sources':{'lex_glue':{'approved_for_stage':True,'held':False,
            'unresolved':0,'unattempted':0,'completed':2,'expected_rows':2}}})
    quality=write_json(tmp_path/'corrected-quality.json',{'code_sha256':chain.file_sha(Path(chain.__file__).parent/'expansion_training_quality.py'),
        'counts':{'accepted':2,'eligible':2,'excluded_union':0,'manual':0,'dynamic':0,'overlap':0}})
    selection=write_json(tmp_path/'corrected-selection.json',{'generation_manifest':generation,'generation_terminal':terminal,
        'reviewed_sources':['lex_glue'],'base_snapshot':base,'training_quality_accounting':quality})
    transport=write_json(tmp_path/'corrected-transport.json',{'base_snapshot':base,
        'selection_sha256':selection['sha256'],'status':'reviewed_complete'})
    audit=write_json(tmp_path/'corrected-format-audit.json',{'status':'passed','selection_sha256':selection['sha256'],
        'transport_manifest_sha256':transport['sha256'],'failed_rows':0})
    ready={'schema':'reviewed-corrected-expansion-ready-v1','status':'complete','chain_policy_sha256':spec['sha256'],
        'task_version':dependency['task_version'],'ontology_artifact':ontology,'source_allowlist':['lex_glue'],
        'training_quality_code_sha256':chain.document(quality)['code_sha256'],
        'manual_exclusions_sha256':policy['manual_exclusions']['sha256'],'source_review':review,
        'generation_manifest':generation,'generation_terminal':terminal,'selection':selection,'transport':transport,'format_audit':audit}
    return policy,spec,base,dependency,ready


def test_corrected_dependency_waits_then_freezes_real_artifact_hashes(tmp_path):
    policy,spec,base,dependency,ready=corrected_ready_fixture(tmp_path)
    assert chain.load_policy(spec['path'],spec['sha256'])[0]==policy
    assert chain.ready_additional_components(policy,spec['sha256'],base) is None
    ready_spec=write_json(Path(dependency['readiness_path']),ready)
    component,=chain.ready_additional_components(policy,spec['sha256'],base)
    assert component['readiness']==ready_spec and component['source_allowlist']==['lex_glue']


@pytest.mark.parametrize('field,value',[('task_version','opaque-old'),('training_quality_code_sha256','0'*64),
                                       ('manual_exclusions_sha256','0'*64),('chain_policy_sha256','0'*64)])
def test_corrected_dependency_rejects_stale_policy_or_task_version(tmp_path,field,value):
    policy,spec,base,dependency,ready=corrected_ready_fixture(tmp_path)
    ready[field]=value;write_json(Path(dependency['readiness_path']),ready)
    with pytest.raises(ValueError):chain.ready_additional_components(policy,spec['sha256'],base)
