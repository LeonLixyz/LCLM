import copy
import json
from pathlib import Path
import pytest
from data import sglang_lexglue_selected as selected
from data.lexglue_task_definition_correction import CONFIGS,amend_lexglue_task
from data.sglang_backlog import sha,file_sha
from test_lexglue_task_definition_correction import fixture_task


FIVE=['case_hold','ecthr_a','ledgar','scotus','unfair_tos']


@pytest.mark.parametrize('raw', ['', '{"counts":', '[]', '{"counts":null}'])
def test_partial_live_progress_is_optional(tmp_path,raw):
    path=tmp_path/'progress.json';path.write_text(raw)
    assert selected.optional_progress(path) is None
    path.write_text('{"counts":{"completed":64}}')
    assert selected.optional_progress(path)=={'completed':64}


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value)+'\n')
    return {'path':str(path),'sha256':file_sha(path)}


def setup(tmp_path,monkeypatch):
    root=tmp_path/'parent';root.mkdir();chunks=[];probe_ids=[];totals={c:0 for c in CONFIGS}
    for index,configs in enumerate([[c for c in CONFIGS for _ in range(10 if c=='case_hold' else 9)],list(CONFIGS),list(CONFIGS)]):
        rows=[]
        for i,config in enumerate(configs):
            row=amend_lexglue_task(fixture_task(config));row.update(task_id=f'task-{index}-{i}',source_row_id=f'{config}-{index*100+i}')
            rows.append(row);totals[config]+=1
        if index==0:probe_ids=[r['task_id'] for r in rows]
        path=root/f'chunk-{index}.jsonl';path.write_bytes(b''.join((json.dumps(row,ensure_ascii=False)+'  \n').encode() for row in rows))
        chunks.append({'index':index,'path':str(path),'rows':len(rows),'sha256':file_sha(path),
            'task_ids_sha256':sha(json.dumps(sorted(r['task_id'] for r in rows)).encode())})
    ontology=write(root/'ontology.json',{'fixture':'definitions'})
    monkeypatch.setattr(selected,'ONTOLOGY_PATH',ontology['path']);monkeypatch.setattr(selected,'DEFINITIONS_SHA256',ontology['sha256'])
    parent=write(root/'manifest.json',{'task_version':selected.VERSION,'ontology_artifact':ontology,
        'generation_config':selected.parent_config(),'original_pending_rows':78,'configs':totals,
        'maud_ontology_path':ontology['path'],'maud_ontology_sha256':ontology['sha256'],
        'sources':[{'source':'lex_glue','rows':14,'probe_rows':64,'probe_ids':probe_ids,'chunks':chunks}]})
    monkeypatch.setattr(selected,'PARENT_MANIFEST_SHA256',parent['sha256'])
    probe_decision=write(root/'probe-decision.json',{'backlog_manifest_sha256':parent['sha256']})
    monkeypatch.setattr(selected,'PARENT_PROBE_DECISION_SHA256',probe_decision['sha256'])
    report=write(root/'probe-report.json',{'status':'complete','rows':64,'completed':64,'source':'lex_glue','chunk':0,
        'backlog_manifest_sha256':parent['sha256'],'decision_sha256':probe_decision['sha256'],
        'input_sha256':chunks[0]['sha256'],'result_hashes':{key:'a'*64 for key in probe_ids}})
    audit=write(root/'audit.json',{'status':'passed','errors':[],'task_version':selected.VERSION,
        'ontology_artifact':ontology,'probe_decision_sha256':probe_decision['sha256'],'input_sha256':chunks[0]['sha256'],
        'exact_original_to_corrected_transforms':64,'exact_corrected_teacher_requests_verified':True})
    decision={'schema':selected.SELECTION_SCHEMA,'decision':'select_corrected_lexglue_configs','reviewed_by':'root',
        'approved_for_preparation':True,'approved_for_generation':True,'approved_for_release':False,
        'scope':'exact_unattempted_config_subset','parent_manifest':parent,'generation_config':selected.generation_config(),
        'config_allowlist':FIVE,'config_reviews':{c:{'approved_for_generation':c in FIVE,
            'evidence_review':'Fixture explicit source decision; no inference from automatic yield.'} for c in CONFIGS},
        'corrected_probe_decision':probe_decision,'corrected_probe_report':report,'corrected_probe_audit':audit,
        'review_artifacts':[write(root/'semantic-review.json',{'reviewed_by':'root','fixture':True})]}
    spec=write(tmp_path/'config-decision.json',decision)
    return spec,decision,selected.document(parent)


def prepare(tmp_path,monkeypatch):
    selection,decision,parent=setup(tmp_path,monkeypatch);reports=[]
    for spec in parent['sources'][0]['chunks']:
        root=tmp_path/'prepared'/str(spec['index'])
        selected.prepare_chunk(spec,root,config_allowlist=FIVE,canary_ids=parent['sources'][0]['probe_ids'],selection_spec=selection)
        reports.append({'path':str(root/'report.json'),'sha256':file_sha(root/'report.json')})
    manifest=selected.finalize_subset(parent,selection,decision,reports,tmp_path/'accounting')
    return selection,decision,parent,reports,manifest


def test_same_serving_prompt_validator_settings():
    config=selected.generation_config();selected.verify_settings(config)
    config['concurrency']=1
    with pytest.raises(ValueError):selected.verify_settings(config)


def test_five_config_subset_has_exact_bytes_and_complete_disposition_accounting(tmp_path,monkeypatch):
    selection,decision,parent,reports,manifest=prepare(tmp_path,monkeypatch)
    assert selected.validate_selection(selection)[0]==decision
    assert manifest['pending_rows']==10 and manifest['original_pending_rows']==78
    assert manifest['plan_accounting']['held_config_unattempted']==4
    assert manifest['plan_accounting']['diagnostic_canaries']==64
    assert manifest['held_configs']==['ecthr_b','eurlex']
    assert manifest['sources'][0]['probe_rows']==0 and len(manifest['sources'][0]['chunks'])==3
    expected={row['task_id']:line for spec in parent['sources'][0]['chunks'] for row,line in selected.exact_rows(spec)}
    canaries=set(parent['sources'][0]['probe_ids']);seen=set()
    for spec in manifest['sources'][0]['chunks'][1:]:
        for row,line in selected.exact_rows(spec):
            assert row['task_id'] not in canaries and row['task_id'] not in seen
            assert line==expected[row['task_id']];seen.add(row['task_id'])
    assert len(seen)==10
    assert selected.finalize_subset(parent,selection,decision,reports,tmp_path/'accounting')==manifest


@pytest.mark.parametrize('change',[{'approved_for_preparation':False},{'approved_for_generation':False},
    {'config_allowlist':[]},{'config_allowlist':['unknown']},{'config_allowlist':['case_hold','case_hold']}])
def test_root_config_choice_cannot_be_inferred_or_widened(tmp_path,monkeypatch,change):
    _,decision,_=setup(tmp_path,monkeypatch);decision.update(change)
    with pytest.raises(ValueError):selected.validate_selection(write(tmp_path/'bad.json',decision))


def test_held_config_cannot_be_implicitly_approved(tmp_path,monkeypatch):
    _,decision,_=setup(tmp_path,monkeypatch)
    decision['config_reviews']['eurlex']['approved_for_generation']=True
    with pytest.raises(ValueError):selected.validate_selection(write(tmp_path/'bad.json',decision))


def test_probe_audit_and_all64_result_ids_must_be_complete(tmp_path,monkeypatch):
    _,decision,_=setup(tmp_path,monkeypatch)
    report=selected.document(decision['corrected_probe_report']);report['result_hashes'].pop(next(iter(report['result_hashes'])))
    decision['corrected_probe_report']=write(tmp_path/'bad-report.json',report)
    with pytest.raises(ValueError):selected.validate_selection(write(tmp_path/'bad.json',decision))


def test_partial_or_changed_input_does_not_publish_subset_report(tmp_path,monkeypatch):
    selection,decision,parent=setup(tmp_path,monkeypatch);spec=parent['sources'][0]['chunks'][1]
    with Path(spec['path']).open('ab') as handle:handle.write(b' ')
    with pytest.raises(ValueError):selected.prepare_chunk(spec,tmp_path/'bad',config_allowlist=FIVE,
        canary_ids=parent['sources'][0]['probe_ids'],selection_spec=selection)
    assert not (tmp_path/'bad/report.json').exists()


def test_completed_preparation_rejects_changed_task_or_selection_bytes(tmp_path,monkeypatch):
    selection,decision,parent=setup(tmp_path,monkeypatch);spec=parent['sources'][0]['chunks'][1]
    kwargs=dict(config_allowlist=FIVE,canary_ids=parent['sources'][0]['probe_ids'],selection_spec=selection)
    report=selected.prepare_chunk(spec,tmp_path/'prepared',**kwargs)
    assert selected.prepare_chunk(spec,tmp_path/'prepared',**kwargs)==report
    with Path(report['tasks']['path']).open('ab') as handle:handle.write(b' ')
    with pytest.raises(ValueError):selected.prepare_chunk(spec,tmp_path/'prepared',**kwargs)


def test_duplicate_global_parent_ids_are_rejected_before_manifest(tmp_path,monkeypatch):
    selection,decision,parent,reports,_=prepare(tmp_path,monkeypatch)
    broken=copy.deepcopy(parent);broken['sources'][0]['chunks'][2]=broken['sources'][0]['chunks'][1]
    with pytest.raises(ValueError):selected.finalize_subset(broken,selection,decision,[reports[0],reports[1],reports[1]],tmp_path/'bad')


def test_continuation_requires_exact_config_manifest_binding(tmp_path,monkeypatch):
    selection,_,_,_,manifest=prepare(tmp_path,monkeypatch)
    decision={'decision':'continue_corrected_lexglue_configs','reviewed_by':'root','approved_for_generation':True,
        'approved_for_release':False,'scope':'reviewed_config_subset','backlog_manifest_sha256':'c'*64,
        'config_selection_sha256':selection['sha256'],'config_allowlist':FIVE,
        'generation_config':manifest['generation_config'],'task_version':selected.VERSION,
        'ontology_artifact':manifest['ontology_artifact'],'evidence_review':'Exact prepared subset of reviewed configs.'}
    assert selected.validate_continuation(decision,manifest,manifest_sha='c'*64,selection_sha=selection['sha256'])=={'lex_glue'}
    decision['config_allowlist']=['case_hold']
    with pytest.raises(ValueError):selected.validate_continuation(decision,manifest,manifest_sha='c'*64,selection_sha=selection['sha256'])
