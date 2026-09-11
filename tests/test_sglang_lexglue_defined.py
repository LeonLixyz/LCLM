import copy
import json
from pathlib import Path
import pytest
from data.sglang_backlog import sha,file_sha
from data.sglang_lexglue_defined import generation_config,verify_settings,prepare_chunk,validate_decision
from data.lexglue_task_definition_correction import CONFIGS,VERSION,DEFINITIONS_SHA256
from data.sglang_backlog_fast import checked_rows
from test_lexglue_task_definition_correction import fixture_task


def spec_fixture(tmp_path):
    rows=[]
    for i,config in enumerate(CONFIGS):
        row=fixture_task(config);row['task_id']=f'task-{i}';rows.append(row)
    path=tmp_path/'original.jsonl'
    path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    return rows,{'path':str(path),'sha256':file_sha(path),'rows':len(rows),'index':0,'kind':'probe',
        'task_ids_sha256':sha(json.dumps(sorted(r['task_id'] for r in rows)).encode())}


def test_only_definition_snapshot_configuration_changes():
    config=generation_config();verify_settings(config)
    config['concurrency']=1
    with pytest.raises(ValueError,match='Serving'):verify_settings(config)


def test_frozen_chunks_change_prompts_not_ids_evidence_or_original_bytes(tmp_path):
    rows,spec=spec_fixture(tmp_path);old=Path(spec['path']).read_bytes()
    report=prepare_chunk(spec,tmp_path/'new')
    assert report['rows']==7 and report['modified']==6 and report['configs']=={c:1 for c in CONFIGS}
    assert report['chunk']['task_ids_sha256']==spec['task_ids_sha256']
    corrected=list(checked_rows(report['chunk']))
    assert Path(spec['path']).read_bytes()==old
    for before,after in zip(rows,corrected):
        for key in ('task_id','source_row_id','gold_answer','expected_final','segments','tools','_attempt_provenance'):
            assert before[key]==after[key]
    assert corrected[0]==rows[0]
    assert all(t['task_definition_correction']['version']==VERSION for t in corrected[1:])
    assert prepare_chunk(spec,tmp_path/'new')==report


def test_changed_original_or_completed_corrected_bytes_fail(tmp_path):
    rows,spec=spec_fixture(tmp_path)
    report=prepare_chunk(spec,tmp_path/'new')
    with Path(report['chunk']['path']).open('ab') as handle:handle.write(b' ')
    with pytest.raises(ValueError):prepare_chunk(spec,tmp_path/'new')
    with Path(spec['path']).open('ab') as handle:handle.write(b' ')
    with pytest.raises(ValueError):prepare_chunk(spec,tmp_path/'other')
    assert not (tmp_path/'other/report.json').exists()


def test_physical_chunk_exclusions_still_preserve_effective_ids(tmp_path):
    rows,spec=spec_fixture(tmp_path)
    spec.update(rows=6,exclude_task_ids=['task-0'],task_ids_sha256=sha(json.dumps([f'task-{i}' for i in range(1,7)]).encode()))
    report=prepare_chunk(spec,tmp_path/'new')
    assert report['rows']==report['modified']==6
    assert 'exclude_task_ids' not in report['chunk']
    assert [t['task_id'] for t in checked_rows(report['chunk'])]==[f'task-{i}' for i in range(1,7)]


def gate_fixture():
    manifest={'generation_config':generation_config(),'ontology_artifact':{'path':'/runs/definitions.json','sha256':DEFINITIONS_SHA256},
        'sources':[{'source':'lex_glue','chunks':[{'sha256':'corrected-input'}]}]}
    decision={'decision':'generate_corrected_lexglue_probe','reviewed_by':'root','approved_for_generation':True,
        'approved_for_release':False,'backlog_manifest_sha256':'manifest','generation_config':manifest['generation_config'],
        'task_version':VERSION,'ontology_artifact':manifest['ontology_artifact'],'evidence_review':'Reviewed primary definitions and raw failure.',
        'scope':'source_probes_only','probe_sources':['lex_glue']}
    return manifest,decision


def test_probe_requires_root_review_of_actual_corrected_version():
    manifest,decision=gate_fixture()
    assert validate_decision(decision,manifest,stage='probe',manifest_sha='manifest')=={'lex_glue'}
    for changed in [{'task_version':'opaque-old'},{'backlog_manifest_sha256':'old'},
                    {'probe_sources':['lex_glue','finqa']},{'approved_for_generation':False}]:
        with pytest.raises(ValueError):validate_decision({**decision,**changed},manifest,stage='probe',manifest_sha='manifest')


def test_continuation_requires_exact_completed_corrected64_probe():
    manifest,decision=gate_fixture()
    decision.update(decision='continue_corrected_lexglue',scope='reviewed_sources',probe_decision_sha256='probe-decision',
        source_reviews={'lex_glue':{'approved_for_generation':True,'evidence_review':'Corrected config review.',
                                  'probe_report_sha256':'probe-report'}})
    report={'source':'lex_glue','chunk':0,'status':'complete','rows':64,'completed':64,
        'backlog_manifest_sha256':'manifest','decision_sha256':'probe-decision','input_sha256':'corrected-input'}
    def check(value):return validate_decision(decision,manifest,stage='continuation',manifest_sha='manifest',
        probe_decision_sha='probe-decision',probe_report=value,probe_report_sha='probe-report')
    assert check(report)=={'lex_glue'}
    for changed in [{'completed':63},{'input_sha256':'old-opaque-input'},{'decision_sha256':'old-probe-decision'}]:
        with pytest.raises(ValueError):check({**report,**changed})
