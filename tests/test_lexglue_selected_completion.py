import copy
import json
from pathlib import Path
import pytest
from data import lexglue_selected_completion as completed
from data import expansion_completion_chain as chain
from data import export_reviewed_sglang27 as core
import test_lexglue_completion as original
from test_export_reviewed_sglang27 import write_json,id_digest


def setup(tmp_path,monkeypatch):
    # Parent64 proof and selection are independently covered by test_sglang_lexglue_selected.
    # This fixture isolates the strict subset export/readiness contract with five attempted rows.
    monkeypatch.setattr(original,'completed',completed)
    policy,review,manifest,terminal,base=original.setup(tmp_path,monkeypatch)
    m=chain.document(manifest);s=m['sources'][0]
    s['chunks'][0].update(rows=0,exclude_task_ids=['probe'],task_ids_sha256=core.sha(b'[]'))
    s['probe_rows']=0;m['pending_rows']=5;m['config_allowlist']=['scotus']
    config_selection=write_json(tmp_path/'config-choice.json',{'config_allowlist':['scotus']})
    m['config_selection']=config_selection
    m['plan_accounting']={'selected_unattempted':5,'held_config_unattempted':0,'diagnostic_canaries':1}
    m['parent_manifest']=write_json(tmp_path/'parent.json',{'fixture':'All parent bytes independently tested'})
    m['diagnostic_canary']={'rows':1,'included_in_training':False,'included_in_generation':False,
        'manifest':m['parent_manifest']}
    manifest=write_json(Path(manifest['path']),m)
    d=chain.document(review);d.update(decision='continue_corrected_lexglue_configs',
        generation_manifest_sha256=manifest['sha256'],backlog_manifest_sha256=manifest['sha256'],
        config_selection_sha256=config_selection['sha256'],config_allowlist=['scotus'])
    review=write_json(Path(review['path']),d)
    t=chain.document(terminal);t.update(manifest_sha256=manifest['sha256'],decision_sha256=review['sha256'],
        config_allowlist=m['config_allowlist'],config_selection=config_selection,
        plan_accounting=m['plan_accounting'],diagnostic_canary=m['diagnostic_canary'])
    item=t['sources']['lex_glue']['chunks'][0]
    r=json.loads(Path(item['report_path']).read_text())
    r.update(backlog_manifest_sha256=manifest['sha256'],decision_sha256=review['sha256'])
    item['report_sha256']=write_json(Path(item['report_path']),r)['sha256']
    terminal=write_json(Path(terminal['path']),t)
    monkeypatch.setattr(completed,'validate_selection',lambda _:({'config_allowlist':['scotus']},{}))
    return policy,review,manifest,terminal,base


def test_canary_diagnostics_never_enter_subset_export(tmp_path,monkeypatch):
    policy,review,manifest,terminal,base=setup(tmp_path,monkeypatch)
    result=completed.freeze_selection(policy['path'],policy['sha256'],review,manifest,terminal,base,tmp_path/'selection')
    selection=chain.document(result['selection'])
    assert selection['sources'][0]['counts']=={'results':5,'accepted':4,'excluded_accepted':3,'exported':1}
    assert selection['sources'][0]['task_ids_sha256']==id_digest(['good'])
    assert len(selection['sources'][0]['chunks'])==1
    assert selection['diagnostic_canary']['included_in_training'] is False
    transport=core.export_reviewed(result['selection']['path'],result['selection']['sha256'],tmp_path/'transport',
        reviewed_sources=['lex_glue'],excluded_task_ids=selection['excluded_task_ids'])
    assert transport['rows']==1


@pytest.mark.parametrize('field,value',[('config_allowlist',['scotus','eurlex']),('plan_accounting',{}),('diagnostic_canary',{})])
def test_terminal_must_preserve_explicit_held_and_canary_accounting(tmp_path,monkeypatch,field,value):
    _,review,manifest,terminal,_=setup(tmp_path,monkeypatch)
    changed=chain.document(terminal);changed[field]=value
    terminal=write_json(tmp_path/'bad-terminal.json',changed)
    with pytest.raises(ValueError):completed.terminal_binding(terminal,manifest,review,chain.document(manifest))


def test_changed_config_selection_cannot_reuse_source_review(tmp_path,monkeypatch):
    policy,review,manifest,_,_=setup(tmp_path,monkeypatch)
    value=chain.document(review);value['config_allowlist']=['scotus','eurlex']
    review=write_json(tmp_path/'bad-review.json',value)
    with pytest.raises(ValueError):completed.governing(policy['path'],policy['sha256'],review,manifest)


def test_canary_row_cannot_be_smuggled_into_continuation(tmp_path,monkeypatch):
    policy,review,manifest,terminal,base=setup(tmp_path,monkeypatch)
    m=chain.document(manifest);m['sources'][0]['chunks'][0]['exclude_task_ids'].append('good')
    manifest=write_json(Path(manifest['path']),m)
    d=chain.document(review);d.update(generation_manifest_sha256=manifest['sha256'],backlog_manifest_sha256=manifest['sha256'])
    review=write_json(Path(review['path']),d)
    t=chain.document(terminal);t.update(manifest_sha256=manifest['sha256'],decision_sha256=review['sha256'])
    item=t['sources']['lex_glue']['chunks'][0];r=json.loads(Path(item['report_path']).read_text())
    r.update(backlog_manifest_sha256=manifest['sha256'],decision_sha256=review['sha256'])
    item['report_sha256']=write_json(Path(item['report_path']),r)['sha256']
    terminal=write_json(Path(terminal['path']),t)
    with pytest.raises(ValueError,match='Diagnostic canary'):
        completed.freeze_selection(policy['path'],policy['sha256'],review,manifest,terminal,base,tmp_path/'bad-selection')


def test_subset_readiness_is_accepted_by_frozen_main_consumer(tmp_path):
    from test_expansion_completion_chain import corrected_ready_fixture
    policy,spec,base,dependency,ready=corrected_ready_fixture(tmp_path)
    ready['config_allowlist']=['case_hold','ecthr_a','ledgar','scotus','unfair_tos']
    ready['plan_accounting']={'selected_unattempted':123421,'held_config_unattempted':62959,'diagnostic_canaries':64}
    ready['diagnostic_canary']={'included_in_generation':False,'included_in_training':False,'rows':64}
    result=completed.publish_readiness(policy,spec['sha256'],base,dependency,ready,tmp_path/'candidate.json')
    assert chain.ready_additional_components(policy,spec['sha256'],base)[0]['readiness']==result
