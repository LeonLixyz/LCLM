import json
from pathlib import Path
import pytest
from data import lexglue_export_recovery as recovery
from data import lexglue_selected_completion as completed
from data import export_reviewed_sglang27 as core
from test_lexglue_selected_completion import setup
from test_export_reviewed_sglang27 import fixture, write_json


def test_retry_owner_allows_only_same_logical_input_and_binding():
    binding={'selection':'fixed'}
    saved={'identity':{'function_call_id':'fc-a','input_id':'in-b:1-0'},'binding':binding}
    assert recovery.same_owner(saved,{'function_call_id':'fc-a','input_id':'in-b:2-0'},binding)
    assert not recovery.same_owner(saved,{'function_call_id':'fc-c','input_id':'in-b:2-0'},binding)
    assert not recovery.same_owner(saved,{'function_call_id':'fc-a','input_id':'in-c:2-0'},binding)
    assert not recovery.same_owner(saved,{'function_call_id':'fc-a','input_id':'in-b:2-0'},{})


def test_parallel_selection_matches_original_every_field(tmp_path,monkeypatch):
    policy,review,manifest,terminal,base=setup(tmp_path,monkeypatch)
    original=completed.freeze_selection(policy['path'],policy['sha256'],review,manifest,terminal,base,tmp_path/'serial-selected')
    recovered=recovery.selection_function()(policy['path'],policy['sha256'],review,manifest,terminal,base,tmp_path/'recovered')
    # Paths differ by destination; content, all counts, exclusion IDs and ID digest must match.
    def normalized(path):
        value=json.loads(Path(path).read_text())
        text=json.dumps(value,sort_keys=True).replace(str(tmp_path/'serial-selected'),'DEST').replace(str(tmp_path/'recovered'),'DEST')
        return json.loads(text)
    a=normalized(original['selection']['path']);b=normalized(recovered['selection']['path'])
    assert a['sources']==b['sources'] and a['excluded_task_ids']==b['excluded_task_ids']
    assert a['diagnostic_canary']==b['diagnostic_canary'] and a['generation_manifest']==b['generation_manifest']
    assert (tmp_path/'serial-selected/training-exclusions.jsonl').read_bytes()==(tmp_path/'recovered/training-exclusions.jsonl').read_bytes()
    qa=normalized(original['quality_accounting']['path']);qb=normalized(recovered['quality_accounting']['path'])
    assert qa==qb


def test_parallel_export_is_byte_identical_and_preserves_all_checks(tmp_path):
    selected=fixture(tmp_path)
    spec=write_json(tmp_path/'selection.json',selected)
    kw={'reviewed_sources':['fixture'],'excluded_task_ids':['known-bad'],'shard_rows':1}
    a=core.export_reviewed(spec['path'],spec['sha256'],tmp_path/'serial',**kw)
    b=recovery.export_function()(spec['path'],spec['sha256'],tmp_path/'parallel',**kw)
    assert a['files']==b['files'] and a['source_counts']==b['source_counts'] and a['task_ids_sha256']==b['task_ids_sha256']
    for x in a['files']:
        assert (tmp_path/'serial'/x['name']).read_bytes()==(tmp_path/'parallel'/x['name']).read_bytes()


@pytest.mark.parametrize('target',['raw','result'])
def test_parallel_reads_reject_tampered_rejected_task_too(tmp_path,target):
    selected=fixture(tmp_path);spec=write_json(tmp_path/'selection.json',selected)
    path=tmp_path/('raw/task-2/attempt-01/raw.json' if target=='raw' else 'results/task-2.json')
    path.write_bytes(path.read_bytes()+b' ')
    with pytest.raises(ValueError,match='Frozen file hash changed'):
        recovery.export_function()(spec['path'],spec['sha256'],tmp_path/'bad',
            reviewed_sources=['fixture'],excluded_task_ids=['known-bad'])
    assert not (tmp_path/'bad/manifest.json').exists()


def test_changed_original_body_is_not_silently_transformed(monkeypatch):
    monkeypatch.setitem(recovery.ORIGINAL_HASHES,'export_reviewed_sglang27.py','0'*64)
    with pytest.raises(ValueError,match='Original audited code changed'):
        recovery.export_function()


def test_recovery_binding_preserves_completed_selection_and_resumes_idempotently(tmp_path):
    spec=write_json(tmp_path/'selection.json',fixture(tmp_path))
    original=Path(spec['path']).read_bytes()
    evidence=write_json(tmp_path/'recovery.json',{'reviewed_by':'root'})
    bound=recovery.bind_selection_recovery(spec,evidence)
    assert Path(spec['path']).read_bytes()==original
    assert json.loads(Path(bound['path']).read_text())['execution_recovery']==evidence
    assert recovery.bind_selection_recovery(spec,evidence)==bound
    assert recovery.bind_selection_recovery(bound,evidence)==bound
    with pytest.raises(ValueError,match='Different execution recovery'):
        recovery.bind_selection_recovery(bound,{'path':evidence['path'],'sha256':'0'*64})
