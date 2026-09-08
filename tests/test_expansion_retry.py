import json
import pytest
from data.prepare_expansion_retry import prepare_source


def setup(tmp_path):
    old = tmp_path/'old'; old.mkdir()
    task = tmp_path/'source.tasks.jsonl'
    task.write_text(''.join(json.dumps({'task_id': str(i), 'question': 'untouched'})+'\n' for i in range(3)))
    for kind, i in [('accepted', 0), ('rejected', 1)]:
        (old/('source.'+kind+'.jsonl')).write_text(json.dumps({'task_id': str(i),
            'verification': {'accepted': kind=='accepted', 'reason': kind+':test'},
            'messages': [{'role':'assistant', 'content':'answer'}]})+'\n')
    return task, old, tmp_path/'prepared'


def test_partition_and_preserve(tmp_path):
    task, old, out = setup(tmp_path)
    before = {p.name: p.read_bytes() for p in old.iterdir()}
    report = prepare_source('source', task, old, out)
    rows = [json.loads(x) for x in (out/'source.tasks.jsonl').read_text().splitlines()]
    assert [r['task_id'] for r in rows] == ['1', '2']
    assert [r['_attempt_provenance']['kind'] for r in rows] == ['retry_rejected', 'previously_unattempted']
    assert all(r['question']=='untouched' for r in rows)
    assert report['counts']=={'previously_accepted':1, 'retry_rejected':1, 'previously_unattempted':1}
    assert before == {p.name: p.read_bytes() for p in old.iterdir()}
    assert prepare_source('source', task, old, out)==report


@pytest.mark.parametrize('change', ['parent', 'prepared', 'parent_new_file'])
def test_changed_inputs_fail_closed(tmp_path, change):
    task, old, out = setup(tmp_path)
    if change=='parent_new_file':
        (old/'source.accepted.jsonl').unlink()
    prepare_source('source', task, old, out)
    path = (old/'source.rejected.jsonl') if change=='parent' else (
        old/'source.accepted.jsonl' if change=='parent_new_file' else out/'source.tasks.jsonl')
    path.write_text('')
    with pytest.raises(ValueError):
        prepare_source('source', task, old, out)


def test_full_maud_ontology_not_just_rejected_labels(tmp_path):
    task, old, out = setup(tmp_path)
    task.write_text(''.join(json.dumps({'task_id':str(i), 'question':'Field-Answer',
        'gold_answer':str(i)})+'\n' for i in range(3)))
    for path in list(old.iterdir()):
        path.rename(old/path.name.replace('source.', 'maud.'))
    prepare_source('maud', task, old, out)
    assert json.loads((out/'maud-choices.json').read_text())=={'Field-Answer':['0','1','2']}


def test_truncated_checkpoint_rejected(tmp_path):
    task, old, out = setup(tmp_path)
    (old/'source.rejected.jsonl').write_text('{')
    with pytest.raises(ValueError, match='Incomplete JSONL'):
        prepare_source('source', task, old, out)


def test_pubmed_retry_only_official_training_ids():
    from data.full_expansion_rollouts import validate_pubmed_ids
    validate_pubmed_ids({'train1'}, {'train1','train2'}, retry_subset=True)
    with pytest.raises(RuntimeError):
        validate_pubmed_ids({'heldout'}, {'train1','train2'}, retry_subset=True)
    with pytest.raises(RuntimeError):
        validate_pubmed_ids({'train1'}, {'train1','train2'})
