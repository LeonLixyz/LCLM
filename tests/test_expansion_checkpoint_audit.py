import json
import pytest
from data.expansion_checkpoint_audit import audit_source


def setup_files(tmp_path):
    task = tmp_path / 'source.tasks.jsonl'
    task.write_text(json.dumps({'task_id':'one'})+'\n'+json.dumps({'task_id':'two'})+'\n')
    row = {'task_id':'one', 'verification':{'accepted':True,'reason':'accepted:exact'},
           'messages':[{'role':'assistant','content':'FINAL: answer'}]}
    path = tmp_path / 'source.accepted.jsonl'
    path.write_text(json.dumps(row)+'\n')
    return task, path, row


def test_valid_partial_checkpoint(tmp_path):
    task, _, _ = setup_files(tmp_path)
    result = audit_source(task,tmp_path,'source')
    assert (result['persisted'],result['remaining'],result['accepted']) == (1,1,1)


@pytest.mark.parametrize('failure',['duplicate','truncated','unknown','wrong_verdict','false_complete'])
def test_bad_checkpoint_fails_closed(tmp_path,failure):
    task,path,row=setup_files(tmp_path)
    if failure=='duplicate':path.write_text(path.read_text()*2)
    elif failure=='truncated':path.write_text(path.read_text().rstrip('\n'))
    elif failure in ('unknown','wrong_verdict'):
        if failure=='unknown':row['task_id']='other'
        else:row['verification']['accepted']=False
        path.write_text(json.dumps(row)+'\n')
    else:
        (tmp_path/'source.generation.json').write_text(json.dumps({
            'status':'complete','reasons':{'accepted:exact':1}}))
    with pytest.raises(ValueError):audit_source(task,tmp_path,'source')
