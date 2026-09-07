import pytest
from data.pubmedqa_split import REVISION, training_ids

def manifest():
    return {'revision':REVISION,'fold':0,'train_ids':[str(i) for i in range(450)],
            'dev_ids':[str(i) for i in range(450,500)],'test_ids':[str(i) for i in range(500,1000)]}

def test_train_allowlist_excludes_dev_and_test():
    value = manifest()
    assert len(training_ids(value)) == 450
    assert not training_ids(value) & set(value['dev_ids']+value['test_ids'])

@pytest.mark.parametrize('change', ['overlap','duplicate','revision','fold'])
def test_invalid_split_fails_closed(change):
    value = manifest()
    if change == 'overlap': value['train_ids'][0] = value['test_ids'][0]
    elif change == 'duplicate': value['train_ids'][0] = value['train_ids'][1]
    elif change == 'revision': value['revision'] = 'unknown'
    else: value['fold'] = 1
    with pytest.raises(ValueError): training_ids(value)
