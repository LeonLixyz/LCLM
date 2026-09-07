from copy import deepcopy
import pytest
from data.techqa_adapter import convert_training

def inputs():
    return ([{'QUESTION_ID':'TRAIN_Q000','QUESTION_TITLE':'Question','QUESTION_TEXT':'Details',
              'DOCUMENT':'doc1','ANSWERABLE':'Y','ANSWER':'answer','START_OFFSET':'2','END_OFFSET':'8'}],
            {'doc1':{'title':'Title','text':'x answer z'}},[{'DOCUMENT':'dev'}],[{'DOCUMENT':'test'}])

def test_exact_training_span_and_original_document_preserved():
    tasks,documents,counts,heldout=convert_training(*inputs())
    assert tasks[0]['answer']=='answer'
    assert tasks[0]['question']=='Question\n\nDetails'
    assert documents[0]['text']=='x answer z'
    assert counts['accepted']==1 and heldout==['dev','test']

@pytest.mark.parametrize('failure',['overlap','answer','unanswerable'])
def test_bad_training_rows_excluded_before_document_pool(failure):
    training,documents,dev,validation=deepcopy(inputs())
    if failure=='overlap':dev[0]['DOCUMENT']='doc1'
    elif failure=='answer':training[0]['ANSWER']='different'
    else:training[0]['ANSWERABLE']='N'
    tasks,pool,counts,_=convert_training(training,documents,dev,validation)
    assert not tasks and not pool

def test_unknown_holdout_schema_fails_closed():
    training,documents,dev,validation=inputs()
    with pytest.raises(ValueError):convert_training(training,documents,dev,[{'unknown':'field'}])

def test_dev_row_in_training_file_fails_closed():
    training,documents,dev,validation=inputs()
    training[0]['QUESTION_ID']='DEV_Q000'
    with pytest.raises(ValueError):convert_training(training,documents,dev,validation)
