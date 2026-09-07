from copy import deepcopy
import pytest
from data.techqa_adapter import convert_training, build_expansion_task

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


def expansion_inputs():
    text='answer\n\n'+'Evidence  '*1000
    documents={'train-doc':{'document_id':'train-doc','title':'Support','text':text},
        'companion':{'document_id':'companion','title':'Other training source','text':'Other facts. '*1000}}
    row={'task_id':'TRAIN_1','document_id':'train-doc','question':'What is the result?',
        'answer':'answer','start_offset':0,'end_offset':6,'source_split':'training_Q_A.json'}
    return row,documents


def test_techqa_expansion_keeps_full_source_and_answer_span():
    row,documents=expansion_inputs()
    task=build_expansion_task(row,documents)
    assert len(task['support_segment_ids'])==2
    assert task['source_evidence_span']=={'document_id':'train-doc','start':0,'end':6}
    assert task['gold_answer']=='answer'
    assert all(len(s['text'].split())>=512 for s in task['segments'])
    assert 'train-doc' in task['question']


@pytest.mark.parametrize('failure',['split','offset'])
def test_techqa_expansion_revalidates_training_receipt(failure):
    row,documents=expansion_inputs()
    if failure=='split':row['source_split']='dev_Q_A.json'
    else:row['end_offset']=5
    with pytest.raises(ValueError):build_expansion_task(row,documents)
