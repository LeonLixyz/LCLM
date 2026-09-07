from data.expansion_task_normalization import prepare_teacher_task

def test_maud_choices_are_in_both_prompts_without_changing_gold_or_source():
    task={'family':'maud','source_dataset':'theatticusproject/maud',
        'question':'Use these source documents: contract-1.\nType of Consideration-Answer\nReturn exactly FINAL: followed by your answer.',
        'gold_answer':'All Cash','segments':[{'segment_id':'seg_1','record_id':'contract-1:part0',
            'text':'Exact clause\nwith table | columns','summary':'contract-1: consideration clause'}]}
    result=prepare_teacher_task(task,{'Type of Consideration-Answer':{'All Cash','All Stock'}})
    for field in ('training_user_prompt','rollout_user_prompt'):
        assert 'All Cash' in result[field] and 'All Stock' in result[field]
    assert result['gold_answer']==task['gold_answer']
    assert result['segments'][0]['text']=='SOURCE contract-1:part0\n'+task['segments'][0]['text']
    assert task['segments'][0]['text'].startswith('Exact clause')
    assert 'Choose exactly one' in result['question']

def test_regular_task_source_identity_is_not_teacher_only():
    task={'family':'finqa','source_dataset':'bevaya/FinQA','question':'What is the total?',
        'segments':[{'segment_id':'seg_1','record_id':'report-1','text':'table','summary':'report-1 financial table'}]}
    result=prepare_teacher_task(task)
    assert 'report-1' in result['training_user_prompt']
    assert 'report-1' in result['rollout_user_prompt']
    assert result['segments'][0]['text']=='SOURCE report-1\ntable'


def test_acord_uses_beir_scale_in_both_prompts_preserving_gold():
    task={'family':'acord','source_dataset':'theatticusproject/acord',
        'question':'Return the integer relevance grade from 1 (irrelevant) to 5 (highly relevant).',
        'gold_answer':'0','segments':[{'segment_id':'seg_1','record_id':'clause1',
            'text':'clause','summary':'clause1'}]}
    result=prepare_teacher_task(task)
    assert result['gold_answer']=='0'
    for key in ('question','training_user_prompt','rollout_user_prompt'):
        assert 'from 0 (irrelevant) to 4 (highly relevant)' in result[key]
