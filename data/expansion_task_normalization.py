"""Task-side normalization, identical for teacher input and saved training input."""
import copy
import json

def maud_field(task):
    lines=task['question'].splitlines()
    return lines[1] if lines[0].startswith('Use these source documents:') else lines[0]

def prepare_teacher_task(task,maud_choices=None):
    from data.synthetic_expansion_agent import format_training_user_prompt,format_rollout_user_prompt
    task=copy.deepcopy(task)
    if task.get('family')=='acord':
        # Upstream qrels use BEIR scores 0..4, not the paper's 1..5 stars.
        # https://huggingface.co/datasets/theatticusproject/acord
        task['question']=task['question'].replace(
            'from 1 (irrelevant) to 5 (highly relevant)',
            'from 0 (irrelevant) to 4 (highly relevant)')
    if task.get('family')=='maud':
        field=maud_field(task)
        choices=(maud_choices or {}).get(field)
        if not choices:raise ValueError('Missing MAUD answer ontology')
        title=field.removesuffix('-Answer')
        question='Identify '+title+' in the specified source. Choose exactly one answer label from: '+json.dumps(sorted(choices))+'.'
        task['question']=task['question'].replace(field,question,1)
    if task.get('family')!='synthetic' and task.get('source_dataset'):
        for segment in task['segments']:
            header='SOURCE '+segment['record_id']+'\n'
            if not segment['text'].startswith(header):segment['text']=header+segment['text']
    task['training_user_prompt']=format_training_user_prompt(task['segments'],task['question'])
    task['user_prompt']=task['training_user_prompt']
    task['rollout_user_prompt']=format_rollout_user_prompt(task['segments'],task['question'])
    task['task_normalization']='source-identity-maud-ontology-acord-beir-v2'
    return task
