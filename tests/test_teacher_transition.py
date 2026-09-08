import json
import pytest
from data.prepare_teacher_transition import SOURCES, prepare_remaining


def fixture(tmp_path):
    tasks=tmp_path/'tasks'; old=tmp_path/'old'; new=tmp_path/'new'
    tasks.mkdir();old.mkdir()
    for s in SOURCES:
        (tasks/(s+'.tasks.jsonl')).write_text(''.join(json.dumps({'task_id':s+str(i)})+'\n' for i in range(3)))
    (old/'billsum.rejected.jsonl').write_text(json.dumps({'task_id':'billsum0',
        'verification':{'accepted':False,'reason':'wrong_answer:test'}})+'\n')
    return tasks,old,new


def test_remaining_partition_and_idempotency(tmp_path):
    paths=fixture(tmp_path); report=prepare_remaining(*paths)
    assert [s['remaining'] for s in report['sources']]==[2,3,3]
    assert report['approved_for_release'] is False
    assert prepare_remaining(*paths)==report
    assert 'billsum0' not in (paths[2]/'billsum.tasks.jsonl').read_text()


def test_changed_parent_is_rejected(tmp_path):
    paths=fixture(tmp_path);prepare_remaining(*paths)
    (paths[1]/'billsum.rejected.jsonl').write_text('')
    with pytest.raises(ValueError,match='Previous teacher'):
        prepare_remaining(*paths)


def test_changed_delta_is_rejected(tmp_path):
    paths=fixture(tmp_path);prepare_remaining(*paths)
    (paths[2]/'billsum.tasks.jsonl').write_text('')
    with pytest.raises(ValueError,match='Changed delta'):
        prepare_remaining(*paths)


def test_new_parent_file_is_rejected(tmp_path):
    paths=fixture(tmp_path);prepare_remaining(*paths)
    (paths[1]/'lex_glue.accepted.jsonl').write_text('')
    with pytest.raises(ValueError,match='Previous teacher file set'):
        prepare_remaining(*paths)


def test_changed_input_roots_rejected(tmp_path):
    paths=fixture(tmp_path);prepare_remaining(*paths)
    with pytest.raises(ValueError,match='Changed transition input roots'):
        prepare_remaining(tmp_path/'other',paths[1],paths[2])


def test_truncated_checkpoint_fails_closed(tmp_path):
    paths=fixture(tmp_path)
    (paths[1]/'billsum.rejected.jsonl').write_text('{')
    with pytest.raises(ValueError,match='Incomplete JSONL'):
        prepare_remaining(*paths)


def test_new_teacher_manifest_and_resume_guard(tmp_path):
    from data.full_expansion_rollouts import generate_all
    revision='1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0'
    config={'model_id':'Qwen/Qwen3.8-27B','revision':revision,'sampling_profile':'recommended'}
    (tmp_path/'synthetic.tasks.jsonl').write_text('')
    (tmp_path/'synthetic.build.json').write_text('{}')
    report=generate_all(None,config['model_id'],revision,tmp_path,lambda:None,
        sources=['synthetic'],teacher_config=config)
    assert report['manifest']['model']==config['model_id']
    assert report['manifest']['teacher_sampling']['temperature']==0.7
    assert report['manifest']['requires_new_source_review'] is True
    assert report['manifest']['teacher_sampling']['extra_body']['chat_template_kwargs']['enable_thinking'] is False
    with pytest.raises(RuntimeError,match='Incompatible resume'):
        generate_all(None,'old','old-revision',tmp_path,lambda:None,sources=['synthetic'])


@pytest.mark.parametrize('change', [{'revision':'other'}, {'sampling_profile':'thinking'}])
def test_unknown_teacher_configuration_rejected(tmp_path,change):
    from data.full_expansion_rollouts import generate_all
    config={'model_id':'Qwen/Qwen3.8-27B','revision':'1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0',
            'sampling_profile':'recommended',**change}
    with pytest.raises(ValueError,match='Unrecognized teacher'):
        generate_all(None,config['model_id'],config['revision'],tmp_path,lambda:None,
                     sources=['synthetic'],teacher_config=config)
