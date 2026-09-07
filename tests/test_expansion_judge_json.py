import pytest
from data.full_expansion_rollouts import parse_judge_json


@pytest.mark.parametrize('wrapper',['{}','```json\n{}\n```','```\n{}\n```'])
def test_accepts_complete_boolean_json_with_optional_fence(wrapper):
    assert parse_judge_json(wrapper.format('{"correct":true,"grounded":false}'))=={
        'correct':True,'grounded':False}


@pytest.mark.parametrize('content',['{"correct":"true","grounded":true}',
    '{"correct":true}', '[true,true]', 'Note: {"correct":true,"grounded":true}'])
def test_rejects_ambiguous_or_nonboolean_votes(content):
    with pytest.raises(ValueError):parse_judge_json(content)
