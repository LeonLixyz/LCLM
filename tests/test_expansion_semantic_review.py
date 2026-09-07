import copy
import json
import pytest
from data.expansion_semantic_review import judge_messages, review_one

def test_blind_prompt_has_no_reference_and_uses_data_fields():
    messages = judge_messages("question", "answer", "evidence")
    assert json.loads(messages[1]["content"]) == {
        "question": "question", "answer": "answer", "evidence": "evidence"}
    assert "untrusted data" in messages[0]["content"]

@pytest.mark.parametrize("votes,keep", [
    ([{"correct": True, "grounded": True}] * 2, True),
    ([{"correct": False, "grounded": True}, {"correct": True, "grounded": True}], False),
    ([{"correct": True, "grounded": True}, {"correct": True, "grounded": False}], False),
])
def test_both_votes_required_and_training_row_unchanged(votes, keep):
    row = {"task_id": "example", "task": "Question?", "gold_answer": "Reference",
           "messages": [{"role": "tool", "content": "Evidence"},
                        {"role": "assistant", "content": "FINAL: Answer"}]}
    original = copy.deepcopy(row)
    requests = []
    def complete(messages):
        requests.append(messages)
        return json.dumps(votes[len(requests) - 1])
    assert review_one(row, complete)["keep"] is keep
    assert row == original
    assert "reference" not in json.loads(requests[0][1]["content"])
    assert json.loads(requests[1][1]["content"])["reference"] == "Reference"

def test_malformed_vote_fails_closed():
    row = {"task_id": "example", "task": "Q", "gold_answer": "A",
           "messages": [{"role": "tool", "content": "E"},
                        {"role": "assistant", "content": "FINAL: A"}]}
    with pytest.raises(ValueError):
        review_one(row, lambda messages: '{"correct":"true","grounded":true}')
