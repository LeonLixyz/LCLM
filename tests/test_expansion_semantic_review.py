import copy
import json
import pytest
from data.expansion_semantic_review import judge_messages, review_one
from data.expansion_semantic_review import check_summary_claims, apply_semantic_review

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

@pytest.mark.parametrize('quote,supported,keep', [('actual source', True, True),
    ('invented source', True, False), ('actual source', False, False), ('', True, False)])
def test_claim_quotes_checked_and_all_sentences_included(quote, supported, keep):
    row = {'messages': [{'role':'tool','content':'actual source\nRELATED SOURCE other:\npadding'},
                        {'role':'assistant','content':'FINAL: First sentence. Second sentence.'}]}
    prompts = []
    def complete(messages):
        prompts.append(json.loads(messages[1]['content']))
        return json.dumps({'supported':supported, 'evidence_quotes':[quote] if quote else []})
    result = check_summary_claims(row, complete)
    assert result['keep'] is keep
    assert [p['sentence'] for p in prompts] == ['First sentence.', 'Second sentence.']
    assert all(p['source']=='actual source' for p in prompts)

@pytest.mark.parametrize('source,reason', [('clapnq','missing_support'),
    ('billsum','no_tool_call'), ('pubmedqa_labeled','wrong_answer:pubmedqa_decision'),
    ('finqa','wrong_answer:finqa_numeric')])
def test_semantic_judge_cannot_override_structural_or_exact_failures(source, reason):
    trace = {'verification':{'accepted':False,'reason':reason}}
    def complete(messages):
        pytest.fail('Judge must not be called')
    assert apply_semantic_review(source, trace, complete) == trace['verification']

def test_summary_claim_failure_overrides_two_positive_votes():
    trace = {'task_id':'example','task':'Summarize','gold_answer':'A',
             'verification':{'accepted':True,'reason':'accepted:summary'},
             'messages':[{'role':'tool','content':'Evidence'},
                         {'role':'assistant','content':'FINAL: Unsupported claim.'}]}
    answers = iter(['{"correct":true,"grounded":true}']*2 +
                   ['{"supported":false,"evidence_quotes":[]}'])
    verdict = apply_semantic_review('billsum', trace, lambda messages: next(answers))
    assert verdict['accepted'] is False
    assert trace['verification']['accepted'] is True
