import pytest
from data.multidoc2dial_dialogue import chronological_question


def example():
    dialogue = {"dial_id": "d_1", "turns": [
        {"turn_id": 1, "role": "user", "utterance": "Earlier || literal [SEP] text", "da": "query"},
        {"turn_id": 2, "role": "agent", "utterance": "Prior\nanswer", "da": "respond"},
        {"turn_id": 3, "role": "user", "utterance": "Current?", "da": "query"},
        {"turn_id": 4, "role": "agent", "utterance": "Final answer", "da": "respond"}]}
    row = {"id": "d_1_3", "utterance": "Final answer",
           "question": "Current?[SEP]agent: Prior answer||user: Earlier || literal [SEP] text"}
    return row, dialogue


def test_chronological_and_literal_delimiters():
    row, dialogue = example()
    result = chronological_question(row, dialogue)
    assert result.index("user: Earlier || literal [SEP] text") < result.index("agent: Prior answer")
    assert result.index("agent: Prior answer") < result.index("Current user request:\nCurrent?")
    assert "Final answer" not in result


def test_first_turn_and_columnar_dialogue():
    row, dialogue = example()
    row.update(id="d_1_1", utterance="Prior\nanswer",
               question="Earlier || literal [SEP] text[SEP]")
    dialogue["turns"] = {k: [t[k] for t in dialogue["turns"]] for k in dialogue["turns"][0]}
    assert "(No prior turns.)" in chronological_question(row, dialogue)


@pytest.mark.parametrize("field,value", [("question", "different"), ("utterance", "different"), ("id", "unknown")])
def test_reject_mismatched_source(field, value):
    row, dialogue = example(); row[field] = value
    with pytest.raises(ValueError):
        chronological_question(row, dialogue)
