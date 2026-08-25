from data.real_expansion_agent import (
    MAX_SEGMENT_WORDS,
    MIN_SEGMENT_WORDS,
    build_real_segment,
    make_real_expansion_task,
    validate_real_task,
    verify_real_trace,
)
from data.harvest_real_expansion import harvest_real_traces


def _document(index, words=600):
    return {
        "document_id": f"doc-{index}",
        "title": f"Document {index}",
        "text": " ".join(f"real{index}_{word}" for word in range(words)),
    }


def test_real_segment_uses_companion_source_text_to_reach_floor():
    primary = _document(1, 100)
    companion = _document(2, 600)
    segment = build_real_segment(primary, [companion])
    assert MIN_SEGMENT_WORDS <= len(segment.split()) <= MAX_SEGMENT_WORDS
    assert "real1_50" in segment
    assert "real2_50" in segment


def test_real_task_has_long_seg_i_memory_blocks_and_one_support():
    task = make_real_expansion_task(
        source_dataset="example/qa",
        source_row_id="row-7",
        question="What is the recorded value?",
        gold_answer="42",
        support_document=_document(1),
        distractor_documents=[_document(2), _document(3)],
        companion_documents=[_document(4), _document(5)],
        source_category="finance",
        seed=9,
    )
    validate_real_task(task)
    assert len(task["segments"]) == 3
    assert len(task["support_segment_ids"]) == 1
    assert task["expected_final"] == "FINAL: 42"
    assert task["training_user_prompt"].count("<|memory_start|>") == 3
    assert "42" not in task["rollout_user_prompt"]


def test_task_id_is_stable_but_segment_order_depends_on_seed():
    kwargs = dict(
        source_dataset="example/qa",
        source_row_id="same-row",
        question="Question?",
        gold_answer="answer",
        support_document=_document(1),
        distractor_documents=[_document(i) for i in range(2, 8)],
        companion_documents=[_document(8)],
        source_category="legal",
    )
    first = make_real_expansion_task(**kwargs, seed=1)
    second = make_real_expansion_task(**kwargs, seed=2)
    assert first["task_id"] == second["task_id"]
    assert [x["record_id"] for x in first["segments"]] != [
        x["record_id"] for x in second["segments"]
    ]


def _messages(task, final):
    segment_id = task["support_segment_ids"][0]
    segment_text = next(
        (segment["text"] for segment in task.get("segments", []) if segment["segment_id"] == segment_id),
        "x",
    )
    return [
        {"role": "assistant", "content": "", "tool_calls": [{
            "id": "call-1",
            "type": "function",
            "function": {"name": "expand", "arguments": {"segment_id": segment_id}},
        }]},
        {"role": "tool", "tool_call_id": "call-1", "name": "expand", "content": segment_text},
        {"role": "assistant", "content": final},
    ]


def _metric_task(source, gold):
    return {
        "source_dataset": source,
        "gold_answer": gold,
        "expected_final": f"FINAL: {gold}",
        "support_segment_ids": ["seg_1"],
        "segments": [{"segment_id": "seg_1", "text": "x"}],
    }


def test_real_verifier_accepts_native_class_and_numeric_metrics():
    contract = _metric_task("stanfordnlp/contract-nli", "NotMentioned")
    assert verify_real_trace(contract, _messages(contract, "FINAL: not mention"))["accepted"]
    pubmed = _metric_task("qiaojin/PubMedQA:pqa_labeled", "yes")
    assert verify_real_trace(pubmed, _messages(pubmed, "FINAL: Yes, the study supports it."))["accepted"]
    finqa = _metric_task("bevaya/FinQA", "6.0%")
    assert verify_real_trace(finqa, _messages(finqa, "Calculation...\nFINAL: 6.00%"))["accepted"]


def test_real_verifier_accepts_grounded_short_answer_contained_in_reference():
    task = _metric_task(
        "PrimeQA/clapnq", "Patty and Selma Bouvier are Marge Simpson's older twin sisters."
    )
    result = verify_real_trace(task, _messages(task, "FINAL: Patty and Selma"))
    assert result["accepted"]


def test_real_verifier_rejects_wrong_numeric_answer_and_missing_support():
    task = _metric_task("bevaya/FinQA", "17%")
    assert not verify_real_trace(task, _messages(task, "FINAL: 18.3%"))["accepted"]
    missing = [{"role": "assistant", "content": "FINAL: 17%"}]
    assert verify_real_trace(task, missing)["reason"] == "no_tool_call"


def test_real_harvest_separates_accepted_and_rejected():
    task = make_real_expansion_task(
        source_dataset="bevaya/FinQA",
        source_row_id="row-1",
        question="What is the percentage?",
        gold_answer="6.0%",
        support_document=_document(1),
        distractor_documents=[_document(2)],
        companion_documents=[_document(3)],
        source_category="finance",
        seed=3,
    )
    good = {"task_id": task["task_id"], "messages": _messages(task, "FINAL: 6.00%")}
    accepted, rejected, report = harvest_real_traces([task], [good])
    assert len(accepted) == 1
    assert not rejected
    assert report["accepted_by_source"] == {"bevaya/FinQA": 1}
