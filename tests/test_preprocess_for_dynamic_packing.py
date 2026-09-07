import json
import pytest

import data.preprocess_for_dynamic_packing as preprocessing


class FakeQwenTokenizer:
    chat_template = "fake-qwen"

    def __init__(self):
        self.pad_token_id = 0
        self.eos_token = "<eos>"
        self.calls = []
        self.special_ids = {
            "<|memory_start|>": 101,
            "<|memory|>": 102,
            "<|memory_end|>": 103,
        }
        self._char_ids = {}

    def add_special_tokens(self, value):
        return len(value.get("additional_special_tokens", []))

    def convert_tokens_to_ids(self, token):
        return self.special_ids[token]

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        ids = []
        cursor = 0
        markers = sorted(self.special_ids, key=len, reverse=True)
        while cursor < len(text):
            marker = next(
                (candidate for candidate in markers if text.startswith(candidate, cursor)),
                None,
            )
            if marker is not None:
                ids.append(self.special_ids[marker])
                cursor += len(marker)
                continue
            char = text[cursor]
            if char not in self._char_ids:
                self._char_ids[char] = 1000 + len(self._char_ids)
            ids.append(self._char_ids[char])
            cursor += 1
        return ids

    @staticmethod
    def _render(messages, tools=None, add_generation_prompt=False):
        text = ""
        if tools:
            text += "<|im_start|>system\n<tools>" + json.dumps(tools, sort_keys=True)
            text += "</tools><|im_end|>\n"
        for message in messages:
            role = message["role"]
            content = message.get("content")
            content = content if isinstance(content, str) else ""
            text += f"<|im_start|>{role}\n{content}"
            for call in message.get("tool_calls", []):
                text += "<tool_call>" + json.dumps(call, sort_keys=True) + "</tool_call>"
            text += "<|im_end|>\n"
        if add_generation_prompt:
            text += "<|im_start|>assistant\n"
        return text

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize,
        add_generation_prompt,
        tools=None,
        **kwargs,
    ):
        del kwargs
        self.calls.append({"messages": messages, "tools": tools})
        rendered = self._render(messages, tools, add_generation_prompt)
        return self.encode(rendered) if tokenize else rendered


def _install_worker_tokenizer(monkeypatch):
    tokenizer = FakeQwenTokenizer()
    monkeypatch.setattr(preprocessing, "_worker_tokenizer", tokenizer)
    monkeypatch.setattr(preprocessing, "_worker_embed_tokenizer", None)
    monkeypatch.setattr(preprocessing, "_worker_memory_start_id", 101)
    monkeypatch.setattr(preprocessing, "_worker_memory_id", 102)
    monkeypatch.setattr(preprocessing, "_worker_memory_end_id", 103)
    return tokenizer


class FastNewlineTokenizer(FakeQwenTokenizer):
    is_fast = True

    def __call__(self, text, **kwargs):
        ids, offsets = [], []
        cursor = 0
        while cursor < len(text):
            end = cursor + (2 if text[cursor:cursor+2] == '\n\n' else 1)
            ids.append(999999 if end-cursor == 2 else 1000+ord(text[cursor]))
            offsets.append((cursor, end))
            cursor = end
        return {'input_ids': ids, 'offset_mapping': offsets}

    def encode(self, text, **kwargs):
        return self(text)['input_ids']


def test_sft_recovers_newline_boundary_and_masks_merged_token():
    tokenizer = FastNewlineTokenizer()
    row = {'compression_prompt': [{'role': 'user', 'content': 'question'}],
           'target': '\nanswer', '_recover_prefix_only': True}
    ids, labels, memories, boundary = preprocessing.process_sft_example(row, tokenizer)
    assert memories == []
    assert ids[boundary-1] == 999999
    assert all(x == -100 for x in labels[:boundary])
    assert ''.join(chr(x-1000) for x in labels[boundary:]) == 'answer<|im_end|>\n'
    del row['_recover_prefix_only']
    assert preprocessing.process_sft_example(row, tokenizer)[1] == labels


def test_sft_recovery_does_not_duplicate_already_valid_rows():
    row = {'compression_prompt': [{'role': 'user', 'content': 'question'}],
           'target': 'answer', '_recover_prefix_only': True}
    assert preprocessing.process_sft_example(row, FastNewlineTokenizer()) is None


def test_legacy_baseline_retry_keeps_recovery_rows_separate():
    row = {'compression_prompt': [{'role': 'user', 'content': 'question'}],
           'target': '\nanswer', '_legacy_sft_prefix_strict': True}
    assert preprocessing.process_sft_example(row, FastNewlineTokenizer()) is None
    row['target'] = 'answer'
    assert preprocessing.process_sft_example(row, FastNewlineTokenizer()) is not None


def test_sft_rejects_inconsistent_offset_tokenization():
    class BadOffsets(FastNewlineTokenizer):
        def __call__(self, text, **kwargs):
            result = super().__call__(text, **kwargs)
            if kwargs.get('return_offsets_mapping'):
                result['input_ids'][0] += 1
            return result
    row = {'compression_prompt': [{'role': 'user', 'content': 'q'}], 'target': '\na'}
    with pytest.raises(ValueError, match='offset tokenization'):
        preprocessing.process_sft_example(row, BadOffsets())


def test_native_json_transport_matches_object_input(monkeypatch):
    _install_worker_tokenizer(monkeypatch)
    row={'messages':[{'role':'user','content':'do it'},
        {'role':'assistant','content':'','tool_calls':[{'type':'function','function':{'name':'run','arguments':{'x':1}}}]}],
        'tools':[{'type':'function','function':{'name':'run','parameters':{'type':'object'}}}]}
    ordinary=preprocessing.worker_process_example((row,16,'compression_prompt',None,None))
    transport={k:json.dumps(v) for k,v in row.items()}
    decoded=preprocessing.worker_process_example((transport,16,'compression_prompt',None,None))
    assert ordinary is not None
    assert decoded==ordinary


def test_sft_target_cot_is_compacted_and_masked():
    tokenizer = FakeQwenTokenizer()
    target = "<|memory_start|>hidden reasoning<|memory_end|>FINAL"
    result = preprocessing.process_sft_example(
        {
            "compression_prompt": [{"role": "user", "content": "problem"}],
            "target": target,
        },
        tokenizer,
    )

    input_ids, labels, memories, prompt_len = result
    assert memories == ["hidden reasoning"]
    start = input_ids.index(101)
    assert input_ids[start : start + 3] == [101, 102, 103]
    assert labels[start : start + 3] == [-100, -100, -100]
    final_ids = tokenizer.encode("FINAL")
    final_start = next(
        index
        for index in range(len(input_ids) - len(final_ids) + 1)
        if input_ids[index : index + len(final_ids)] == final_ids
    )
    assert labels[final_start : final_start + len(final_ids)] == final_ids
    assert all(label == -100 for label in labels[:prompt_len])


def test_worker_keeps_native_agent_trace_uncompressed_and_supervises_all_assistants(
    monkeypatch,
):
    tokenizer = _install_worker_tokenizer(monkeypatch)
    messages = [
        {"role": "user", "content": "inspect"},
        {"role": "assistant", "content": "running command"},
        {"role": "user", "content": "terminal output"},
        {"role": "assistant", "content": "done"},
    ]

    result = preprocessing.worker_process_example(
        (
            {"messages": messages, "sub_dataset": "agent2"},
            16,
            "compression_prompt",
            None,
            None,
        )
    )

    assert result is not None
    assert result["memory_strings"] == []
    assert result["memory_positions"] == []
    assert result["memory_tokens"] == 0
    trainable_ids = [
        token_id
        for token_id, label in zip(result["base_input_ids"], result["base_labels"])
        if label != -100
    ]
    assert tokenizer.encode("running command") != []
    for token_id in tokenizer.encode("running command") + tokenizer.encode("done"):
        assert token_id in trainable_ids


def test_worker_compacts_agent_input_segments_and_supervises_expand_calls(
    monkeypatch,
):
    tokenizer = _install_worker_tokenizer(monkeypatch)
    messages = [
        {
            "role": "user",
            "content": (
                "seg_1\n<|memory_start|>compressed segment summary<|memory_end|>"
            ),
        },
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {
                        "name": "expand",
                        "arguments": {"segment_id": "seg_1"},
                    },
                }
            ],
        },
        {
            "role": "tool",
            "name": "expand",
            "tool_call_id": "call-1",
            "content": "the original segment text",
        },
        {"role": "assistant", "content": "FINAL: answer"},
    ]

    result = preprocessing.worker_process_example(
        (
            {
                "messages": messages,
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "expand",
                            "parameters": {"type": "object"},
                        },
                    }
                ],
                "sub_dataset": "synthetic_expansion",
            },
            16,
            "compression_prompt",
            None,
            None,
        )
    )

    assert result is not None
    assert result["memory_strings"] == ["compressed segment summary"]
    assert len(result["memory_positions"]) == 1
    start = result["memory_positions"][0]
    assert result["base_input_ids"][start : start + 3] == [101, 102, 103]
    assert result["base_labels"][start : start + 3] == [-100, -100, -100]

    trainable_ids = [
        token_id
        for token_id, label in zip(result["base_input_ids"], result["base_labels"])
        if label != -100
    ]
    rendered_call = json.dumps(messages[1]["tool_calls"][0], sort_keys=True)
    for token_id in tokenizer.encode(rendered_call) + tokenizer.encode("FINAL: answer"):
        assert token_id in trainable_ids

    tool_ids = tokenizer.encode("the original segment text")
    sequence = result["base_input_ids"]
    tool_start = next(
        index
        for index in range(len(sequence) - len(tool_ids) + 1)
        if sequence[index : index + len(tool_ids)] == tool_ids
    )
    assert result["base_labels"][tool_start : tool_start + len(tool_ids)] == [
        -100
    ] * len(tool_ids)


def test_worker_rejects_memory_marker_anywhere_in_agent_structure(monkeypatch):
    _install_worker_tokenizer(monkeypatch)
    result = preprocessing.worker_process_example(
        (
            {
                "messages": [
                    {"role": "user", "content": "inspect"},
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "type": "function",
                                "function": {
                                    "name": "terminal",
                                    "arguments": {"cmd": "<|memory|>"},
                                },
                            }
                        ],
                    },
                ]
            },
            16,
            "compression_prompt",
            None,
            None,
        )
    )
    assert result is None


def test_worker_rejects_conflicting_agent_trajectory_columns(monkeypatch):
    _install_worker_tokenizer(monkeypatch)
    result = preprocessing.worker_process_example(
        (
            {
                "messages": [{"role": "user", "content": "one"}],
                "conversations": [{"role": "user", "content": "two"}],
            },
            16,
            "compression_prompt",
            None,
            None,
        )
    )
    assert result is None
