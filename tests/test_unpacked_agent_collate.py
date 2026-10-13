import json
from types import SimpleNamespace

from data.dataset import collate_batch


class FakeQwenTokenizer:
    chat_template = "fake-qwen"
    pad_token_id = 0

    @staticmethod
    def _render(messages, *, tools, add_generation_prompt):
        text = ""
        if tools:
            text += "S>" + json.dumps(tools, sort_keys=True) + "<eos>"
        for message in messages:
            role = message["role"]
            content = message.get("content")
            content = content if isinstance(content, str) else ""
            prefix = "A>" if role == "assistant" else role[:1].upper() + ">"
            text += prefix + content
            for tool_call in message.get("tool_calls", []):
                text += "<tool_call>" + json.dumps(tool_call, sort_keys=True)
                text += "</tool_call>"
            text += "<eoa>" if role == "assistant" else "<eou>"
        if add_generation_prompt:
            text += "A>"
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
        rendered = self._render(
            messages, tools=tools, add_generation_prompt=add_generation_prompt
        )
        return [ord(character) for character in rendered] if tokenize else rendered


def decode_trainable(input_ids, labels):
    return "".join(
        chr(int(token))
        for token, label in zip(input_ids.tolist(), labels.tolist())
        if label != -100
    )


def test_unpacked_collate_supports_native_agent_rows_and_right_padding():
    processor = SimpleNamespace(decoder_tokenizer=FakeQwenTokenizer())
    examples = [
        {
            "messages": [
                {"role": "user", "content": "task one"},
                {"role": "assistant", "content": "first action"},
                {"role": "user", "content": "observation"},
                {"role": "assistant", "content": "finished"},
            ]
        },
        {
            "conversations": [
                {"role": "user", "content": "task two"},
                {"role": "assistant", "content": "done"},
            ]
        },
    ]

    batch = collate_batch(examples, processor, max_memory_length=0)

    assert batch["input_ids"].shape[0] == 2
    assert batch["input_ids"].shape == batch["labels"].shape
    assert batch["memory_token_ids"] == [[], []]
    assert batch["memory_positions"] == [[], []]
    assert batch["latent_counts"] == [[], []]

    first = decode_trainable(batch["input_ids"][0], batch["labels"][0])
    second = decode_trainable(batch["input_ids"][1], batch["labels"][1])
    assert first == "first action<eoa>finished<eoa>"
    assert second == "done<eoa>"
    assert "task" not in first + second
    assert "observation" not in first


def test_unpacked_collate_rejects_memory_markers_in_agent_trace():
    processor = SimpleNamespace(decoder_tokenizer=FakeQwenTokenizer())
    try:
        collate_batch(
            [
                {
                    "messages": [
                        {"role": "user", "content": "task"},
                        {"role": "assistant", "content": "bad <|memory|>"},
                    ]
                }
            ],
            processor,
            max_memory_length=0,
        )
    except ValueError as error:
        assert "must not contain" in str(error)
    else:
        raise AssertionError("agent memory marker should be rejected")

