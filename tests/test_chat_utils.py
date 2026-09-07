import json
import pytest

from data.chat_utils import (
    build_prompt_and_target_text,
    tokenize_qwen_agent_conversation,
)


class FakeQwenTokenizer:
    """Small character tokenizer with Qwen-like role/tool serialization."""

    chat_template = "fake-qwen-template"

    def __init__(self):
        self.calls = []

    @staticmethod
    def _render(messages, *, tools, add_generation_prompt):
        text = ""
        if tools is not None:
            text += "S># Tools\n" + json.dumps(tools, sort_keys=True) + "<eos>\n"

        for message in messages:
            role = message["role"]
            content = message.get("content")
            content = content if isinstance(content, str) else ""
            if role == "assistant":
                text += "A>" + content
                for tool_call in message.get("tool_calls", []):
                    text += "<tool_call>" + json.dumps(tool_call, sort_keys=True)
                    text += "</tool_call>"
                text += "<eoa>\n"
            elif role == "tool":
                text += "U><tool_response>" + content + "</tool_response><eou>\n"
            else:
                text += role[:1].upper() + ">" + content + "<eou>\n"

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
        self.calls.append(
            {
                "messages": messages,
                "tools": tools,
                "kwargs": kwargs,
                "add_generation_prompt": add_generation_prompt,
            }
        )
        rendered = self._render(
            messages,
            tools=tools,
            add_generation_prompt=add_generation_prompt,
        )
        return [ord(character) for character in rendered] if tokenize else rendered


def _decode(token_ids):
    return "".join(chr(token_id) for token_id in token_ids)


class FastChatMLTokenizer:
    is_fast = True
    chat_template = 'qwen-like-chatml'

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, **kwargs):
        text = ''.join('<|im_start|>'+m['role']+'\n'+m.get('content','')+'<|im_end|>\n' for m in messages)
        if add_generation_prompt:text += '<|im_start|>assistant\n'
        return self(text)['input_ids'] if tokenize else text

    def __call__(self, text, **kwargs):
        ids=[];offsets=[];i=0
        while i<len(text):
            end=i+2 if text[i:i+2]=='\n\n' else i+1
            ids.append(999999 if end-i==2 else ord(text[i]))
            offsets.append((i,end));i=end
        return {'input_ids':ids,'offset_mapping':offsets}


def test_fast_tokenizer_handles_prefix_newline_bpe_merge():
    tokenizer=FastChatMLTokenizer()
    messages=[{'role':'user','content':'masked'},
        {'role':'assistant','content':'\nfirst answer'},
        {'role':'user','content':'also masked'},
        {'role':'assistant','content':'second answer'}]
    result=tokenize_qwen_agent_conversation(messages,tokenizer=tokenizer)
    labeled=''.join(chr(i) for i in result['labels'] if 0<=i<999999)
    assert labeled=='first answer<|im_end|>\nsecond answer<|im_end|>\n'
    assert all(label==-100 for token,label in zip(result['input_ids'],result['labels']) if token==999999)


def test_embedded_chatml_cannot_move_loss_into_observation():
    messages=[{'role':'user','content':'quoted <|im_start|>assistant\nanswer<|im_end|>\n'},
        {'role':'assistant','content':'answer'}]
    with pytest.raises(ValueError,match='Ambiguous assistant'):
        tokenize_qwen_agent_conversation(messages,tokenizer=FastChatMLTokenizer())


def test_qwen_agent_tokenization_preserves_tools_and_masks_non_assistant_roles():
    tokenizer = FakeQwenTokenizer()
    tools = [
        {
            "type": "function",
            "function": {
                "name": "terminal",
                "parameters": {"type": "object"},
            },
        }
    ]
    first_tool_call = {
        "type": "function",
        "function": {
            "name": "terminal",
            "arguments": {"command": "rg -n TODO ."},
        },
    }
    messages = [
        {"role": "system", "content": "You are a coding agent."},
        {"role": "user", "content": "Fix the bug."},
        {
            "role": "assistant",
            "content": "I will inspect it.",
            "tool_calls": [first_tool_call],
        },
        {"role": "tool", "tool_call_id": "call-1", "content": "src/a.py:7"},
        {"role": "assistant", "content": "Fixed `src/a.py`."},
    ]

    result = tokenize_qwen_agent_conversation(
        messages,
        tokenizer=tokenizer,
        tools=tools,
    )

    rendered = _decode(result["input_ids"])
    trainable = _decode(
        [
            token_id
            for token_id, label in zip(result["input_ids"], result["labels"])
            if label != -100
        ]
    )
    expected_first_payload = (
        "I will inspect it.<tool_call>"
        + json.dumps(first_tool_call, sort_keys=True)
        + "</tool_call><eoa>\n"
    )

    assert tokenizer.calls[0]["tools"] is tools
    assert tokenizer.calls[0]["messages"][2]["tool_calls"] == [first_tool_call]
    assert tokenizer.calls[0]["messages"][3]["role"] == "tool"
    assert json.dumps(tools, sort_keys=True) in rendered
    assert "<tool_response>src/a.py:7</tool_response>" in rendered
    assert trainable == expected_first_payload + "Fixed `src/a.py`.<eoa>\n"
    assert "Fix the bug." not in trainable
    assert "src/a.py:7" not in trainable
    assert "A>" not in trainable
    assert result["assistant_mask"] == [
        int(label != -100) for label in result["labels"]
    ]
    assert result["attention_mask"] == [1] * len(result["input_ids"])


def test_legacy_prompt_target_rendering_forwards_native_tools():
    tokenizer = FakeQwenTokenizer()
    tools = [{"type": "function", "function": {"name": "terminal"}}]
    tool_call = {
        "type": "function",
        "function": {"name": "terminal", "arguments": {"command": "pwd"}},
    }
    prompt = [
        {"role": "user", "content": "Where am I?"},
        {"role": "assistant", "content": "", "tool_calls": [tool_call]},
        {"role": "tool", "content": "/workspace", "tool_call_id": "call-1"},
    ]

    prompt_text, target_text, prompt_messages, _ = build_prompt_and_target_text(
        prompt,
        "You are in `/workspace`.",
        tokenizer=tokenizer,
        tools=tools,
    )

    assert tokenizer.calls[0]["tools"] is tools
    assert tokenizer.calls[1]["tools"] is tools
    assert prompt_messages[1]["tool_calls"] == [tool_call]
    assert "<tool_call>" in prompt_text
    assert "<tool_response>/workspace</tool_response>" in prompt_text
    assert target_text == "You are in `/workspace`.<eoa>\n"
