"""Utilities for rendering and tokenizing chat-formatted training data.

In addition to the legacy ``prompt`` + ``target`` format, this module handles
native multi-turn Qwen agent conversations.  Agent messages are intentionally
passed to ``apply_chat_template`` without projecting them down to
``role``/``content`` so that ``tool_calls`` and ``tool`` messages survive.
"""

from collections.abc import Mapping, Sequence
import re
from typing import Any, Dict, List, Optional, Tuple

from transformers import PreTrainedTokenizerBase

Message = Dict[str, Any]


def _validate_messages(messages_data: Any) -> List[Message]:
    """Validate the outer chat structure without discarding native fields."""
    if not isinstance(messages_data, list):
        raise TypeError(f"messages must be a list, got {type(messages_data)}")

    messages: List[Message] = []
    for index, message in enumerate(messages_data):
        if not isinstance(message, Mapping):
            raise TypeError(
                f"messages[{index}] must be a mapping, got {type(message)}"
            )
        role = message.get("role")
        if not isinstance(role, str) or not role:
            raise ValueError(f"messages[{index}] must have a non-empty string role")

        # This is a shallow copy on purpose: it retains content=None, tool_calls,
        # reasoning_content, and any tokenizer-specific message fields verbatim.
        messages.append(dict(message))
    return messages


def _chat_template_kwargs(
    *,
    tools: Optional[Sequence[Mapping[str, Any]]],
    chat_template_kwargs: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    kwargs = dict(chat_template_kwargs or {})
    if tools is not None:
        if "tools" in kwargs:
            raise ValueError("Pass tools either directly or in chat_template_kwargs, not both.")
        # Keep the original tool definitions intact. Qwen's template serializes
        # these into its system tool block.
        kwargs["tools"] = tools
    return kwargs


def _apply_chat_template(
    tokenizer: PreTrainedTokenizerBase,
    messages: List[Message],
    *,
    tokenize: bool,
    add_generation_prompt: bool,
    tools: Optional[Sequence[Mapping[str, Any]]] = None,
    chat_template_kwargs: Optional[Mapping[str, Any]] = None,
) -> Any:
    kwargs = _chat_template_kwargs(
        tools=tools,
        chat_template_kwargs=chat_template_kwargs,
    )
    return tokenizer.apply_chat_template(
        messages,
        tokenize=tokenize,
        add_generation_prompt=add_generation_prompt,
        **kwargs,
    )


def _as_token_ids(value: Any) -> List[int]:
    """Normalize the non-batched output of ``apply_chat_template``."""
    if isinstance(value, Mapping):
        if "input_ids" not in value:
            raise TypeError("Chat-template mapping did not contain input_ids.")
        value = value["input_ids"]
    if hasattr(value, "tolist"):
        value = value.tolist()
    if (
        isinstance(value, list)
        and len(value) == 1
        and isinstance(value[0], list)
    ):
        value = value[0]
    if not isinstance(value, list) or any(not isinstance(token, int) for token in value):
        raise TypeError("apply_chat_template(tokenize=True) must return one token-id list.")
    return value


def _find_token_subsequence(
    sequence: List[int],
    subsequence: List[int],
    *,
    start: int,
) -> int:
    """Find a token subsequence, using its first (normally special) token as an anchor."""
    if not subsequence:
        raise ValueError("Cannot locate an empty assistant turn.")

    candidate = start
    limit = len(sequence) - len(subsequence)
    while candidate <= limit:
        try:
            candidate = sequence.index(subsequence[0], candidate, limit + 1)
        except ValueError as exc:
            raise ValueError(
                "Could not align an assistant turn with the rendered conversation. "
                "Use the Qwen3-4B-Instruct-2507 tokenizer chat template unchanged."
            ) from exc
        if sequence[candidate : candidate + len(subsequence)] == subsequence:
            return candidate
        candidate += 1

    raise ValueError(
        "Could not align an assistant turn with the rendered conversation. "
        "Use the Qwen3-4B-Instruct-2507 tokenizer chat template unchanged."
    )


def tokenize_qwen_agent_conversation(
    messages_data: Any,
    *,
    tokenizer: PreTrainedTokenizerBase,
    tools: Optional[Sequence[Mapping[str, Any]]] = None,
    chat_template_kwargs: Optional[Mapping[str, Any]] = None,
    ignore_index: int = -100,
) -> Dict[str, List[int]]:
    """Tokenize a native Qwen agent trace with loss on every assistant turn.

    The complete conversation is rendered once through the tokenizer's own
    chat template.  ``tools`` are forwarded unchanged, assistant
    ``tool_calls`` remain on their messages, and ``tool`` observations remain
    tool-role messages.  Labels are then enabled only for each assistant
    payload (normal content, serialized tool calls, and the turn terminator);
    system/user/tool text and the assistant role prefix are masked.

    This alignment deliberately targets the shipped
    ``Qwen/Qwen3-4B-Instruct-2507`` template.  It validates that a standalone
    assistant turn has the same tokenization inside the full conversation and
    fails loudly instead of silently assigning loss to the wrong role.
    """
    if not getattr(tokenizer, "chat_template", None):
        raise ValueError("Native agent tokenization requires a tokenizer chat template.")

    messages = _validate_messages(messages_data)
    if not messages:
        raise ValueError("messages must contain at least one chat message")

    if getattr(tokenizer, 'is_fast', False):
        # BPE can merge the assistant prefix's newline with leading payload
        # whitespace. Token-prefix matching then rejects otherwise valid rows.
        # Align characters first and use offsets from the full tokenization.
        rendered = _apply_chat_template(tokenizer, messages, tokenize=False,
            add_generation_prompt=False, tools=tools,
            chat_template_kwargs=chat_template_kwargs)
        encoded = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
        full_ids = list(encoded['input_ids'])
        spans = []
        cursor = 0
        for message in messages:
            if message['role'] != 'assistant':
                continue
            turn = _apply_chat_template(tokenizer, [message], tokenize=False,
                add_generation_prompt=False, chat_template_kwargs=chat_template_kwargs)
            prefix = '<|im_start|>assistant\n'
            if not turn.startswith(prefix):
                raise ValueError('Expected the unchanged Qwen3-Instruct ChatML template')
            start = rendered.find(turn, cursor)
            if start < 0:
                raise ValueError('Cannot align assistant turn in complete Qwen conversation')
            spans.append((start + len(prefix), start + len(turn)))
            cursor = start + len(turn)
        # Fail closed if message text injects extra ChatML turns; otherwise a
        # quoted assistant turn inside a tool result could receive loss.
        parsed = [(m.start(1), m.end()) for m in re.finditer(
            r'<\|im_start\|>assistant\n(.*?)<\|im_end\|>\n', rendered, re.S)]
        if parsed != spans:
            raise ValueError('Ambiguous assistant boundaries in rendered ChatML')
        mask = []
        span_index = 0
        for start, end in encoded['offset_mapping']:
            while span_index < len(spans) and start >= spans[span_index][1]:
                span_index += 1
            mask.append(int(span_index < len(spans) and start < end
                and spans[span_index][0] <= start and end <= spans[span_index][1]))
        return {'input_ids': full_ids, 'attention_mask': [1] * len(full_ids),
            'labels': [token if enabled else ignore_index for token, enabled in zip(full_ids, mask)],
            'assistant_mask': mask}

    full_ids = _as_token_ids(
        _apply_chat_template(
            tokenizer,
            messages,
            tokenize=True,
            add_generation_prompt=False,
            tools=tools,
            chat_template_kwargs=chat_template_kwargs,
        )
    )

    # Obtain the exact assistant role prefix from the template rather than
    # hard-coding token IDs for ``<|im_start|>assistant\n``.
    dummy_user = [{"role": "user", "content": ""}]
    dummy_without_generation = _as_token_ids(
        _apply_chat_template(
            tokenizer,
            dummy_user,
            tokenize=True,
            add_generation_prompt=False,
            chat_template_kwargs=chat_template_kwargs,
        )
    )
    dummy_with_generation = _as_token_ids(
        _apply_chat_template(
            tokenizer,
            dummy_user,
            tokenize=True,
            add_generation_prompt=True,
            chat_template_kwargs=chat_template_kwargs,
        )
    )
    if dummy_with_generation[: len(dummy_without_generation)] != dummy_without_generation:
        raise ValueError("The tokenizer's generation prompt is not prefix-stable.")
    assistant_prefix_ids = dummy_with_generation[len(dummy_without_generation) :]
    if not assistant_prefix_ids:
        raise ValueError("The tokenizer chat template has no assistant generation prefix.")

    labels = [ignore_index] * len(full_ids)
    assistant_mask = [0] * len(full_ids)
    search_start = 0

    for message in messages:
        if message["role"] != "assistant":
            continue

        # Do not pass top-level tools here: that would prepend Qwen's system
        # tool block. The message's own tool_calls are still rendered natively.
        assistant_turn_ids = _as_token_ids(
            _apply_chat_template(
                tokenizer,
                [message],
                tokenize=True,
                add_generation_prompt=False,
                chat_template_kwargs=chat_template_kwargs,
            )
        )
        if assistant_turn_ids[: len(assistant_prefix_ids)] != assistant_prefix_ids:
            raise ValueError(
                "Assistant-turn serialization does not start with the tokenizer's "
                "assistant generation prefix."
            )

        turn_start = _find_token_subsequence(
            full_ids,
            assistant_turn_ids,
            start=search_start,
        )
        payload_start = turn_start + len(assistant_prefix_ids)
        turn_end = turn_start + len(assistant_turn_ids)
        for index in range(payload_start, turn_end):
            labels[index] = full_ids[index]
            assistant_mask[index] = 1
        search_start = turn_end

    return {
        "input_ids": full_ids,
        "attention_mask": [1] * len(full_ids),
        "labels": labels,
        "assistant_mask": assistant_mask,
    }


def build_prompt_and_target_text(
    prompt_data: Any,
    target_data: Any,
    *,
    tokenizer: PreTrainedTokenizerBase,
    tools: Optional[Sequence[Mapping[str, Any]]] = None,
    chat_template_kwargs: Optional[Mapping[str, Any]] = None,
) -> Tuple[str, str, List[Message], List[Message]]:
    """
    Construct prompt/target strings using the tokenizer's chat template.

    Args:
        prompt_data: List of chat message dicts with 'role' and 'content'
        target_data: Plain string (assistant's response)
        tokenizer: Model tokenizer (must support apply_chat_template)
        tools: Optional native tool definitions forwarded to the chat template
        chat_template_kwargs: Optional extra tokenizer chat-template arguments

    Returns:
        Tuple of:
            - prompt_text: String used as chat prompt (generation-ready)
            - target_text: Assistant continuation string
            - prompt_messages: Prompt messages list
            - target_messages: Target messages list
    """
    prompt_messages = _validate_messages(prompt_data)
    
    # Target is always a plain string
    if not isinstance(target_data, str):
        raise TypeError(f"target_data must be a string, got {type(target_data)}")
    
    # Create target message
    target_messages = [{"role": "assistant", "content": target_data}]

    if getattr(tokenizer, "chat_template", None):
        prompt_text = _apply_chat_template(
            tokenizer,
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
            tools=tools,
            chat_template_kwargs=chat_template_kwargs,
        )

        # Build full conversation and extract target portion
        full_messages = prompt_messages + target_messages
        full_text = _apply_chat_template(
            tokenizer,
            full_messages,
            tokenize=False,
            add_generation_prompt=False,
            tools=tools,
            chat_template_kwargs=chat_template_kwargs,
        )
        if not full_text.startswith(prompt_text):
            raise ValueError("Prompt text is not a prefix of the full conversation.")
        target_text = full_text[len(prompt_text):]
    else:
        if tools is not None:
            raise ValueError("Cannot serialize native tools without a chat template.")
        # Fallback for tokenizers without chat templates
        prompt_text = "\n".join(message["content"] for message in prompt_messages)
        if prompt_messages and prompt_text and not prompt_text.endswith("\n"):
            prompt_text += "\n"
        target_text = target_data

    return prompt_text, target_text, prompt_messages, target_messages
