"""
Utility helpers for preparing chat-formatted data.

Simple, explicit functions for the Code-LLaVA dataset format:
- prompt: list of message dicts with 'role' and 'content'
- target: plain string (assistant response)
"""

from typing import Any, Dict, List, Tuple
from transformers import PreTrainedTokenizerBase

Message = Dict[str, str]


def build_prompt_and_target_text(
    prompt_data: Any,
    target_data: Any,
    *,
    tokenizer: PreTrainedTokenizerBase,
) -> Tuple[str, str, List[Message], List[Message]]:
    """
    Construct prompt/target strings using the tokenizer's chat template.

    Args:
        prompt_data: List of chat message dicts with 'role' and 'content'
        target_data: Plain string (assistant's response)
        tokenizer: Model tokenizer (must support apply_chat_template)

    Returns:
        Tuple of:
            - prompt_text: String used as chat prompt (generation-ready)
            - target_text: Assistant continuation string
            - prompt_messages: Prompt messages list
            - target_messages: Target messages list
    """
    # Prompt is always a list of message dicts
    if not isinstance(prompt_data, list):
        raise TypeError(f"prompt_data must be a list, got {type(prompt_data)}")
    prompt_messages = prompt_data
    
    # Target is always a plain string
    if not isinstance(target_data, str):
        raise TypeError(f"target_data must be a string, got {type(target_data)}")
    
    # Create target message
    target_messages = [{"role": "assistant", "content": target_data}]

    if getattr(tokenizer, "chat_template", None):
        prompt_text = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Build full conversation and extract target portion
        full_messages = prompt_messages + target_messages
        full_text = tokenizer.apply_chat_template(
            full_messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        if not full_text.startswith(prompt_text):
            raise ValueError("Prompt text is not a prefix of the full conversation.")
        target_text = full_text[len(prompt_text):]
    else:
        # Fallback for tokenizers without chat templates
        prompt_text = "\n".join(message["content"] for message in prompt_messages)
        if prompt_messages and prompt_text and not prompt_text.endswith("\n"):
            prompt_text += "\n"
        target_text = target_data

    return prompt_text, target_text, prompt_messages, target_messages
