"""Helpers for splicing LCLM soft tokens into a vLLM prompt-embeds tensor.

Used by the two-stage vLLM CLI (``inference.vllm_inference.encode`` and
``inference.vllm_inference.decode``). Not user-facing on its own — the
encoder and decoder live in separate processes so vLLM's GPU-memory
allocation doesn't starve the HF encoder.
"""

from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch


# --- Prompt parsing & soft-token splicing -----------------------------------


_MEMORY_START = "<|memory_start|>"
_MEMORY_END = "<|memory_end|>"
_MEMORY_PLACEHOLDER = "<|memory|>"


def _expand_memory_placeholders(prompt_str: str, embed_tokenizer, compression_ratio: int) -> Tuple[str, List[str]]:
    """Replace ``<|memory_start|>...content...<|memory_end|>`` with
    ``<|memory_start|>(<|memory|> * N)<|memory_end|>`` where ``N`` is
    the number of chunks the encoder will produce for ``content``.
    Returns the rewritten prompt and the list of raw content strings
    (in the order they appear).
    """
    pattern = re.escape(_MEMORY_START) + r"(.*?)" + re.escape(_MEMORY_END)
    contents: List[str] = []
    out = prompt_str
    for m in re.finditer(pattern, prompt_str, re.DOTALL):
        content = m.group(1)
        contents.append(content)
        token_ids = embed_tokenizer.encode(content, add_special_tokens=False)
        n_chunks = max(1, math.ceil(len(token_ids) / compression_ratio))
        replacement = _MEMORY_START + (_MEMORY_PLACEHOLDER * n_chunks) + _MEMORY_END
        # Replace just the first remaining occurrence so multiple
        # identical-content blocks don't collapse.
        original = _MEMORY_START + content + _MEMORY_END
        out = out.replace(original, replacement, 1)
    return out, contents


def _find_placeholder_spans(input_ids: List[int], decoder_tokenizer) -> List[Tuple[int, int]]:
    """Return ``[(start, end), ...]`` byte-positions of each contiguous
    ``<|memory|>`` run in ``input_ids``. ``start`` is the first
    placeholder position, ``end`` is one past the last (exclusive) —
    suitable for ``embeds[start:end] = ...``.
    """
    placeholder_id = decoder_tokenizer.convert_tokens_to_ids(_MEMORY_PLACEHOLDER)
    spans: List[Tuple[int, int]] = []
    i = 0
    n = len(input_ids)
    while i < n:
        if input_ids[i] == placeholder_id:
            j = i
            while j < n and input_ids[j] == placeholder_id:
                j += 1
            spans.append((i, j))
            i = j
        else:
            i += 1
    return spans


def _build_prompt_embeds(
    prompt: List[Dict[str, str]],
    lclm,
) -> torch.Tensor:
    """Single-prompt convenience wrapper around ``_build_prompt_embeds_batch``."""
    return _build_prompt_embeds_batch([prompt], lclm)[0]


def _build_prompt_embeds_batch(
    prompts,
    lclm,
) -> List[torch.Tensor]:
    """Encode + splice prompts in one batched encoder forward.

    Each prompt may be either:
      - a chat-format ``List[Dict[str, str]]`` (gets ``apply_chat_template``-d), or
      - a pre-templated ``str`` (used as-is, e.g. when the dataset already
        chat-templated its prompts).

    Finds every ``<|memory_start|>...<|memory_end|>`` span, flattens all
    spans across all prompts, runs the LCLM encoder *once* (mini-batched
    by ``encoder.max_encode_batch_size``), then splices the resulting
    latent tokens back into each prompt's decoder-embedding sequence.

    Returns one ``[seq_len, hidden_size]`` tensor per prompt, ready for
    ``vllm.LLM.generate(prompt_embeds=...)``.
    """
    decoder_tokenizer = lclm.decoder_tokenizer
    embed_tokenizer = lclm.encoder.embed_tokenizer
    compression_ratio = lclm.encoder.compression_ratio
    device = next(lclm.decoder.parameters()).device

    # 1. Per prompt: tokenize, expand placeholders, find spans.
    per_prompt: List[Tuple[List[int], List[Tuple[int, int]], List[str]]] = []
    for p in prompts:
        if isinstance(p, str):
            chat_str = p
        else:
            chat_str = decoder_tokenizer.apply_chat_template(
                p, tokenize=False, add_generation_prompt=True
            )
        expanded, contents = _expand_memory_placeholders(
            chat_str, embed_tokenizer, compression_ratio
        )
        input_ids = decoder_tokenizer.encode(expanded, add_special_tokens=True)
        spans = _find_placeholder_spans(input_ids, decoder_tokenizer)
        per_prompt.append((input_ids, spans, contents))

    # 2. Flatten memory spans across prompts -> one big batch for the encoder.
    flat_token_ids: List[List[int]] = []
    span_owners: List[int] = []  # which prompt each flat span belongs to
    for i, (_, _, contents) in enumerate(per_prompt):
        for c in contents:
            flat_token_ids.append(embed_tokenizer.encode(c, add_special_tokens=False))
            span_owners.append(i)

    # 3. ONE batched encoder + adapter forward. Internally mini-batched by
    #    encoder.max_encode_batch_size (windows-per-encoder-pass).
    if flat_token_ids:
        all_soft = lclm._process_latent_embeddings(flat_token_ids)
    else:
        all_soft = []

    # 4. Per prompt: build base embeddings, splice soft latents at spans.
    results: List[torch.Tensor] = []
    span_cursor = 0
    for i, (input_ids, spans, contents) in enumerate(per_prompt):
        n_spans = len(contents)
        soft_for_this = all_soft[span_cursor : span_cursor + n_spans]
        span_cursor += n_spans

        input_ids_t = torch.tensor(input_ids, device=device).unsqueeze(0)
        embeds = lclm.decoder.get_input_embeddings()(input_ids_t)

        if soft_for_this:
            if len(spans) != len(soft_for_this):
                raise RuntimeError(
                    f"Prompt {i}: {len(spans)} placeholder runs vs "
                    f"{len(soft_for_this)} encoder outputs"
                )
            for (start, end), soft in zip(spans, soft_for_this):
                n_pos = end - start
                n_soft = soft.shape[0]
                if n_pos != n_soft:
                    k = min(n_pos, n_soft)
                    embeds[0, start : start + k] = soft[:k]
                else:
                    embeds[0, start:end] = soft

        results.append(embeds.squeeze(0))

    return results
