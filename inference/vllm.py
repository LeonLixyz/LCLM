"""Track A: HF-encoder + vLLM-decoder serving for LCLM.

This is the production-leaning inference path. The encoder runs in
HuggingFace Transformers (we keep the original implementation so the
custom pooling / window logic stays exact); the decoder runs in vLLM,
which provides paged-attention KV-cache and high throughput. The
encoder's output soft tokens are spliced into the decoder's prompt
embedding sequence at the ``<|memory|>`` placeholder positions and
handed to ``vllm.LLM.generate(prompt_embeds=...)``.

This is the API the LCLM paper §5 alludes to when it says "compatible
with vLLM and SGLang": users get standard vLLM throughput on the
decoder side while the latent-context compression happens upstream.

Typical usage::

    from inference.vllm import LCLMVLLMDecoder

    runner = LCLMVLLMDecoder(
        checkpoint_path="latent-context/0.6b-4b-LCLM-16x",
        tensor_parallel_size=2,
    )
    outputs = runner.generate(
        prompts=[
            [{"role": "user",
              "content": "<|memory_start|>...long doc...<|memory_end|> "
                         "Summarize."}],
        ],
        max_tokens=512,
        temperature=0.0,
    )
    print(outputs[0])

For one-shot use, ``LCLMVLLMDecoder.generate(...)`` returns a list of
strings — one per prompt. For more control (logprobs, samplers, etc.),
the underlying ``vllm.LLM`` is exposed as ``runner.vllm``.
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


# --- The user-facing class --------------------------------------------------


class LCLMVLLMDecoder:
    """End-to-end LCLM inference: HF encoder (this process), vLLM decoder
    (process or remote, depending on vLLM mode).

    Parameters
    ----------
    checkpoint_path : str, optional
        Path or HF repo ID of an LCLM checkpoint. Either this or
        ``model`` must be given.
    model : LCLM, optional
        An already-loaded LCLM (e.g. when sharing the encoder with other
        code paths). Takes precedence over ``checkpoint_path``.
    tensor_parallel_size : int
        vLLM TP size for the decoder.
    decoder_path : str, optional
        Override which directory vLLM loads the decoder weights from.
        Defaults to ``<checkpoint_path>/decoder``.
    vllm_kwargs : dict, optional
        Extra kwargs forwarded to ``vllm.LLM(...)``. ``enable_prompt_embeds=True``
        is always set.

    Attributes
    ----------
    lclm : LCLM
        The loaded LCLM model (encoder + adapter + a CPU-or-GPU copy of
        the decoder used only for the embedding lookup).
    vllm : vllm.LLM
        The vLLM engine handling decoding. Use this directly if you need
        fine-grained control over sampling.
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        *,
        model: Optional[Any] = None,
        tensor_parallel_size: int = 1,
        decoder_path: Optional[str] = None,
        vllm_kwargs: Optional[Dict[str, Any]] = None,
        max_encode_batch_size: int = 128,
    ):
        if model is None and checkpoint_path is None:
            raise ValueError("Provide either checkpoint_path or model")
        if model is None:
            from latent_context import from_pretrained
            model, _, _ = from_pretrained(checkpoint_path)
        self.lclm = model

        # Cap on encoder windows per mini-batch. With W=1024 the default 128
        # corresponds to ~128k input tokens per encoder forward pass.
        self.lclm.encoder.max_encode_batch_size = max(0, int(max_encode_batch_size))

        if decoder_path is None:
            if checkpoint_path is None:
                raise ValueError("decoder_path is required when constructing from a preloaded model")
            import os
            if not os.path.isdir(checkpoint_path) and "/" in checkpoint_path:
                # HF repo id — already snapshot-downloaded by load_model in
                # from_pretrained above; reuse that local snapshot.
                from huggingface_hub import snapshot_download
                checkpoint_path = snapshot_download(repo_id=checkpoint_path, repo_type="model")
            decoder_path = os.path.join(checkpoint_path, "decoder")
            if not os.path.isdir(decoder_path):
                raise FileNotFoundError(f"{decoder_path} does not exist")

        from vllm import LLM  # imported lazily so import-time has no vLLM dep
        kwargs: Dict[str, Any] = {"enable_prompt_embeds": True,
                                  "tensor_parallel_size": tensor_parallel_size}
        if vllm_kwargs:
            kwargs.update(vllm_kwargs)
        self.vllm = LLM(model=decoder_path, **kwargs)

    # ------------------------------------------------------------------
    def encode(self, prompt: List[Dict[str, str]]) -> torch.Tensor:
        """Run the encoder + soft-token splicing on a single chat prompt."""
        with torch.inference_mode():
            return _build_prompt_embeds_batch([prompt], self.lclm)[0]

    def encode_batch(self, prompts: Sequence[List[Dict[str, str]]]) -> List[torch.Tensor]:
        """Batched encode + splice: one Encoder forward across all prompts."""
        with torch.inference_mode():
            return _build_prompt_embeds_batch(prompts, self.lclm)

    def generate(
        self,
        prompts: Sequence[List[Dict[str, str]]],
        *,
        max_tokens: int = 512,
        temperature: float = 0.0,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        sampling_params: Optional[Any] = None,
    ) -> List[str]:
        """Generate text for each prompt. Each prompt is a chat-format
        list of ``{"role": ..., "content": ...}`` dicts and may contain
        any number of ``<|memory_start|>...<|memory_end|>`` spans.

        The encoder runs **once** over all memory spans across all prompts
        (mini-batched by ``max_encode_batch_size`` set at construction);
        vLLM then does its own continuous batching on the decoder side.

        Returns the decoded text of the first sequence per prompt.
        """
        from vllm import SamplingParams
        if sampling_params is None:
            sampling_params = SamplingParams(
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
            )

        embeds = self.encode_batch(prompts)
        # vLLM expects float prompt embeds on CPU; move + dtype-cast.
        vllm_inputs = [{"prompt_embeds": e.float().cpu()} for e in embeds]
        outputs = self.vllm.generate(vllm_inputs, sampling_params=sampling_params)
        return [o.outputs[0].text for o in outputs]
