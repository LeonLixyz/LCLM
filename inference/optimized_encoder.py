"""Inference-only fast path for the LCLM encoder.

This is the encoder implementation behind every LCLM timing in the paper
(Figures 1, 4 and 5). It changes how encoder windows are packed and executed;
the computation is mathematically unchanged:

* **Right padding, no mask.** The stock encoder left-pads windows and passes a
  padding mask. With a causal encoder, padding placed *after* the valid tokens
  is never attended to by them, so the fast path right-pads, starts every
  window at position 0 and runs attention with ``is_causal=True`` and no mask.
  This lets PyTorch SDPA use its FlashAttention backend.
* **torch.compile + CUDA graphs.** The encoder transformer is compiled
  (``fullgraph=True, dynamic=True``) and captured in one CUDA graph per input
  shape; later calls copy into static buffers and replay the graph.
* **Fixed encoder batch.** ``batch_windows`` windows per encoder forward
  (default 32, i.e. 32,768 tokens per pass at ``W=1024``).
* **Vectorized mean pooling.**

Supported configuration (all released checkpoints): causal encoder mask,
``pooling="mean"``, ``boundary_overlap=0``, no sliding-window layers, CUDA.

The first call with a new batch shape compiles and captures a graph, which
takes seconds to minutes. Treat it as setup and warm up before timing. At most
two shapes are kept (full batches and the trailing partial batch); a third
shape frees them and captures again. Graph buffers are created under the
caller's grad mode, so call the model under ``torch.inference_mode()``
consistently.

Latents are not bit-identical to the stock encoder: in BF16, the Flash kernel
and ``torch.compile`` change numerics (about 6% relative L2 difference on the
synthetic 4k timing input). The paper's accuracy results use the stock
encoder; this path is used for its timings.

Usage::

    model, tokenizer, processor = load_model(checkpoint)
    model.eval()
    optimization = enable_encoder_optimization(model)
    ...
    optimization.restore()  # back to the stock encoder
"""

import time
import types
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

DEFAULT_BATCH_WINDOWS = 32
MAX_CACHED_GRAPHS = 2
GRAPH_WARMUP_ITERS = 3


def set_attn_implementation(model, name: str) -> None:
    """Switch a Transformers model's attention backend in place."""
    if hasattr(model, "set_attn_implementation"):
        model.set_attn_implementation(name)
        return
    for module in model.modules():
        config = getattr(module, "config", None)
        if config is not None and hasattr(config, "_attn_implementation"):
            config._attn_implementation = name


def check_encoder_supported(encoder) -> None:
    """Raise ``ValueError`` if ``encoder`` cannot use the fast path."""
    problems = []
    if not encoder.is_causal:
        problems.append("encoder_mask_type must be 'causal'")
    if encoder.boundary_overlap != 0:
        problems.append(f"boundary_overlap must be 0 (got {encoder.boundary_overlap})")
    if encoder.pooling != "mean":
        problems.append(f"pooling must be 'mean' (got {encoder.pooling!r})")
    if getattr(encoder.embed_model, "has_sliding_layers", False):
        problems.append("sliding-window attention layers are not supported")
    if problems:
        raise ValueError("Optimized encoder does not support this encoder: " + "; ".join(problems))


def _mean_pool(encoder, pool_hs: torch.Tensor, pool_len: int, hidden_dim: int) -> torch.Tensor:
    """Vectorized ``Encoder._pool_groupwise`` for ``pooling="mean"``."""
    ratio = encoder.compression_ratio
    num_full, tail = divmod(pool_len, ratio)
    head = pool_hs[: num_full * ratio].reshape(num_full, ratio, hidden_dim).mean(dim=1)
    if not tail:
        return head
    return torch.cat([head, pool_hs[num_full * ratio :].mean(dim=0, keepdim=True)], dim=0)


def _right_padded_forward(self, memory_token_ids) -> List[torch.Tensor]:
    """``Encoder.forward`` with right-padded windows.

    Same windows, batching and output layout as
    ``latent_context.encoder.Encoder.forward``. With ``boundary_overlap=0`` and
    a causal mask a window is exactly its own tokens, so ``pool_start`` is 0.
    """
    device = next(self.embed_model.parameters()).device

    # 1) Tile each sample into windows: (sample_idx, tokens).
    windows: List[Tuple[int, List[int]]] = []
    empty_samples = set()
    for sample_idx, token_ids in enumerate(memory_token_ids):
        token_ids = self._to_list(token_ids)
        if not token_ids:
            empty_samples.add(sample_idx)
            continue
        for offset in range(0, len(token_ids), self.encoder_window_size):
            windows.append((sample_idx, token_ids[offset : offset + self.encoder_window_size]))

    if not windows:
        return [torch.zeros(1, self.embedding_dim, device=device) for _ in memory_token_ids]

    # 2) Pack windows into a right-padded batch; positions restart at 0 per window.
    max_seq_len = max(len(tokens) for _, tokens in windows)
    total = len(windows)
    packed = np.full((total, max_seq_len), self.pad_token_id, dtype=np.int64)
    for i, (_, tokens) in enumerate(windows):
        packed[i, : len(tokens)] = tokens
    input_ids = torch.from_numpy(packed).to(device)
    attention_mask = torch.ones_like(input_ids)
    position_ids = torch.arange(max_seq_len, device=device, dtype=torch.long)[None, :].expand(total, -1)

    # 3) Encode in batches of max_encode_batch_size windows.
    hidden_states = self._batched_encode(input_ids, attention_mask, position_ids, total, device)

    # 4) Pool each window's valid prefix and regroup by sample.
    hidden_dim = self.embed_model.config.hidden_size
    by_sample: Dict[int, List[torch.Tensor]] = defaultdict(list)
    for i, (sample_idx, tokens) in enumerate(windows):
        pool_len = len(tokens)
        by_sample[sample_idx].append(_mean_pool(self, hidden_states[i, :pool_len], pool_len, hidden_dim))

    result: List[torch.Tensor] = []
    for sample_idx in range(len(memory_token_ids)):
        if sample_idx in empty_samples:
            result.append(torch.zeros(1, self.embedding_dim, device=device, dtype=hidden_states.dtype))
        else:
            result.append(torch.cat(by_sample[sample_idx], dim=0))
    return result


def _encode_without_mask(embed_model, input_ids, position_ids, cache_position=None) -> torch.Tensor:
    # A dict skips Transformers' mask construction; ``None`` makes every
    # attention layer run SDPA with is_causal=True.
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        return embed_model(
            input_ids=input_ids,
            attention_mask={"full_attention": None},
            position_ids=position_ids,
            cache_position=cache_position,
            use_cache=False,
        ).last_hidden_state


class _GraphedEncoderForward:
    """Replacement for ``Encoder._forward_one`` that replays one CUDA graph per shape."""

    def __init__(self, encoder):
        self.encoder = encoder
        self.graphs = {}
        self.capture_log: List[dict] = []

    @torch.no_grad()
    def __call__(self, input_ids, attention_mask, position_ids):
        del attention_mask  # right padding under a causal mask needs no mask
        key = tuple(input_ids.shape)
        if key not in self.graphs:
            self.graphs[key] = self._capture(input_ids, position_ids)
        static_ids, static_pos, _, graph, output = self.graphs[key]
        static_ids.copy_(input_ids)
        static_pos.copy_(position_ids)
        graph.replay()
        return output.clone()

    def _capture(self, input_ids, position_ids):
        if len(self.graphs) >= MAX_CACHED_GRAPHS:
            torch.cuda.synchronize()
            self.graphs.clear()
            torch.cuda.empty_cache()
        start = time.perf_counter()
        embed_model = self.encoder.embed_model
        static_ids = input_ids.clone()
        static_pos = position_ids.clone()
        cache_position = torch.arange(input_ids.shape[1], device=input_ids.device)

        def run():
            return _encode_without_mask(embed_model, static_ids, static_pos, cache_position)

        # Warm up (and trigger torch.compile) on a side stream before capture.
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(GRAPH_WARMUP_ITERS):
                run()
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = run()
        torch.cuda.synchronize()
        self.capture_log.append(
            dict(shape=list(input_ids.shape), setup_seconds=time.perf_counter() - start)
        )
        return static_ids, static_pos, cache_position, graph, output


class EncoderOptimization:
    """Handle returned by :func:`enable_encoder_optimization`."""

    def __init__(self, encoder, compile_encoder: bool, batch_windows: int):
        self.encoder = encoder
        self.compile_encoder = compile_encoder
        self.batch_windows = batch_windows
        embed_model = encoder.embed_model
        self._saved = dict(
            max_encode_batch_size=encoder.max_encode_batch_size,
            use_cache=embed_model.config.use_cache,
            attn_implementation=getattr(embed_model.config, "_attn_implementation", None),
            encoder_attrs={name: encoder.__dict__[name] for name in ("forward", "_forward_one") if name in encoder.__dict__},
            embed_forward=embed_model.__dict__.get("forward"),
        )
        self._graphed = _GraphedEncoderForward(encoder)

    @property
    def capture_log(self) -> List[dict]:
        """Shape and setup seconds of every CUDA graph captured so far."""
        return self._graphed.capture_log

    def _install(self) -> None:
        encoder, embed_model = self.encoder, self.encoder.embed_model
        set_attn_implementation(embed_model, "sdpa")
        embed_model.config.use_cache = False
        encoder.max_encode_batch_size = self.batch_windows
        if self.compile_encoder:
            embed_model.forward = torch.compile(
                embed_model.forward, fullgraph=True, dynamic=True, mode="default"
            )
        encoder.forward = types.MethodType(_right_padded_forward, encoder)
        encoder._forward_one = self._graphed
        encoder._lclm_optimization = self

    def restore(self) -> None:
        """Reinstate the stock encoder and release captured graphs."""
        encoder, embed_model = self.encoder, self.encoder.embed_model
        for name in ("forward", "_forward_one"):
            if name in self._saved["encoder_attrs"]:
                setattr(encoder, name, self._saved["encoder_attrs"][name])
            else:
                encoder.__dict__.pop(name, None)
        if self._saved["embed_forward"] is not None:
            embed_model.forward = self._saved["embed_forward"]
        else:
            embed_model.__dict__.pop("forward", None)
        encoder.max_encode_batch_size = self._saved["max_encode_batch_size"]
        embed_model.config.use_cache = self._saved["use_cache"]
        if self._saved["attn_implementation"] is not None:
            set_attn_implementation(embed_model, self._saved["attn_implementation"])
        encoder.__dict__.pop("_lclm_optimization", None)
        self._graphed.graphs.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def enable_encoder_optimization(
    model,
    compile_encoder: bool = True,
    batch_windows: int = DEFAULT_BATCH_WINDOWS,
) -> EncoderOptimization:
    """Install the optimized encoder on an LCLM in eval mode.

    Args:
        model: an ``LCLM`` (e.g. from ``inference.hf.load_model``) on a CUDA
            device, already in ``eval()`` mode.
        compile_encoder: wrap the encoder transformer in ``torch.compile``.
            With ``False`` the fast path still uses right padding, Flash SDPA
            and CUDA graphs.
        batch_windows: encoder windows per forward pass.

    Switches the encoder's attention to SDPA and patches ``model.encoder`` in
    place. The first call per batch shape compiles and captures a CUDA graph
    (not free; warm up before timing). Returns a handle whose ``restore()``
    reinstates the stock encoder.
    """
    if model.training:
        raise ValueError(
            "enable_encoder_optimization is inference-only; call model.eval() first"
        )
    encoder = model.encoder
    if getattr(encoder, "_lclm_optimization", None) is not None:
        raise RuntimeError("Encoder optimization is already enabled; call restore() first")
    check_encoder_supported(encoder)
    if int(batch_windows) < 1:
        raise ValueError(f"batch_windows must be >= 1 (got {batch_windows})")
    device = next(encoder.embed_model.parameters()).device
    if device.type != "cuda":
        raise ValueError(f"The optimized encoder needs a CUDA device (encoder is on {device})")
    optimization = EncoderOptimization(encoder, bool(compile_encoder), int(batch_windows))
    optimization._install()
    return optimization
