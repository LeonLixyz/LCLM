"""LCLM encoder: compresses input tokens into latent vectors.

For an input of T tokens, produces ``ceil(T / compression_ratio)`` latents. The
encoder transformer processes the input in windows of ``encoder_window_size``
(``W``) tokens; within each window, hidden states are pooled per
``compression_ratio``-sized group.

When ``W == compression_ratio``, each chunk is encoded in isolation (the simple
case). When ``W > compression_ratio``, multiple chunks share an encoder forward
pass and attend to each other within the window.

Pooling modes (``pooling``):
- ``eos``    append ``num_latents`` EOS markers at the end of the encoder
             input window (after any boundary overlap), encode, and read
             out the marker hidden states. Under causal masks each marker
             sees the full window; under bidirectional masks all positions
             see each other. Same idea as the older ``summary`` mode, just
             reusing the EOS token instead of N distinct ``<|memory_i|>``
             tokens (the differentiation comes from position).
- ``mean``   encode the raw window, split hidden states into ``compression_ratio``
             groups, take the mean of each group.
- ``concat`` encode the raw window, split hidden states into ``compression_ratio``
             groups, flatten each (output width ``= compression_ratio * hidden_size``).

For ``W > compression_ratio``, an optional ``boundary_overlap`` adds
``boundary_overlap`` context tokens before each window (and after, for
bidirectional masks) so chunks near window boundaries see neighboring text.
"""
import math
from collections import defaultdict
from typing import List, Optional, Tuple

import torch
import torch.nn as nn


class Encoder(nn.Module):
    """Token → latent-vector encoder. See module docstring."""

    def __init__(
        self,
        compression_ratio: int,
        max_length: int,
        encoder_model,
        encoder_tokenizer,
        train_encoder: bool,
        max_encode_batch_size: int = 0,
        pooling: str = "mean",
        encoder_mask_type: str = "causal",
        encoder_window_size: int = 1024,
        boundary_overlap: int = 0,
        accelerator=None,
    ):
        super().__init__()
        if pooling not in ("eos", "mean", "concat"):
            raise ValueError(
                f"pooling must be one of eos/mean/concat, got {pooling!r}"
            )
        if encoder_window_size % compression_ratio != 0:
            raise ValueError(
                f"encoder_window_size ({encoder_window_size}) must be a "
                f"multiple of compression_ratio ({compression_ratio})"
            )

        self.compression_ratio = compression_ratio
        self.encoder_window_size = encoder_window_size
        self.chunks_per_window = encoder_window_size // compression_ratio
        self.pooling = pooling
        self.boundary_overlap = boundary_overlap
        self.max_length = max_length

        self.embed_tokenizer = encoder_tokenizer
        self.embed_model = encoder_model
        self.train_encoder = train_encoder
        self.pad_token_id = encoder_tokenizer.pad_token_id

        self.max_encode_batch_size = max(0, int(max_encode_batch_size))
        self.accelerator = accelerator

        self.is_causal = encoder_mask_type == "causal"
        self._set_attention_mask_type(encoder_mask_type)

        print(
            f"Encoder: pooling={pooling}, compression_ratio={compression_ratio}, "
            f"W={encoder_window_size}, boundary_overlap={boundary_overlap}, "
            f"mask={encoder_mask_type}"
        )

    def _set_attention_mask_type(self, mask_type: str):
        is_causal = mask_type == "causal"
        for module in self.embed_model.modules():
            if hasattr(module, "is_causal"):
                module.is_causal = is_causal

    @property
    def embedding_dim(self) -> int:
        h = self.embed_model.config.hidden_size
        return self.compression_ratio * h if self.pooling == "concat" else h

    def setup_accelerator(self, accelerator):
        self.accelerator = accelerator

    # --- Forward ---------------------------------------------------------

    def forward(self, memory_token_ids: List[List[int]]) -> List[torch.Tensor]:
        """Encode pre-tokenized samples into latent vectors.

        Returns a list of tensors, one per sample, each of shape
        ``[ceil(T/compression_ratio), embedding_dim]``.
        """
        device = next(self.embed_model.parameters()).device

        # 1) Tile each sample into windows.
        # Each window: (sample_idx, full_seq, pool_start, pool_len, marker_positions, num_latents)
        #   full_seq         encoder input for this window
        #   pool_start       index into full_seq where the pool region begins (after left overlap)
        #   pool_len         length of the pool region in full_seq (excludes EOS markers, used by mean/concat)
        #   marker_positions absolute indices into full_seq of EOS markers (eos only; [] otherwise)
        #   num_latents      number of latent vectors this window contributes
        windows: List[Tuple[int, List[int], int, int, List[int], int]] = []
        empty_samples: List[int] = []

        for sample_idx, token_ids in enumerate(memory_token_ids):
            token_ids = self._to_list(token_ids)
            if not token_ids:
                empty_samples.append(sample_idx)
                continue

            for offset in range(0, len(token_ids), self.encoder_window_size):
                pool_tokens = token_ids[offset : offset + self.encoder_window_size]
                pool_len = len(pool_tokens)
                num_latents = max(1, math.ceil(pool_len / self.compression_ratio))

                # Boundary-overlap context (no markers inside it).
                left_ctx_start = max(0, offset - self.boundary_overlap)
                left_ctx = token_ids[left_ctx_start:offset]
                right_overlap = 0 if self.is_causal else self.boundary_overlap
                right_ctx_end = min(
                    len(token_ids), offset + pool_len + right_overlap
                )
                right_ctx = token_ids[offset + pool_len : right_ctx_end]

                full_seq, pool_start, marker_positions = self._build_window_input(
                    left_ctx, pool_tokens, right_ctx
                )
                windows.append(
                    (sample_idx, full_seq, pool_start, pool_len, marker_positions, num_latents)
                )

        if not windows:
            return [
                torch.zeros(1, self.embedding_dim, device=device)
                for _ in memory_token_ids
            ]

        # 2) Pack windows into a left-padded batch tensor.
        max_seq_len = max(len(w[1]) for w in windows)
        total = len(windows)
        input_ids = torch.full(
            (total, max_seq_len), self.pad_token_id, dtype=torch.long, device=device
        )
        attention_mask = torch.zeros(
            (total, max_seq_len), dtype=torch.long, device=device
        )
        position_ids = torch.zeros(
            (total, max_seq_len), dtype=torch.long, device=device
        )
        for i, (_, full_seq, _, _, _, _) in enumerate(windows):
            seq_len = len(full_seq)
            start = max_seq_len - seq_len  # left-padded
            input_ids[i, start:] = torch.tensor(
                full_seq, dtype=torch.long, device=device
            )
            attention_mask[i, start:] = 1
            position_ids[i, start:] = torch.arange(
                seq_len, device=device, dtype=torch.long
            )

        # 3) Encode (optionally mini-batched, with FSDP sync if distributed).
        hidden_states = self._batched_encode(
            input_ids, attention_mask, position_ids, total, device
        )

        # 4) Pool per window.
        per_window_embs: List[Tuple[int, torch.Tensor]] = []
        hidden_dim = self.embed_model.config.hidden_size
        for i, (sample_idx, full_seq, pool_start, pool_len, marker_positions, _) in enumerate(
            windows
        ):
            seq_len = len(full_seq)
            left_pad = max_seq_len - seq_len
            hs = hidden_states[i, left_pad : left_pad + seq_len]  # [seq_len, H]

            if self.pooling == "eos":
                idx = torch.tensor(marker_positions, dtype=torch.long, device=hs.device)
                emb = hs.index_select(0, idx)  # [num_latents, H]
            else:
                pool_hs = hs[pool_start : pool_start + pool_len]  # [pool_len, H]
                emb = self._pool_groupwise(pool_hs, pool_len, hidden_dim)

            per_window_embs.append((sample_idx, emb))

        # 5) Regroup by sample.
        by_sample: dict[int, List[torch.Tensor]] = defaultdict(list)
        for sample_idx, emb in per_window_embs:
            by_sample[sample_idx].append(emb)

        out_dim = self.embedding_dim
        result: List[torch.Tensor] = []
        empty_set = set(empty_samples)
        for sample_idx in range(len(memory_token_ids)):
            if sample_idx in empty_set:
                result.append(
                    torch.zeros(1, out_dim, device=device, dtype=hidden_states.dtype)
                )
            else:
                result.append(torch.cat(by_sample[sample_idx], dim=0))
        return result

    # --- Window construction --------------------------------------------

    def _build_window_input(
        self, left_ctx: List[int], pool_tokens: List[int], right_ctx: List[int]
    ) -> Tuple[List[int], int, List[int]]:
        """Return (full_seq, pool_start, marker_positions) for one window.

        For ``pooling == "eos"`` ``num_latents`` EOS markers are appended at
        the end of the window (after any right-side boundary overlap). For
        ``mean``/``concat`` no markers are inserted; ``marker_positions`` is
        empty.
        """
        pool_len = len(pool_tokens)
        if self.pooling != "eos":
            full_seq = left_ctx + pool_tokens + right_ctx
            return full_seq, len(left_ctx), []

        num_latents = max(1, math.ceil(pool_len / self.compression_ratio))
        full_seq = (
            left_ctx
            + pool_tokens
            + right_ctx
            + [self.pad_token_id] * num_latents
        )
        markers = list(range(len(full_seq) - num_latents, len(full_seq)))
        return full_seq, len(left_ctx), markers

    # --- Group-wise pooling for mean / concat ---------------------------

    def _pool_groupwise(
        self, pool_hs: torch.Tensor, pool_len: int, hidden_dim: int
    ) -> torch.Tensor:
        """Split ``pool_hs`` into groups of ``compression_ratio`` along dim 0 and
        reduce each group to one latent vector by mean or concat."""
        groups: List[torch.Tensor] = []
        target_concat_width = self.compression_ratio * hidden_dim
        for start in range(0, pool_len, self.compression_ratio):
            end = min(start + self.compression_ratio, pool_len)
            g = pool_hs[start:end]  # [<=compression_ratio, H]
            if self.pooling == "concat":
                flat = g.reshape(-1)
                if flat.shape[0] < target_concat_width:
                    pad = torch.zeros(
                        target_concat_width - flat.shape[0],
                        device=flat.device,
                        dtype=flat.dtype,
                    )
                    flat = torch.cat([flat, pad], dim=0)
                groups.append(flat)
            else:  # mean
                groups.append(g.mean(dim=0))
        return torch.stack(groups, dim=0)  # [num_latents, out_dim]

    # --- Batched encoder forward (with FSDP sync) -----------------------

    def _batched_encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        total: int,
        device,
    ) -> torch.Tensor:
        needs_batching = (
            self.max_encode_batch_size > 0 and total > self.max_encode_batch_size
        )
        num_batches = (
            (total + self.max_encode_batch_size - 1) // self.max_encode_batch_size
            if needs_batching
            else 1
        )

        if self.accelerator is not None and self.accelerator.num_processes > 1:
            import torch.distributed as dist

            t = torch.tensor([num_batches], device=device, dtype=torch.int64)
            dist.all_reduce(t, op=dist.ReduceOp.MAX)
            sync_batches = int(t.item())
        else:
            sync_batches = num_batches

        if sync_batches == 1:
            return self._forward_one(input_ids, attention_mask, position_ids)

        max_seq_len = input_ids.shape[1]
        chunks: List[torch.Tensor] = []
        for b in range(sync_batches):
            s = b * self.max_encode_batch_size
            e = min(s + self.max_encode_batch_size, total)
            if s >= total:
                # Dummy forward to keep FSDP ranks in sync.
                dummy = torch.full(
                    (1, max_seq_len),
                    self.pad_token_id,
                    dtype=torch.long,
                    device=device,
                )
                self._forward_one(dummy, torch.zeros_like(dummy), torch.zeros_like(dummy))
            else:
                chunks.append(
                    self._forward_one(
                        input_ids[s:e], attention_mask[s:e], position_ids[s:e]
                    )
                )
        return torch.cat(chunks, dim=0)

    def _forward_one(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        if self.train_encoder:
            out = self.embed_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )
        else:
            with torch.no_grad():
                out = self.embed_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                )
        return out.last_hidden_state

    # --- Helpers ---------------------------------------------------------

    @staticmethod
    def _to_list(token_ids) -> List[int]:
        if torch.is_tensor(token_ids):
            return token_ids.tolist()
        return list(token_ids)
