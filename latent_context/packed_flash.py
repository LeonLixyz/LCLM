"""
Packed Attention implementation for Qwen3 using FlashAttention varlen.
Uses flash_attn_varlen_func for efficient variable-length sequence attention.
"""
import threading
import torch
from typing import List, Optional, Tuple
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3Attention,
    apply_rotary_pos_emb,
)
from flash_attn import flash_attn_varlen_func


# Thread-local context for passing sample_lens to attention layers
# Keep a global fallback for checkpoint recompute threads.
_packed_context = threading.local()
_packed_sample_lens_global: Optional[List[List[int]]] = None


def set_packed_sample_lens(sample_lens: List[List[int]]):
    """Set sample_lens for current packed forward pass."""
    _packed_context.sample_lens = sample_lens
    global _packed_sample_lens_global
    _packed_sample_lens_global = sample_lens


def get_packed_sample_lens() -> Optional[List[List[int]]]:
    """Get sample_lens for current packed forward pass."""
    sample_lens = getattr(_packed_context, 'sample_lens', None)
    if sample_lens is None:
        return _packed_sample_lens_global
    return sample_lens


def clear_packed_sample_lens():
    """Clear sample_lens after forward pass."""
    _packed_context.sample_lens = None
    global _packed_sample_lens_global
    _packed_sample_lens_global = None


class Qwen3PackedFlashAttention(Qwen3Attention):
    """
    Packed sequence attention for Qwen3 using FlashAttention varlen.
    Inherits from Qwen3Attention to reuse all projection layers and norms.
    """

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[tuple] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        sample_lens = get_packed_sample_lens()

        if sample_lens is not None and self.training:
            return self._forward_flash_varlen(
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                sample_lens=sample_lens,
            )

        return super().forward(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            cache_position=cache_position,
            **kwargs,
        )

    def _forward_flash_varlen(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        sample_lens: List[List[int]],
    ) -> Tuple[torch.Tensor, None]:
        bsz, q_len, _ = hidden_states.size()
        device = hidden_states.device
        original_dtype = hidden_states.dtype

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.config.num_attention_heads, self.head_dim)
        key_states = key_states.view(bsz, q_len, self.config.num_key_value_heads, self.head_dim)
        value_states = value_states.view(bsz, q_len, self.config.num_key_value_heads, self.head_dim)

        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        if position_embeddings is None:
            raise ValueError("position_embeddings must be provided")

        cos, sin = position_embeddings
        cos = cos.to(query_states.dtype)
        sin = sin.to(key_states.dtype)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        compute_dtype = query_states.dtype
        if compute_dtype not in (torch.float16, torch.bfloat16):
            compute_dtype = torch.bfloat16
            query_states = query_states.to(compute_dtype)
            key_states = key_states.to(compute_dtype)
            value_states = value_states.to(compute_dtype)

        # flash_attn handles GQA natively - no need to expand K/V

        attn_outputs = []

        for b in range(bsz):
            batch_sample_lens = sample_lens[b] if b < len(sample_lens) else sample_lens[0]
            actual_len = sum(batch_sample_lens)

            q = query_states[b, :, :actual_len, :].transpose(0, 1).contiguous()
            k = key_states[b, :, :actual_len, :].transpose(0, 1).contiguous()
            v = value_states[b, :, :actual_len, :].transpose(0, 1).contiguous()

            cu_seqlens = torch.zeros(len(batch_sample_lens) + 1, dtype=torch.int32, device=device)
            cu_seqlens[1:] = torch.cumsum(torch.tensor(batch_sample_lens, dtype=torch.int32, device=device), dim=0)

            max_seqlen = max(batch_sample_lens)

            attn_out = flash_attn_varlen_func(
                q=q, k=k, v=v,
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_k=cu_seqlens,
                max_seqlen_q=max_seqlen,
                max_seqlen_k=max_seqlen,
                causal=True,
                softmax_scale=self.scaling,
            )

            attn_out = attn_out.reshape(actual_len, -1)

            if actual_len < q_len:
                pad = torch.zeros(q_len - actual_len, attn_out.size(-1), dtype=attn_out.dtype, device=device)
                attn_out = torch.cat([attn_out, pad], dim=0)

            attn_outputs.append(attn_out)

        attn_output = torch.stack(attn_outputs, dim=0)

        if attn_output.dtype != original_dtype:
            attn_output = attn_output.to(original_dtype)

        attn_output = self.o_proj(attn_output)

        return attn_output, None


def replace_with_packed_attention(model):
    """Replace all Qwen3Attention layers with Qwen3PackedFlashAttention.

    Uses __class__ swap to avoid weight copying - just changes the forward method.
    """
    actual_model = model
    if hasattr(model, 'base_model') and hasattr(model.base_model, 'model'):
        actual_model = model.base_model.model
        if hasattr(actual_model, 'model'):
            actual_model = actual_model.model
    elif hasattr(model, 'model'):
        actual_model = model.model

    if not hasattr(actual_model, 'layers'):
        raise ValueError(f"Model must have 'layers' attribute. Got: {type(actual_model)}")

    for layer in actual_model.layers:
        if hasattr(layer, 'self_attn'):
            attn = layer.self_attn
            if not isinstance(attn, Qwen3PackedFlashAttention):
                # Swap class - keeps all weights, just changes methods
                attn.__class__ = Qwen3PackedFlashAttention

    return model
