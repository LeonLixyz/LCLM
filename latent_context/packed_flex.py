"""
Packed Attention implementation for Qwen3 using PyTorch flex_attention.
Uses BlockMask for efficient block-sparse attention patterns.
"""
import torch
from typing import Optional, Tuple
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3Attention,
    apply_rotary_pos_emb,
)
from torch.nn.attention.flex_attention import flex_attention, BlockMask

# Compile flex_attention for performance
flex_attention = torch.compile(flex_attention)


class Qwen3PackedFlexAttention(Qwen3Attention):
    """
    Packed sequence attention for Qwen3 using flex_attention.
    Inherits from Qwen3Attention to reuse all projection layers and norms.
    Uses BlockMask for efficient block-sparse attention patterns.
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
        # Use flex attention with BlockMask during training
        if isinstance(attention_mask, BlockMask) and self.training:
            return self._forward_flex(
                hidden_states=hidden_states,
                block_mask=attention_mask,
                position_embeddings=position_embeddings,
            )

        return super().forward(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            cache_position=cache_position,
            **kwargs,
        )

    def _forward_flex(
        self,
        hidden_states: torch.Tensor,
        block_mask: BlockMask,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    ) -> Tuple[torch.Tensor, None]:
        bsz, q_len, _ = hidden_states.size()

        # Match Qwen3 order: proj -> view -> norm -> transpose
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

        # flex_attention handles GQA natively with enable_gqa=True
        attn_output = flex_attention(
            query_states,
            key_states,
            value_states,
            block_mask=block_mask,
            enable_gqa=(self.num_key_value_groups > 1),
            scale=self.scaling,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, -1)

        attn_output = self.o_proj(attn_output)

        return attn_output, None


def replace_with_packed_attention(model):
    """Replace all Qwen3Attention layers with Qwen3PackedFlexAttention.

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
            if not isinstance(attn, Qwen3PackedFlexAttention):
                # Swap class - keeps all weights, just changes methods
                attn.__class__ = Qwen3PackedFlexAttention

    return model
