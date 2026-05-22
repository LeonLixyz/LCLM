"""LCLM Adapter — projects encoder latents into decoder embedding space.

Defined as an ``nn.Module`` so it can be wrapped by FSDP.
"""
import torch
import torch.nn as nn
from typing import Optional

from transformers import PretrainedConfig


ADAPTER_TYPES = ("mlp", "mlp_attn", "attn_mlp")


class Adapter(nn.Module):
    """Encoder-dim → decoder-dim projection, optionally with attention layers.

    Three flavors, selected by ``adapter_type``:

    - ``"mlp"``      RMSNorm → Linear(encoder_dim → decoder_dim) → GELU →
                     Linear(decoder_dim → decoder_dim). No attention.
    - ``"mlp_attn"`` MLP first (as above), then N decoder-dim attention layers.
                     Pass ``attn_config = decoder.config`` for layer sizing.
    - ``"attn_mlp"`` N encoder-dim attention layers first, then MLP.
                     Pass ``attn_config = encoder.config`` for layer sizing.
    """

    def __init__(
        self,
        encoder_dim: int,
        decoder_dim: int,
        adapter_type: str = "mlp",
        num_layers: int = 1,
        attn_config: Optional[PretrainedConfig] = None,
    ):
        super().__init__()
        if adapter_type not in ADAPTER_TYPES:
            raise ValueError(
                f"adapter_type must be one of {ADAPTER_TYPES}, got {adapter_type!r}"
            )
        if adapter_type != "mlp" and attn_config is None:
            raise ValueError(
                f"adapter_type={adapter_type!r} requires attn_config "
                f"(pass decoder.config for mlp_attn or encoder.config for attn_mlp)"
            )

        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim
        self.adapter_type = adapter_type
        self.num_layers = num_layers

        # MLP path (always present).
        self.norm = nn.RMSNorm(encoder_dim, eps=1e-6)
        self.fc1 = nn.Linear(encoder_dim, decoder_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(decoder_dim, decoder_dim)
        for m in (self.fc1, self.fc2):
            if m.bias is not None:
                nn.init.zeros_(m.bias)

        # Optional attention path.
        if adapter_type != "mlp":
            self.attention_layers = nn.ModuleList(
                [AttentionLayer(attn_config, layer_idx=i) for i in range(num_layers)]
            )

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """``x``: [num_chunks, encoder_dim] or [batch, num_chunks, encoder_dim]."""
        needs_squeeze = False
        if self.adapter_type != "mlp" and x.dim() == 2:
            x = x.unsqueeze(0)
            if attention_mask is not None:
                attention_mask = attention_mask.unsqueeze(0)
            needs_squeeze = True

        if self.adapter_type == "attn_mlp":
            for layer in self.attention_layers:
                x = layer(x, attention_mask=attention_mask)
            x = self.fc2(self.act(self.fc1(self.norm(x))))
        else:  # "mlp" or "mlp_attn"
            x = self.fc2(self.act(self.fc1(self.norm(x))))
            if self.adapter_type == "mlp_attn":
                for layer in self.attention_layers:
                    x = layer(x, attention_mask=attention_mask)

        return x.squeeze(0) if needs_squeeze else x


class AttentionLayer(nn.Module):
    """
    Single transformer layer using Qwen3 architecture with RoPE.

    Uses Qwen3DecoderLayer and Qwen3RotaryEmbedding from the provided config.
    Initialized with random weights.
    Uses bidirectional attention (no causal mask).
    Supports Flash Attention (uses config's _attn_implementation).
    Liger kernel optimizations are applied globally by the trainer if enabled.

    Args:
        config: Model configuration (LLM or embedder config)
        layer_idx: Layer index for positional encoding
    """

    def __init__(
        self,
        config: PretrainedConfig,
        layer_idx: int = 0,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.config = config

        # Ensure _attn_implementation is set (needed by newer transformers)
        if not hasattr(config, '_attn_implementation') or config._attn_implementation is None:
            config._attn_implementation = "eager"

        # Import Qwen3 components from transformers
        # Note: If Liger kernel was applied globally by the trainer, these will be Liger-optimized
        from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer, Qwen3RotaryEmbedding

        # Create decoder layer with layer_idx
        self.decoder_layer = Qwen3DecoderLayer(config, layer_idx=layer_idx)

        # Create rotary embedding for position encoding
        self.rotary_emb = Qwen3RotaryEmbedding(config)

        # Set attention to bidirectional (not causal)
        self.decoder_layer.self_attn.is_causal = False

        # Initialize with random weights
        self._init_weights()

    def _init_weights(self):
        """Initialize all weights randomly."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through the attention layer.

        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_size]
            attention_mask: Optional mask [batch, seq_len] with 1 for valid, 0 for padding

        Returns:
            Output tensor [batch, seq_len, hidden_size]
        """
        batch_size, seq_len, _ = hidden_states.shape
        device = hidden_states.device

        # Flash attention requires bf16/fp16 — cast if needed
        # (mixed precision keeps params in fp32, but flash attn needs reduced precision inputs)
        if hidden_states.dtype == torch.float32:
            hidden_states = hidden_states.to(torch.bfloat16)

        # Create position IDs: [0, 1, 2, ..., seq_len-1]
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)

        # Get rotary position embeddings (cos, sin)
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # Forward through decoder layer
        # Pass 2D attention_mask directly [batch, seq_len] with 1=valid, 0=padding
        # HuggingFace Flash Attention handles cu_seqlens conversion internally
        hidden_states = self.decoder_layer(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
            past_key_values=None,
            use_cache=False,
        )

        return hidden_states
