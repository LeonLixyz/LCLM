"""
Clean Code LLaVA implementation with simplified single forward method.
"""
import torch
import torch.nn as nn
from transformers import AutoTokenizer, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from typing import Optional, Union, Tuple, List
from transformers.cache_utils import Cache

from .encoder import Encoder
from .processor import LCLMProcessor
from .adapter import Adapter

try:
    from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
    _LIGER_FLCE_AVAILABLE = True
except ImportError:
    _LIGER_FLCE_AVAILABLE = False


class LCLM(nn.Module):
    """
    Clean Code LLaVA model that uses the processor for all code handling.

    This model:
    1. Uses processor to expand code placeholders
    2. Uses Encoder to get code embeddings
    3. Uses processor to replace placeholder tokens with code embeddings
    4. Passes through the LLM for generation
    """

    def __init__(
        self,
        decoder: PreTrainedModel,
        decoder_tokenizer: AutoTokenizer,
        embed_model: PreTrainedModel,
        embed_tokenizer: AutoTokenizer,
        processor: LCLMProcessor,
        compression_ratio: int = 100,
        max_memory_length: int = 8192,
        train_decoder: bool = True,
        train_encoder: bool = True,
        max_encode_batch_size: int = 0,
        pooling: str = "mean",
        encoder_mask_type: str = "causal",
        encoder_window_size: int = 1024,
        boundary_overlap: int = 0,
        accelerator = None,
        packed_attention_backend: str = None,  # None, "flash", or "flex"
        adapter_type: str = "mlp",  # "mlp" | "mlp_attn" | "attn_mlp"
        num_adapter_layers: int = 1,
        use_fused_ce: bool = False,  # Liger fused linear+CE; slower at small bs, needed only when HF path OOMs (e.g. bs≥4 at packed seq=16384)
    ):
        super().__init__()

        self.decoder = decoder
        self.decoder_tokenizer = decoder_tokenizer
        self.train_decoder = train_decoder
        self.train_encoder = train_encoder
        self.processor = processor
        self.accelerator = accelerator
        self.packed_attention_backend = packed_attention_backend
        # Note: Packed attention is enabled by trainer BEFORE LoRA, not here

        self.encoder = Encoder(
            encoder_model=embed_model,
            encoder_tokenizer=embed_tokenizer,
            compression_ratio=compression_ratio,
            max_length=max_memory_length,
            train_encoder=train_encoder,
            max_encode_batch_size=max_encode_batch_size,
            pooling=pooling,
            encoder_mask_type=encoder_mask_type,
            encoder_window_size=encoder_window_size,
            boundary_overlap=boundary_overlap,
            accelerator=accelerator,
        )
        
        # Adapter to project latents from encoder dim to decoder dim.
        # For pooling="concat" encoder.embedding_dim = compression_ratio * encoder_hidden, so
        # the adapter input dim already accounts for the compression ratio.
        encoder_dim = self.encoder.embedding_dim
        decoder_dim = self.decoder.config.hidden_size
        if adapter_type == "mlp_attn":
            attn_config = decoder.config  # post-MLP attention in decoder space
        elif adapter_type == "attn_mlp":
            attn_config = embed_model.config  # pre-MLP attention in encoder space
        else:
            attn_config = None
        self.adapter = Adapter(
            encoder_dim=encoder_dim,
            decoder_dim=decoder_dim,
            adapter_type=adapter_type,
            num_layers=num_adapter_layers,
            attn_config=attn_config,
        )

        # Fused linear cross-entropy (Triton) — avoids materializing the
        # [packed_seq, vocab] fp32 logits tensor, which OOMs at 16k packed seq.
        # Applied in forward() when labels are provided and we're training.
        # Toggle via `use_fused_ce` in the config (default: True).
        self.use_fused_ce = use_fused_ce
        self._liger_flce = (
            LigerFusedLinearCrossEntropyLoss()
            if (use_fused_ce and _LIGER_FLCE_AVAILABLE)
            else None
        )
        if use_fused_ce and not _LIGER_FLCE_AVAILABLE:
            print("⚠ use_fused_ce=True but liger_kernel not importable; falling back to HF CE path")

    def setup_accelerator(self, accelerator):
        """Set the accelerator after initialization for both model and encoder."""
        self.accelerator = accelerator

    def _enable_packed_attention(self, backend: str):
        """Replace attention layers with packed attention.

        Args:
            backend: "flash" for FlashAttention varlen, "flex" for PyTorch flex_attention
        """
        try:
            if backend == "flash":
                from .packed_flash import replace_with_packed_attention, Qwen3PackedFlashAttention as PackedAttentionClass
            elif backend == "flex":
                from .packed_flex import replace_with_packed_attention, Qwen3PackedFlexAttention as PackedAttentionClass
            else:
                raise ValueError(f"Unknown packed attention backend: {backend}. Use 'flash' or 'flex'.")

            # Check if packed attention is already applied
            check_model = self.decoder
            if hasattr(check_model, 'base_model') and hasattr(check_model.base_model, 'model'):
                check_model = check_model.base_model.model
            if hasattr(check_model, 'model'):
                check_model = check_model.model

            if hasattr(check_model, 'layers') and len(check_model.layers) > 0:
                if isinstance(check_model.layers[0].self_attn, PackedAttentionClass):
                    print(f"✓ Packed attention ({backend}) already enabled (skipping)")
                    return

            replace_with_packed_attention(self.decoder)
            print(f"✓ Enabled packed attention (backend: {backend})")

        except Exception as e:
            print(f"✗ Error enabling packed attention: {e}")
            raise
    
    def _process_latent_embeddings(
        self,
        memory_token_ids: Union[List[List[int]], List[List[List[int]]]]
    ) -> Union[List[torch.Tensor], List[List[torch.Tensor]]]:
        """
        Process pre-tokenized code into embeddings using the chunker.

        Args:
            memory_token_ids: List of pre-tokenized code (embed token IDs)

        Returns:
            List of code embeddings, one tensor per code sample: [num_chunks, decoder_dim]
        """
        if not memory_token_ids:
            raise ValueError("memory_token_ids cannot be empty - every sequence must have code embeddings")
        # Multi-segment per-sample (List[List[List[int]]])
        if isinstance(memory_token_ids[0], list) and len(memory_token_ids[0]) > 0 and isinstance(memory_token_ids[0][0], list):
            # Process each batch item SEPARATELY to avoid cross-batch interference
            # This ensures B=1 and B=2 produce identical embeddings for the same code
            regrouped: List[List[torch.Tensor]] = []

            for batch_idx, segs in enumerate(memory_token_ids):  # type: ignore[index]
                if not segs:
                    regrouped.append([])
                    continue

                # Process this batch item's segments
                batch_embeds = self.encoder(segs)
                batch_projected = self._project_latent_embeddings_synced(batch_embeds)
                regrouped.append(batch_projected)

            return regrouped
        # Single-segment per-sample (List[List[int]])
        chunk_embeddings_list = self.encoder(memory_token_ids)  # type: ignore[arg-type]
        projected_embeddings_list = self._project_latent_embeddings_synced(chunk_embeddings_list)
        return projected_embeddings_list

    # if we wrap the adapter with FSDP, we need to the operations across all ranks as well. Here we just concat all embeddings and do 1 forward pass as the memory would be cheap.
    def _project_latent_embeddings_synced(
        self,
        chunk_embeddings_list: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """
        Project code embeddings through the adapter.

        When adapter attention is disabled: concatenates all and runs MLP (element-wise).
        When adapter attention is enabled: batches segments with padding so each
        segment's chunks only attend to themselves, not to other segments.
        """
        if not chunk_embeddings_list:
            return []

        # Record chunk sizes for splitting later
        chunk_sizes = [emb.shape[0] for emb in chunk_embeddings_list]

        # Ensure dtype matches adapter
        adapter_dtype = self.adapter.fc1.weight.dtype

        # When the adapter has attention layers, batch segments with padding.
        if self.adapter.adapter_type != "mlp":
            # Pad all segments to max length and batch
            max_chunks = max(chunk_sizes)
            batch_size = len(chunk_embeddings_list)
            embed_dim = chunk_embeddings_list[0].shape[1]
            device = chunk_embeddings_list[0].device

            # Create padded batch tensor and attention mask
            padded_embeds = torch.zeros(batch_size, max_chunks, embed_dim, device=device, dtype=adapter_dtype)
            attention_mask = torch.zeros(batch_size, max_chunks, device=device, dtype=torch.long)

            for i, (emb, size) in enumerate(zip(chunk_embeddings_list, chunk_sizes)):
                if emb.dtype != adapter_dtype:
                    emb = emb.to(adapter_dtype)
                padded_embeds[i, :size] = emb
                attention_mask[i, :size] = 1

            # Batched adapter call with attention mask
            projected_padded = self.adapter(padded_embeds, attention_mask=attention_mask)

            # Extract valid (non-padded) outputs for each segment
            projected_embeddings_list: List[torch.Tensor] = []
            for i, size in enumerate(chunk_sizes):
                projected_embeddings_list.append(projected_padded[i, :size])

            return projected_embeddings_list

        else:
            # No attention: concatenate all and run MLP (element-wise, no cross-segment interaction)
            all_embeds = torch.cat(chunk_embeddings_list, dim=0)  # [total_chunks, embed_dim]

            if all_embeds.dtype != adapter_dtype:
                all_embeds = all_embeds.to(adapter_dtype)

            # Single batched adapter call - all ranks call once
            projected_all = self.adapter(all_embeds)  # [total_chunks, decoder_dim]

            # Split back into per-segment chunks
            projected_embeddings_list = []
            idx = 0
            for size in chunk_sizes:
                projected_embeddings_list.append(projected_all[idx:idx + size])
                idx += size

            return projected_embeddings_list

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        memory_token_ids: Optional[List[List[int]]] = None,
        memory_positions: Optional[List[Tuple[int, int]]] = None,
        latent_counts: Optional[List[int]] = None,
        sample_lens: Optional[List[int]] = None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        """
        Forward pass using the processor approach.

        When memory_token_ids are provided, the processor should have already:
        1. Expanded the <|memory|> placeholders to <|memory_start|> + N <|memory|> + <|memory_end|>
        2. Tokenized the expanded prompt
        3. Provided memory_positions and latent_counts
        4. Pre-tokenized codes with embed_tokenizer (memory_token_ids)

        This method then:
        1. Gets embeddings for the tokenized input
        2. Processes pre-tokenized code through the chunker (no tokenization needed)
        3. Uses processor to replace placeholder embeddings with code embeddings
        4. Passes through LLM
        """

        device = next(self.decoder.parameters()).device

        input_ids = input_ids.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        if labels is not None:
            labels = labels.to(device)
        # Normalize to nested structures
        if isinstance(memory_token_ids, list) and len(memory_token_ids) > 0 and isinstance(memory_token_ids[0], list):
            # Check if it's List[List[int]] (single sample) or List[List[List[int]]] (batch)
            if len(memory_token_ids[0]) > 0 and isinstance(memory_token_ids[0][0], list):
                codes_nested: List[List[List[int]]] = memory_token_ids  # type: ignore[assignment]
            else:
                # Single sample: List[List[int]] -> wrap in batch dimension
                codes_nested = [memory_token_ids]  # type: ignore[list-item]
        elif isinstance(memory_token_ids, list):
            codes_nested = [[c] for c in memory_token_ids]  # type: ignore[list-item]
        else:
            codes_nested = [[] for _ in range(input_ids.size(0))]

        if isinstance(memory_positions, list) and len(memory_positions) > 0 and isinstance(memory_positions[0], list):
            pos_nested: List[List[Tuple[int, int]]] = memory_positions  # type: ignore[assignment]
        elif isinstance(memory_positions, list):
            pos_nested = [memory_positions]
        else:
            pos_nested = [[] for _ in range(input_ids.size(0))]

        if isinstance(latent_counts, list) and len(latent_counts) > 0 and isinstance(latent_counts[0], list):
            counts_nested: List[List[int]] = latent_counts  # type: ignore[assignment]
        elif isinstance(latent_counts, list):
            counts_nested = [latent_counts]
        else:
            counts_nested = [[] for _ in range(input_ids.size(0))]

        # Create new tensor and copy embeddings (avoids in-place modification issues with FSDP)
        # Cast to compute dtype to avoid LayerNorm dtype mismatch with FSDP mixed precision
        text_embeds = self.decoder.get_input_embeddings()(input_ids)
        compute_dtype = next(self.decoder.parameters()).dtype
        text_embeds = text_embeds.to(compute_dtype)
        inputs_embeds = text_embeds.new_zeros(text_embeds.shape)
        inputs_embeds.copy_(text_embeds)

        # Process code through chunker and adapter
        latent_embeddings_nested = self._process_latent_embeddings(codes_nested)

        # Store memory_token_ids and encoder reference for debugging in processor
        self.processor._debug_memory_token_ids_nested = codes_nested
        self.processor._debug_encoder = self.encoder
        self.processor._debug_sample_lens = sample_lens  # For extracting individual samples from packed batch

        combined_embeds = self.processor.replace_memory_tokens_with_embeddings(
            inputs_embeds=inputs_embeds,
            latent_embeddings=latent_embeddings_nested,  # type: ignore[arg-type]
            memory_positions=pos_nested,
            latent_counts=counts_nested,
            input_ids=input_ids,
        )

        # Clear debug references
        self.processor._debug_memory_token_ids_nested = None
        self.processor._debug_encoder = None
        self.processor._debug_sample_lens = None
        
        # Verify shape consistency
        assert combined_embeds.shape[0] == input_ids.shape[0], f"Batch size mismatch: {combined_embeds.shape[0]} vs {input_ids.shape[0]}"
        assert combined_embeds.shape[1] == input_ids.shape[1], f"Seq len mismatch: {combined_embeds.shape[1]} vs {input_ids.shape[1]}"
        assert combined_embeds.dim() == 3, f"Wrong dims: {combined_embeds.shape}"

        # Create block mask and position IDs for packed sequences if needed
        position_ids = None
        packed_sample_lens_for_flash = None  # For FlashAttention varlen

        if self.packed_attention_backend and sample_lens is not None and self.training:
            num_heads = self.decoder.config.num_attention_heads
            actual_seq_len = combined_embeds.shape[1]

            if labels is not None and labels.shape[1] != actual_seq_len:
                raise ValueError(f"Labels shape {labels.shape} doesn't match embeds shape {combined_embeds.shape}")

            if self.packed_attention_backend == "flash":
                from .packed_flash import set_packed_sample_lens
                # Handle both List[int] and List[List[int]]
                if isinstance(sample_lens[0], list):
                    packed_sample_lens_for_flash = sample_lens
                else:
                    packed_sample_lens_for_flash = [sample_lens]
                set_packed_sample_lens(packed_sample_lens_for_flash)

            elif self.packed_attention_backend == "flex":
                from data.packing_utils import create_block_mask_for_packed, create_block_mask_for_packed_batch
                if isinstance(sample_lens[0], list):
                    attention_mask = create_block_mask_for_packed_batch(
                        batch_sample_lens=sample_lens,
                        num_heads=num_heads,
                        device=device,
                        block_size=128,
                        total_len=actual_seq_len,
                    )
                else:
                    attention_mask = create_block_mask_for_packed(
                        sample_lens=sample_lens,
                        num_heads=num_heads,
                        device=device,
                        block_size=128,
                        total_len=actual_seq_len,
                    )

        # Forward through LLM
        # Note: Don't clear packed_sample_lens here - gradient checkpointing needs it
        # during backward pass recomputation. It gets overwritten on next forward anyway.

        # Fast path: when training with labels, bypass lm_head + HF CE so the
        # [packed_seq, vocab] logits tensor is never materialized. Uses
        # liger's fused linear+CE triton kernel.
        if (
            labels is not None
            and self.training
            and self._liger_flce is not None
            and not output_hidden_states
            and not kwargs.get("use_cache", False)
        ):
            backbone = getattr(self.decoder, "model", None)
            lm_head = getattr(self.decoder, "lm_head", None)
            if backbone is not None and lm_head is not None:
                backbone_out = backbone(
                    input_ids=None,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=None,
                    inputs_embeds=combined_embeds,
                    output_hidden_states=False,
                    **kwargs,
                )
                hidden_states = backbone_out.last_hidden_state  # [B, S, H]

                # Shift for next-token prediction (same convention as HF Qwen3ForCausalLM)
                shift_hidden = hidden_states[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                flat_hidden = shift_hidden.view(-1, shift_hidden.size(-1))
                flat_labels = shift_labels.view(-1).to(flat_hidden.device)

                loss = self._liger_flce(
                    lm_head.weight,
                    flat_hidden,
                    flat_labels,
                    getattr(lm_head, "bias", None),
                )

                return CausalLMOutputWithPast(
                    loss=loss,
                    logits=None,
                    past_key_values=None,
                    hidden_states=None,
                    attentions=None,
                )

        # Standard path (generation, eval, or non-liger fallback)
        output = self.decoder(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=combined_embeds,
            labels=labels,
            output_hidden_states=output_hidden_states,
            **kwargs,
        )

        return output

    def generate(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        memory_token_ids: Optional[List[List[int]]] = None,
        memory_positions: Optional[List[Tuple[int, int]]] = None,
        latent_counts: Optional[List[int]] = None,
        **generation_kwargs
    ):
        """
        Generate text with code context using the processor.

        Args:
            input_ids: Tokenized input IDs
            attention_mask: Attention mask
            memory_token_ids: List of pre-tokenized code (embed token IDs)
            memory_positions: Positions of code placeholders in input
            latent_counts: Number of chunks per code segment
            **generation_kwargs: Additional generation arguments

        Returns:
            Generated token IDs
        """

        device = next(self.decoder.parameters()).device

        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        # Normalize to nested
        if isinstance(memory_token_ids, list) and len(memory_token_ids) > 0 and isinstance(memory_token_ids[0], list):
            if len(memory_token_ids[0]) > 0 and isinstance(memory_token_ids[0][0], list):
                codes_nested: List[List[List[int]]] = memory_token_ids  # type: ignore[assignment]
            else:
                codes_nested = [memory_token_ids]  # type: ignore[assignment]
        else:
            codes_nested = [[] for _ in range(input_ids.size(0))]

        if isinstance(memory_positions, list) and len(memory_positions) > 0 and isinstance(memory_positions[0], list):
            pos_nested: List[List[Tuple[int, int]]] = memory_positions  # type: ignore[assignment]
        elif isinstance(memory_positions, list):
            pos_nested = [memory_positions]
        else:
            pos_nested = [[] for _ in range(input_ids.size(0))]

        if isinstance(latent_counts, list) and len(latent_counts) > 0 and isinstance(latent_counts[0], list):
            counts_nested: List[List[int]] = latent_counts  # type: ignore[assignment]
        elif isinstance(latent_counts, list):
            counts_nested = [latent_counts]
        else:
            counts_nested = [[] for _ in range(input_ids.size(0))]

        # Create new tensor and copy embeddings (avoids in-place modification issues)
        # Cast to compute dtype to avoid LayerNorm dtype mismatch
        compute_dtype = next(self.decoder.parameters()).dtype
        text_embeds = self.decoder.get_input_embeddings()(input_ids).to(compute_dtype)
        inputs_embeds = text_embeds.new_zeros(text_embeds.shape)
        inputs_embeds.copy_(text_embeds)
        latent_embeddings_nested = self._process_latent_embeddings(codes_nested)

        combined_embeds = self.processor.replace_memory_tokens_with_embeddings(
            inputs_embeds=inputs_embeds,
            latent_embeddings=latent_embeddings_nested,  # type: ignore[arg-type]
            memory_positions=pos_nested,
            latent_counts=counts_nested,
            input_ids=input_ids,
        )
        
        # Generate using the LLM
        return self.decoder.generate(
            inputs_embeds=combined_embeds,
            attention_mask=attention_mask,
            **generation_kwargs
        )
