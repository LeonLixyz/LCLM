"""
Clean Code LLaVA implementation with simplified single forward method.
"""
import torch
import torch.nn as nn
from numbers import Integral
from transformers import AutoTokenizer, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from typing import Any, Optional, Union, Tuple, List
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
        # Set on every forward after canonical metadata validation. The trainer
        # uses this to skip encoder/adapter optimizer state updates when an
        # entire distributed optimizer step contains no compressed regions.
        self._last_local_has_memory = False
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
        self.encoder.setup_accelerator(accelerator)

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
        # Multi-segment per-sample (List[List[List[int]]]). Inspect the complete
        # outer list so a leading uncompressed sample cannot hide later segments.
        if all(self._is_segment_list(item) for item in memory_token_ids):
            normalized = self._normalize_batched_memory_token_ids(
                memory_token_ids, len(memory_token_ids)
            )
            regrouped, _ = self._process_batched_latent_embeddings(
                normalized,
                ensure_participation=(
                    self._distributed_encoder_participation_required()
                    or (self.training and torch.is_grad_enabled())
                ),
            )
            return regrouped
        # Single-segment per-sample (List[List[int]])
        if not all(self._is_token_sequence(item) for item in memory_token_ids):
            raise ValueError(
                "memory_token_ids must have shape [segment][token] or "
                "[batch][segment][token]"
            )
        chunk_embeddings_list = self.encoder(memory_token_ids)  # type: ignore[arg-type]
        projected_embeddings_list = self._project_latent_embeddings_synced(chunk_embeddings_list)
        return projected_embeddings_list

    @staticmethod
    def _is_token_sequence(value: Any) -> bool:
        """Return whether ``value`` is one (possibly empty) 1-D token sequence."""
        if isinstance(value, torch.Tensor):
            return value.dim() == 1
        if not isinstance(value, (list, tuple)):
            return False
        return all(
            not isinstance(token, (list, tuple, torch.Tensor))
            for token in value
        )

    @classmethod
    def _is_segment_list(cls, value: Any) -> bool:
        """Return whether ``value`` is a per-sample list of token sequences."""
        return isinstance(value, (list, tuple)) and all(
            cls._is_token_sequence(segment) for segment in value
        )

    @staticmethod
    def _token_sequence_to_list(value: Any, *, location: str) -> List[int]:
        if isinstance(value, torch.Tensor):
            if value.dim() != 1:
                raise ValueError(
                    f"{location} must be a 1-D token sequence, got tensor shape "
                    f"{tuple(value.shape)}"
                )
            value = value.detach().cpu().tolist()
        elif isinstance(value, tuple):
            value = list(value)

        if not isinstance(value, list):
            raise ValueError(f"{location} must be a token sequence, got {type(value).__name__}")

        result: List[int] = []
        for token_idx, token in enumerate(value):
            if isinstance(token, bool) or not isinstance(token, Integral):
                raise ValueError(
                    f"{location}[{token_idx}] must be an integer token ID, got {token!r}"
                )
            result.append(int(token))
        return result

    @classmethod
    def _normalize_batched_memory_token_ids(
        cls,
        memory_token_ids: Any,
        batch_size: int,
    ) -> List[List[List[int]]]:
        """Normalize memory IDs to ``[batch][segment][token]``.

        Packed data already uses the canonical three-level representation, but
        older single-example callers pass ``[segment][token]``.  A batch whose
        first sample is uncompressed (``[[], [[...]]]``) must not be classified
        by inspecting only that first empty sample.
        """
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if memory_token_ids is None:
            return [[] for _ in range(batch_size)]
        if isinstance(memory_token_ids, torch.Tensor):
            memory_token_ids = memory_token_ids.detach().cpu().tolist()
        if not isinstance(memory_token_ids, (list, tuple)):
            raise ValueError(
                "memory_token_ids must be a list in [batch][segment][token] "
                f"or legacy [segment][token] form, got {type(memory_token_ids).__name__}"
            )

        raw = list(memory_token_ids)
        if not raw:
            return [[] for _ in range(batch_size)]

        # Canonical [batch][segment][token]. Empty per-sample segment lists are
        # intentionally valid and represent examples that bypass compression.
        if len(raw) == batch_size and all(cls._is_segment_list(item) for item in raw):
            batched = [list(item) for item in raw]
        # Legacy single-example [segment][token].
        elif batch_size == 1 and all(cls._is_token_sequence(item) for item in raw):
            batched = [raw]
        # Compatibility form for a multi-example batch with one segment per
        # sample: [batch][token]. An empty token list means no segment.
        elif len(raw) == batch_size and all(cls._is_token_sequence(item) for item in raw):
            batched = [[item] if len(item) > 0 else [] for item in raw]
        else:
            raise ValueError(
                "memory_token_ids has an ambiguous or invalid shape for batch "
                f"size {batch_size}; expected [batch][segment][token]"
            )

        normalized: List[List[List[int]]] = []
        for batch_idx, segments in enumerate(batched):
            normalized_segments: List[List[int]] = []
            for segment_idx, segment in enumerate(segments):
                token_ids = cls._token_sequence_to_list(
                    segment,
                    location=f"memory_token_ids[{batch_idx}][{segment_idx}]",
                )
                if not token_ids:
                    raise ValueError(
                        f"memory_token_ids[{batch_idx}][{segment_idx}] is empty; "
                        "use an empty per-sample segment list to disable compression"
                    )
                normalized_segments.append(token_ids)
            normalized.append(normalized_segments)
        return normalized

    @staticmethod
    def _is_position(value: Any) -> bool:
        return (
            isinstance(value, (list, tuple))
            and len(value) == 2
            and all(
                isinstance(endpoint, Integral) and not isinstance(endpoint, bool)
                for endpoint in value
            )
        )

    @classmethod
    def _normalize_batched_memory_positions(
        cls,
        memory_positions: Any,
        batch_size: int,
    ) -> List[List[Tuple[int, int]]]:
        """Normalize positions to ``[batch][segment](start, end)``."""
        if memory_positions is None:
            return [[] for _ in range(batch_size)]
        if not isinstance(memory_positions, (list, tuple)):
            raise ValueError(
                f"memory_positions must be a list, got {type(memory_positions).__name__}"
            )
        raw = list(memory_positions)
        if not raw:
            return [[] for _ in range(batch_size)]

        is_position_list = lambda value: isinstance(value, (list, tuple)) and all(  # noqa: E731
            cls._is_position(position) for position in value
        )
        if len(raw) == batch_size and all(is_position_list(item) for item in raw):
            batched = [list(item) for item in raw]
        elif batch_size == 1 and all(cls._is_position(item) for item in raw):
            batched = [raw]
        elif len(raw) == batch_size and all(cls._is_position(item) for item in raw):
            batched = [[item] for item in raw]
        else:
            raise ValueError(
                "memory_positions has an ambiguous or invalid shape for batch "
                f"size {batch_size}; expected [batch][segment](start, end)"
            )

        return [
            [(int(start), int(end)) for start, end in positions]
            for positions in batched
        ]

    @staticmethod
    def _is_count(value: Any) -> bool:
        return isinstance(value, Integral) and not isinstance(value, bool)

    @classmethod
    def _normalize_batched_latent_counts(
        cls,
        latent_counts: Any,
        batch_size: int,
    ) -> List[List[int]]:
        """Normalize latent counts to ``[batch][segment]``."""
        if latent_counts is None:
            return [[] for _ in range(batch_size)]
        if not isinstance(latent_counts, (list, tuple)):
            raise ValueError(
                f"latent_counts must be a list, got {type(latent_counts).__name__}"
            )
        raw = list(latent_counts)
        if not raw:
            return [[] for _ in range(batch_size)]

        is_count_list = lambda value: isinstance(value, (list, tuple)) and all(  # noqa: E731
            cls._is_count(count) for count in value
        )
        if len(raw) == batch_size and all(is_count_list(item) for item in raw):
            batched = [list(item) for item in raw]
        elif batch_size == 1 and all(cls._is_count(item) for item in raw):
            batched = [raw]
        elif len(raw) == batch_size and all(cls._is_count(item) for item in raw):
            batched = [[item] for item in raw]
        else:
            raise ValueError(
                "latent_counts has an ambiguous or invalid shape for batch "
                f"size {batch_size}; expected [batch][segment]"
            )

        return [[int(count) for count in counts] for counts in batched]

    @classmethod
    def _normalize_and_validate_memory_batch(
        cls,
        memory_token_ids: Any,
        memory_positions: Any,
        latent_counts: Any,
        batch_size: int,
        sequence_length: Optional[int] = None,
    ) -> Tuple[
        List[List[List[int]]],
        List[List[Tuple[int, int]]],
        List[List[int]],
    ]:
        codes = cls._normalize_batched_memory_token_ids(memory_token_ids, batch_size)
        positions = cls._normalize_batched_memory_positions(memory_positions, batch_size)
        counts = cls._normalize_batched_latent_counts(latent_counts, batch_size)

        for batch_idx, (sample_codes, sample_positions, sample_counts) in enumerate(
            zip(codes, positions, counts)
        ):
            if not (
                len(sample_codes) == len(sample_positions) == len(sample_counts)
            ):
                raise ValueError(
                    f"Batch {batch_idx} has inconsistent memory metadata: "
                    f"{len(sample_codes)} token segments, "
                    f"{len(sample_positions)} positions, and "
                    f"{len(sample_counts)} latent counts"
                )
            for segment_idx, ((start, end), count) in enumerate(
                zip(sample_positions, sample_counts)
            ):
                if count <= 0:
                    raise ValueError(
                        f"latent_counts[{batch_idx}][{segment_idx}] must be positive, got {count}"
                    )
                if start < 0 or end <= start:
                    raise ValueError(
                        f"memory_positions[{batch_idx}][{segment_idx}] is invalid: "
                        f"({start}, {end})"
                    )
                if end - start != count:
                    raise ValueError(
                        f"Batch {batch_idx}, segment {segment_idx}: position width "
                        f"{end - start} does not match latent count {count}"
                    )
                if sequence_length is not None and end > sequence_length:
                    raise ValueError(
                        f"memory_positions[{batch_idx}][{segment_idx}] ends at {end}, "
                        f"past sequence length {sequence_length}"
                    )
        return codes, positions, counts

    def _distributed_encoder_participation_required(self) -> bool:
        return (
            self.accelerator is not None
            and getattr(self.accelerator, "num_processes", 1) > 1
        )

    def _process_batched_latent_embeddings(
        self,
        memory_token_ids: List[List[List[int]]],
        *,
        ensure_participation: bool,
    ) -> Tuple[List[List[torch.Tensor]], Optional[torch.Tensor]]:
        """Encode all real segments in one synchronized encoder/adapter call.

        When a rank has no compressed segments during training, one synthetic
        segment is forwarded through both modules. Its scalar zero contribution
        is returned so the caller can connect that work to the decoder loss,
        giving encoder/adapter parameters zero (rather than missing) gradients.
        """
        regrouped: List[List[torch.Tensor]] = [
            [] for _ in range(len(memory_token_ids))
        ]
        flat_segments: List[List[int]] = []
        owners: List[Tuple[int, int]] = []
        for batch_idx, segments in enumerate(memory_token_ids):
            for segment_idx, segment in enumerate(segments):
                flat_segments.append(segment)
                owners.append((batch_idx, segment_idx))

        used_dummy = not flat_segments
        if used_dummy and not ensure_participation:
            return regrouped, None

        encoder_inputs = flat_segments
        if used_dummy:
            dummy_token_id = getattr(self.encoder, "pad_token_id", None)
            if dummy_token_id is None:
                tokenizer = getattr(self.encoder, "embed_tokenizer", None)
                dummy_token_id = getattr(tokenizer, "eos_token_id", None)
            if dummy_token_id is None:
                dummy_token_id = 0
            encoder_inputs = [[int(dummy_token_id)]]

        encoded = self.encoder(encoder_inputs)
        projected = self._project_latent_embeddings_synced(encoded)
        if len(projected) != len(encoder_inputs):
            raise RuntimeError(
                "Encoder/adapter output count mismatch: "
                f"expected {len(encoder_inputs)}, got {len(projected)}"
            )

        if used_dummy:
            zero_dependency = projected[0].sum() * 0.0
            return regrouped, zero_dependency

        for owner, embedding in zip(owners, projected):
            batch_idx, _ = owner
            regrouped[batch_idx].append(embedding)
        return regrouped, None

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
        batch_size, sequence_length = input_ids.shape[:2]
        codes_nested, pos_nested, counts_nested = (
            self._normalize_and_validate_memory_batch(
                memory_token_ids,
                memory_positions,
                latent_counts,
                batch_size=batch_size,
                sequence_length=sequence_length,
            )
        )
        self._last_local_has_memory = any(codes_nested)

        # Create new tensor and copy embeddings (avoids in-place modification issues with FSDP)
        # Cast to compute dtype to avoid LayerNorm dtype mismatch with FSDP mixed precision
        text_embeds = self.decoder.get_input_embeddings()(input_ids)
        compute_dtype = next(self.decoder.parameters()).dtype
        text_embeds = text_embeds.to(compute_dtype)
        inputs_embeds = text_embeds.new_zeros(text_embeds.shape)
        inputs_embeds.copy_(text_embeds)

        # Every training rank must enter encoder/adapter exactly once. This is
        # required by FSDP/DDP when a local packed batch happens to contain only
        # examples that intentionally bypass compression.
        ensure_participation = (
            self._distributed_encoder_participation_required()
            or (self.training and torch.is_grad_enabled())
        )
        latent_embeddings_nested, zero_dependency = (
            self._process_batched_latent_embeddings(
                codes_nested,
                ensure_participation=ensure_participation,
            )
        )

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

        if zero_dependency is not None:
            # Keep the dummy encoder/adapter graph in backward without changing
            # any decoder embedding or loss value.
            combined_embeds = combined_embeds + zero_dependency.to(
                device=combined_embeds.device,
                dtype=combined_embeds.dtype,
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
        batch_size, sequence_length = input_ids.shape[:2]
        codes_nested, pos_nested, counts_nested = (
            self._normalize_and_validate_memory_batch(
                memory_token_ids,
                memory_positions,
                latent_counts,
                batch_size=batch_size,
                sequence_length=sequence_length,
            )
        )

        # Create new tensor and copy embeddings (avoids in-place modification issues)
        # Cast to compute dtype to avoid LayerNorm dtype mismatch
        compute_dtype = next(self.decoder.parameters()).dtype
        text_embeds = self.decoder.get_input_embeddings()(input_ids).to(compute_dtype)
        inputs_embeds = text_embeds.new_zeros(text_embeds.shape)
        inputs_embeds.copy_(text_embeds)
        latent_embeddings_nested, zero_dependency = (
            self._process_batched_latent_embeddings(
                codes_nested,
                ensure_participation=self._distributed_encoder_participation_required(),
            )
        )

        combined_embeds = self.processor.replace_memory_tokens_with_embeddings(
            inputs_embeds=inputs_embeds,
            latent_embeddings=latent_embeddings_nested,  # type: ignore[arg-type]
            memory_positions=pos_nested,
            latent_counts=counts_nested,
            input_ids=input_ids,
        )
        if zero_dependency is not None:
            combined_embeds = combined_embeds + zero_dependency.to(
                device=combined_embeds.device,
                dtype=combined_embeds.dtype,
            )
        
        # Generate using the LLM
        return self.decoder.generate(
            inputs_embeds=combined_embeds,
            attention_mask=attention_mask,
            **generation_kwargs
        )
