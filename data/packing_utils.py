"""
Complete packing implementation for Code-LLaVA with flex attention.
Clean, tested, production-ready code.
"""

import pickle
from typing import List, Dict, Any, Tuple, Optional
import torch.nn.functional as F
from tqdm import tqdm
import torch

from data.chat_utils import build_prompt_and_target_text


def compute_pre_compression_stats(
    example: Dict[str, Any],
    processor,
    prompt_column: str = "compression_prompt",
) -> Dict[str, Any]:
    """
    Compute token counts BEFORE compression (original text without placeholder replacement).

    Returns dict with:
    - trainable_tokens: tokens we compute loss on (target/assistant response)
    - non_trainable_tokens: tokens masked from loss (prompt/user message)
    - total_tokens: total sequence length
    """
    import re

    prompt_data = example[prompt_column]
    target_data = example.get("target", "")

    # Detect format: "NA_string_only" target = continual pretraining
    is_continual_pretraining = (target_data == "NA_string_only")

    if is_continual_pretraining:
        # Continual pretraining: extract original text from wrapped chat format
        if isinstance(prompt_data, list) and len(prompt_data) > 0:
            original_text = prompt_data[0].get("content", "")
        else:
            original_text = str(prompt_data)

        # Count trainable vs non-trainable
        # Non-trainable = text inside <|memory_start|>...<|memory_end|> tags
        # Trainable = text outside tags
        memory_pattern = re.compile(
            r"<\|memory_start\|>(.*?)<\|memory_end\|>",
            re.DOTALL
        )

        # Find all wrapped regions
        non_trainable_text = ""
        for match in memory_pattern.finditer(original_text):
            non_trainable_text += match.group(1)  # Content inside tags

        # Remove wrapped regions to get trainable text
        trainable_text = memory_pattern.sub("", original_text)

        non_trainable_tokens = len(processor.decoder_tokenizer.encode(non_trainable_text, add_special_tokens=False)) if non_trainable_text else 0
        trainable_tokens = len(processor.decoder_tokenizer.encode(trainable_text, add_special_tokens=False)) if trainable_text else 0

        return {
            "format": "continual_pretraining",
            "trainable_tokens": trainable_tokens,
            "non_trainable_tokens": non_trainable_tokens,
            "total_tokens": trainable_tokens + non_trainable_tokens,
        }
    else:
        # Chat format: prompt = non-trainable, target = trainable
        prompt_text, target_text, _, _ = build_prompt_and_target_text(
            prompt_data,
            target_data,
            tokenizer=processor.decoder_tokenizer,
        )
        non_trainable_tokens = len(processor.decoder_tokenizer.encode(prompt_text, add_special_tokens=False))
        trainable_tokens = len(processor.decoder_tokenizer.encode(target_text, add_special_tokens=False))

        return {
            "format": "chat",
            "trainable_tokens": trainable_tokens,
            "non_trainable_tokens": non_trainable_tokens,
            "total_tokens": trainable_tokens + non_trainable_tokens,
        }


def compute_post_compression_stats(
    processed_result: Dict[str, Any],
    is_chat_format: bool,
) -> Dict[str, Any]:
    """
    Compute token counts AFTER compression (with placeholders).

    Returns dict with:
    - trainable_tokens: labels != -100 (compute loss)
    - non_trainable_tokens: labels == -100 (masked, no loss)
    - total_tokens: total sequence length
    """
    labels = processed_result['processed']['labels']

    # Handle both tensor and list formats
    if hasattr(labels, 'tolist'):
        labels_list = labels.squeeze().tolist()
    elif isinstance(labels[0], list):
        labels_list = labels[0]
    else:
        labels_list = labels

    total_tokens = len(labels_list)

    # Count trainable tokens (labels != -100)
    trainable_tokens = sum(1 for l in labels_list if l != -100)
    non_trainable_tokens = total_tokens - trainable_tokens

    return {
        "format": "chat" if is_chat_format else "continual_pretraining",
        "trainable_tokens": trainable_tokens,
        "non_trainable_tokens": non_trainable_tokens,
        "total_tokens": total_tokens,
    }


def preprocess_single_example(
    example: Dict[str, Any],
    processor,
    prompt_column: str = "compression_prompt",
) -> Dict[str, Any]:
    """
    Process a single example through the processor to get actual sequence length.

    Handles two formats:
    1. Chat format (SFT): compression_prompt is a list of message dicts, target is response
    2. Continual pretraining: target is "NA_string_only", extract user content for processing

    Args:
        example: Dict with 'compression_prompt' and 'target' keys
        processor: LCLMProcessor instance
        prompt_column: Which column to use for prompt (default: "compression_prompt")

    Returns:
        Dict with 'processed' (processor output) and 'seq_len' (actual length)
    """
    prompt_data = example[prompt_column]
    target_data = example.get("target", "")

    # Detect format: "NA_string_only" target = continual pretraining (wrapped in chat format)
    is_continual_pretraining = (target_data == "NA_string_only")

    if is_continual_pretraining:
        # Continual pretraining format: extract user content from wrapped chat format
        # prompt_data is [{"role": "user", "content": original_string}]
        if isinstance(prompt_data, list) and len(prompt_data) > 0:
            # Extract the original string from user message
            original_text = prompt_data[0].get("content", "")
        else:
            original_text = str(prompt_data)

        # Process without chat template, train on non-wrapped portions
        result = processor.process_continual_pretraining_batch(
            texts=[original_text],
            padding=False,
            truncation=False,
            return_tensors="pt",
        )
    else:
        # Chat format (SFT): use chat template
        prompt_text, target_text, _, _ = build_prompt_and_target_text(
            prompt_data,
            target_data,
            tokenizer=processor.decoder_tokenizer,
        )

        # Process through processor (this expands code placeholders)
        result = processor.process_wrapped_batch(
            prompts=[prompt_text],
            targets=[target_text],
            padding=False,
            truncation=False,
            return_tensors="pt",
        )

    # Get actual sequence length
    seq_len = result['input_ids'].shape[-1]

    return {
        'processed': result,
        'seq_len': seq_len,
    }


def pack_sequences(
    processed_examples: List[Dict],
    max_packed_length: int = 4096,
    shuffle: bool = True,
    random_seed: int = 42,
) -> List[List[Dict]]:
    """
    Pack sequences using simple greedy first-fit algorithm.
    
    Args:
        processed_examples: List of dicts with 'processed' and 'seq_len'
        max_packed_length: Maximum tokens in a packed sequence
        shuffle: If True, shuffle examples before packing (recommended for data mixture)
        random_seed: Random seed for shuffling
    
    Returns:
        List of packed batches (each batch is a list of examples)
    """
    import random
    
    # Create a pool of examples to pack
    pool = list(processed_examples)
    
    # Shuffle pool for random selection (ensures good data mixture)
    if shuffle:
        random.seed(random_seed)
        random.shuffle(pool)
    
    packed_batches = []
    current_batch = []
    current_length = 0
    skipped_count = 0
    
    # Track statistics for each packed batch
    batch_stats = []  # List of (seq_len, num_examples) tuples
    
    # Simple greedy packing
    pbar = tqdm(pool, desc="Packing sequences")
    for ex in pbar:
        seq_len = ex['seq_len']
        
        # Skip sequences that are too long for any batch
        if seq_len > max_packed_length:
            skipped_count += 1
            continue
        
        # Try to add to current batch
        if current_length + seq_len <= max_packed_length:
            # Fits in current batch
            current_batch.append(ex)
            current_length += seq_len
        else:
            # Doesn't fit - close current batch and start new one
            if current_batch:
                batch_stats.append((current_length, len(current_batch)))
                packed_batches.append(current_batch)
            current_batch = [ex]
            current_length = seq_len
    
    # Don't forget the last batch
    if current_batch:
        batch_stats.append((current_length, len(current_batch)))
        packed_batches.append(current_batch)
    
    # Print summary statistics
    if packed_batches and batch_stats:
        total_examples_packed = len([ex for batch in packed_batches for ex in batch])
        total_tokens = sum(ex['seq_len'] for batch in packed_batches for ex in batch)
        
        # Extract stats
        seq_lengths = [stats[0] for stats in batch_stats]
        num_examples_list = [stats[1] for stats in batch_stats]
        
        num_sequences = len(packed_batches)
        avg_tokens_per_seq = sum(seq_lengths) / len(seq_lengths) if seq_lengths else 0
        avg_data_per_seq = sum(num_examples_list) / len(num_examples_list) if num_examples_list else 0
        
        print(f"\n{'='*70}")
        print(f"PACKING COMPLETE")
        print(f"{'='*70}")
        print(f"Total examples processed:          {len(processed_examples):,}")
        print(f"Examples packed:                   {total_examples_packed:,}")
        print(f"Examples skipped (too long):       {skipped_count:,}")
        print(f"\n{'='*70}")
        print(f"Number of packed sequences:        {num_sequences:,}")
        print(f"Average tokens per sequence:       {avg_tokens_per_seq:,.0f} / {max_packed_length:,} ({avg_tokens_per_seq/max_packed_length*100:.1f}%)")
        print(f"Average examples per sequence:     {avg_data_per_seq:.1f}")
        print(f"\nMin tokens per sequence:           {min(seq_lengths):,}")
        print(f"Max tokens per sequence:           {max(seq_lengths):,}")
        print(f"Min examples per sequence:         {min(num_examples_list)}")
        print(f"Max examples per sequence:         {max(num_examples_list)}")
        print(f"{'='*70}\n")
    
    return packed_batches


def _to_1d_long(x) -> Optional[torch.Tensor]:
    """
    Normalize input_ids/labels to a 1D long tensor.
    Handles tensor, list[int], or list[list[int]] formats.
    """
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return x.reshape(-1).to(dtype=torch.long)
    # x is a Python list
    if len(x) > 0 and isinstance(x[0], list):
        x = x[0]
    return torch.as_tensor(x, dtype=torch.long)


def collate_packed_batch(
    packed_batch: List[Dict],
    target_length: Optional[int] = None,
    pad_token_id: int = 0,
) -> Dict[str, Any]:
    """
    Collate a single pre-packed batch into tensors for training.

    Optimized torch-first implementation:
    - Uses torch.cat() instead of list.extend() for token arrays
    - Uses F.pad() instead of Python list padding
    - Avoids .tolist() conversions (tensor→Python is expensive)
    - Keeps only ragged metadata (positions, embed_ids, counts) as Python lists

    Args:
        packed_batch: List of processed examples to pack together
        target_length: If provided, pad to this length for consistent tensor shapes
                      (avoids torch.compile recompilation)
        pad_token_id: Token ID to use for padding

    Returns:
        Dict with packed tensors ready for model (batch_size=1)
    """
    input_tensors: List[torch.Tensor] = []
    label_tensors: List[Optional[torch.Tensor]] = []
    have_any_labels = False

    packed_memory_token_ids: List[int] = []
    packed_memory_positions: List[Tuple[int, int]] = []
    packed_latent_counts: List[int] = []
    sample_lens: List[int] = []

    current_offset = 0

    for ex_dict in packed_batch:
        ex = ex_dict['processed']

        # Normalize to 1D tensor (no .tolist() - stays in torch)
        inp = _to_1d_long(ex['input_ids'])
        input_tensors.append(inp)

        seq_len = inp.numel()
        sample_lens.append(seq_len)

        # Handle labels
        lab = _to_1d_long(ex.get('labels'))
        if lab is not None:
            have_any_labels = True
            label_tensors.append(lab)
        else:
            label_tensors.append(None)

        # Handle code positions (adjust by offset) - small, Python list is fine
        positions = ex.get('memory_positions')
        if positions and positions[0]:
            packed_memory_positions.extend(
                (s + current_offset, e + current_offset) for s, e in positions[0]
            )

        # Handle memory_token_ids (flatten) - small, Python list is fine
        memory_token_ids = ex.get('memory_token_ids') or ex.get('codes')
        if memory_token_ids and memory_token_ids[0]:
            packed_memory_token_ids.extend(memory_token_ids[0])

        # Handle latent_counts (flatten) - small, Python list is fine
        counts = ex.get('latent_counts')
        if counts and counts[0]:
            packed_latent_counts.extend(counts[0])

        current_offset += seq_len

    # Concatenate in torch (C++ fast path) instead of Python list.extend()
    input_ids = torch.cat(input_tensors, dim=0) if len(input_tensors) > 1 else input_tensors[0]

    labels = None
    if have_any_labels:
        # Fill missing labels with -100 tensors so we can cat once
        fixed_label_tensors = []
        for inp, lab in zip(input_tensors, label_tensors):
            if lab is None:
                fixed_label_tensors.append(torch.full((inp.numel(),), -100, dtype=torch.long))
            else:
                fixed_label_tensors.append(lab)
        labels = torch.cat(fixed_label_tensors, dim=0) if len(fixed_label_tensors) > 1 else fixed_label_tensors[0]

    # Pad using F.pad (C++ fast path) instead of Python list extension
    if target_length is not None:
        cur_len = input_ids.numel()
        if cur_len < target_length:
            pad_len = target_length - cur_len
            input_ids = F.pad(input_ids, (0, pad_len), value=pad_token_id)
            if labels is not None:
                labels = F.pad(labels, (0, pad_len), value=-100)
        elif cur_len > target_length:
            # Sequence exceeds target_length - this is fine, collate_multiple_packed_batches
            # will handle padding to the longest sequence in the batch
            pass

    # Add batch dimension (batch_size=1)
    input_ids = input_ids.unsqueeze(0)
    if labels is not None:
        labels = labels.unsqueeze(0)

    # Note: attention_mask is NOT included - replaced by BlockMask in forward() when sample_lens is provided
    return {
        'input_ids': input_ids,                          # [1, seq_len]
        'labels': labels,                                # [1, seq_len] or None
        'memory_token_ids': [packed_memory_token_ids],       # List of batch_size=1, each element is List[int]
        'memory_positions': [packed_memory_positions],       # List of batch_size=1
        'latent_counts': [packed_latent_counts],           # List of batch_size=1
        'sample_lens': [sample_lens],                    # List[List[int]] for batched flex attention
    }


def collate_multiple_packed_batches(
    batch_of_packed: List[Dict[str, Any]],
    target_length: int = None,
    pad_token_id: int = 0,
) -> Dict[str, Any]:
    """
    Collate multiple pre-packed sequences into a batched tensor.

    Each item in batch_of_packed is already a collated packed sequence (from collate_packed_batch).
    This function stacks them into batch dimension.

    Args:
        batch_of_packed: List of collated packed batches (each has batch_size=1)
        target_length: If provided, pad all sequences to this length
        pad_token_id: Token ID to use for padding
        

    Returns:
        Dict with batched tensors [batch_size, seq_len]
    """
    batch_size = len(batch_of_packed)

    if batch_size == 0:
        raise ValueError("Empty batch")

    # Find max sequence length in this batch
    # Always compute actual max to handle sequences exceeding target_length
    actual_max_len = max(b['input_ids'].shape[-1] for b in batch_of_packed)

    if target_length is not None:
        # Use the larger of target_length and actual max (handles overflow gracefully)
        max_len = max(target_length, actual_max_len)
    else:
        max_len = actual_max_len

    # If batch_size == 1 and already at max_len, just return as-is
    if batch_size == 1 and batch_of_packed[0]['input_ids'].shape[-1] == max_len:
        return batch_of_packed[0]

    # Prepare batched tensors
    batched_input_ids = []
    batched_labels = []
    batched_memory_token_ids = []
    batched_memory_positions = []
    batched_latent_counts = []
    batched_sample_lens = []

    for b in batch_of_packed:
        seq_len = b['input_ids'].shape[-1]
        pad_len = max_len - seq_len

        # Pad input_ids
        input_ids = b['input_ids'].squeeze(0)  # [seq_len]
        if pad_len > 0:
            input_ids = torch.cat([
                input_ids,
                torch.full((pad_len,), pad_token_id, dtype=torch.long)
            ])
        batched_input_ids.append(input_ids)

        # Pad labels
        if b['labels'] is not None:
            labels = b['labels'].squeeze(0)  # [seq_len]
            if pad_len > 0:
                labels = torch.cat([
                    labels,
                    torch.full((pad_len,), -100, dtype=torch.long)
                ])
            batched_labels.append(labels)

        # Keep per-batch metadata as lists
        batched_memory_token_ids.append(b['memory_token_ids'][0])  # List[List[int]]
        batched_memory_positions.append(b['memory_positions'][0])  # List[Tuple]
        batched_latent_counts.append(b['latent_counts'][0])  # List[int]
        batched_sample_lens.append(b['sample_lens'][0])  # List[int] - document lengths

    # Stack into batch tensors
    return {
        'input_ids': torch.stack(batched_input_ids, dim=0),  # [batch_size, seq_len]
        'labels': torch.stack(batched_labels, dim=0) if batched_labels else None,  # [batch_size, seq_len]
        'memory_token_ids': batched_memory_token_ids,  # List[List[List[int]]] - per batch item
        'memory_positions': batched_memory_positions,  # List[List[Tuple]] - per batch item
        'latent_counts': batched_latent_counts,  # List[List[int]] - per batch item
        'sample_lens': batched_sample_lens,  # List[List[int]] - per batch item, for block mask
    }


def create_document_mask(sample_lens: List[int], total_len: int, device: torch.device):
    """
    Create document boundary mask for flex attention (single batch item).

    Args:
        sample_lens: List of sequence lengths (actual data, not including padding)
        total_len: Total sequence length including padding
        device: Device to create mask on

    Returns:
        Mask function for use with create_block_mask
    """
    data_len = sum(sample_lens)

    # Validate: data should not exceed total_len
    if data_len > total_len:
        raise ValueError(
            f"sum(sample_lens)={data_len} exceeds total_len={total_len}. "
            f"This means the packed data is longer than max_packed_length."
        )

    # Create document IDs - each sample gets a unique ID
    # Padding tokens (if any) get ID = -1 and will be masked out
    doc_ids = []
    for i, length in enumerate(sample_lens):
        doc_ids.append(torch.full((length,), i, dtype=torch.long, device=device))

    if data_len < total_len:
        # Padding gets special ID that won't match anything
        pad_len = total_len - data_len
        doc_ids.append(torch.full((pad_len,), -1, dtype=torch.long, device=device))

    document_id = torch.cat(doc_ids)

    # Verify document_id length matches total_len
    assert document_id.shape[0] == total_len, f"document_id length {document_id.shape[0]} != total_len {total_len}"

    def mask_fn(b, h, q_idx, kv_idx):
        """Causal mask within same document. Padding tokens (-1) never match."""
        return (document_id[q_idx] == document_id[kv_idx]) & (document_id[q_idx] >= 0) & (q_idx >= kv_idx)

    return mask_fn


def create_batched_document_mask(
    batch_sample_lens: List[List[int]],
    total_len: int,
    device: torch.device,
):
    """
    Create document boundary mask for flex attention with batched input.

    Args:
        batch_sample_lens: List of sample_lens per batch item. Each is a list of document lengths.
        total_len: Total sequence length including padding (same for all batch items)
        device: Device to create mask on

    Returns:
        Mask function for use with create_block_mask
    """
    batch_size = len(batch_sample_lens)

    # Create document IDs for each batch item: [batch_size, total_len]
    all_doc_ids = []

    for sample_lens in batch_sample_lens:
        data_len = sum(sample_lens)

        if data_len > total_len:
            raise ValueError(
                f"sum(sample_lens)={data_len} exceeds total_len={total_len}. "
                f"This means the packed data is longer than max_packed_length."
            )

        # Create document IDs for this batch item
        doc_ids = []
        for i, length in enumerate(sample_lens):
            doc_ids.append(torch.full((length,), i, dtype=torch.long, device=device))

        if data_len < total_len:
            pad_len = total_len - data_len
            doc_ids.append(torch.full((pad_len,), -1, dtype=torch.long, device=device))

        all_doc_ids.append(torch.cat(doc_ids))

    # Stack into [batch_size, total_len]
    document_ids = torch.stack(all_doc_ids, dim=0)

    def mask_fn(b, h, q_idx, kv_idx):
        """Causal mask within same document per batch item. Padding tokens (-1) never match."""
        return (document_ids[b, q_idx] == document_ids[b, kv_idx]) & (document_ids[b, q_idx] >= 0) & (q_idx >= kv_idx)

    return mask_fn


def create_block_mask_for_packed(
    sample_lens: List[int],
    num_heads: int,
    device: torch.device,
    block_size: int = 128,
    total_len: int = None,
):
    """
    Create block mask for packed sequence (single batch item, B=1).

    Args:
        sample_lens: List of sequence lengths (actual data)
        num_heads: Number of attention heads
        device: Device
        block_size: Block size for block-sparse attention
        total_len: Total sequence length including padding. If None, uses sum(sample_lens).

    Returns:
        BlockMask for flex_attention
    """
    from torch.nn.attention.flex_attention import create_block_mask

    if total_len is None:
        total_len = sum(sample_lens)
    mask_fn = create_document_mask(sample_lens, total_len, device)

    block_mask = create_block_mask(
        mask_fn,
        B=1,
        H=num_heads,
        Q_LEN=total_len,
        KV_LEN=total_len,
        device=device,
        BLOCK_SIZE=block_size,
        _compile=True,
    )

    return block_mask


def create_block_mask_for_packed_batch(
    batch_sample_lens: List[List[int]],
    num_heads: int,
    device: torch.device,
    block_size: int = 128,
    total_len: int = None,
):
    """
    Create block mask for batched packed sequences (B > 1).

    Args:
        batch_sample_lens: List of sample_lens per batch item
        num_heads: Number of attention heads
        device: Device
        block_size: Block size for block-sparse attention
        total_len: Total sequence length including padding. If None, uses max of sums.

    Returns:
        BlockMask for flex_attention
    """
    from torch.nn.attention.flex_attention import create_block_mask

    batch_size = len(batch_sample_lens)

    if total_len is None:
        total_len = max(sum(lens) for lens in batch_sample_lens)

    mask_fn = create_batched_document_mask(batch_sample_lens, total_len, device)

    block_mask = create_block_mask(
        mask_fn,
        B=batch_size,
        H=num_heads,
        Q_LEN=total_len,
        KV_LEN=total_len,
        device=device,
        BLOCK_SIZE=block_size,
        _compile=True,
    )

    return block_mask
