import os
from typing import List, Dict

import torch
from data.chat_utils import build_prompt_and_target_text
from data.packing_utils import collate_multiple_packed_batches

# StatefulDataLoader for dynamic packing with auto-resume
try:
    from torchdata.stateful_dataloader import StatefulDataLoader
    HAS_STATEFUL_DATALOADER = True
except ImportError as e:
    HAS_STATEFUL_DATALOADER = False
    print(f"[WARNING] StatefulDataLoader not available: {e}")


def collate_batch(
    examples: List[Dict[str, str]],
    processor,
    max_memory_length: int,
):
    """Collator for prompt/target datasets.

    Handles two formats per example:
    1. Chat format (SFT): target is assistant response, uses chat template
    2. Continual pretraining: target == "NA_string_only", uses raw text without chat template

    Mixed batches are supported - each example is processed according to its format,
    then results are merged.
    """
    tokenizer = processor.decoder_tokenizer

    # Separate examples by format
    chat_examples = []
    chat_indices = []
    continual_examples = []
    continual_indices = []

    for i, ex in enumerate(examples):
        target = ex.get("target", "")
        if target == "NA_string_only":
            continual_indices.append(i)
            continual_examples.append(ex)
        else:
            chat_indices.append(i)
            chat_examples.append(ex)

    results = [None] * len(examples)

    # Process chat format examples
    if chat_examples:
        prompt_texts = []
        target_texts = []

        for ex in chat_examples:
            prompt_text, target_text, _, _ = build_prompt_and_target_text(
                ex.get("prompt"),
                ex.get("target", ""),
                tokenizer=tokenizer,
            )
            prompt_texts.append(prompt_text)
            target_texts.append(target_text)

        chat_result = processor.process_wrapped_batch(
            prompts=prompt_texts,
            targets=target_texts,
            max_length=None,
            padding=True,  # Pad to longest in this batch
            truncation=False,
            return_tensors="pt",
        )

        for i, idx in enumerate(chat_indices):
            results[idx] = {
                'input_ids': chat_result['input_ids'][i],
                'attention_mask': chat_result['attention_mask'][i],
                'labels': chat_result['labels'][i],
                'memory_positions': chat_result['memory_positions'][i],
                'latent_counts': chat_result['latent_counts'][i],
                'memory_token_ids': chat_result['memory_token_ids'][i],
            }

    # Process continual pretraining examples
    if continual_examples:
        texts = []
        for ex in continual_examples:
            prompt_data = ex.get("prompt")
            # prompt_data is [{"role": "user", "content": original_string}]
            if isinstance(prompt_data, list) and len(prompt_data) > 0:
                original_text = prompt_data[0].get("content", "")
            else:
                original_text = str(prompt_data)
            texts.append(original_text)

        continual_result = processor.process_continual_pretraining_batch(
            texts=texts,
            padding=True,  # Pad to longest in this batch
            truncation=False,
            return_tensors="pt",
        )

        for i, idx in enumerate(continual_indices):
            results[idx] = {
                'input_ids': continual_result['input_ids'][i],
                'attention_mask': continual_result['attention_mask'][i],
                'labels': continual_result['labels'][i],
                'memory_positions': continual_result['memory_positions'][i],
                'latent_counts': continual_result['latent_counts'][i],
                'memory_token_ids': continual_result['memory_token_ids'][i],
            }

    # Merge and pad to longest
    max_len = max(r['input_ids'].shape[0] for r in results)
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    batch_input_ids = []
    batch_attention_mask = []
    batch_labels = []
    batch_memory_positions = []
    batch_latent_counts = []
    batch_memory_token_ids = []

    for r in results:
        seq_len = r['input_ids'].shape[0]
        pad_len = max_len - seq_len

        if pad_len > 0:
            # Pad on the right
            batch_input_ids.append(torch.cat([r['input_ids'], torch.full((pad_len,), pad_token_id, dtype=r['input_ids'].dtype)]))
            batch_attention_mask.append(torch.cat([r['attention_mask'], torch.zeros(pad_len, dtype=r['attention_mask'].dtype)]))
            batch_labels.append(torch.cat([r['labels'], torch.full((pad_len,), -100, dtype=r['labels'].dtype)]))
        else:
            batch_input_ids.append(r['input_ids'])
            batch_attention_mask.append(r['attention_mask'])
            batch_labels.append(r['labels'])

        batch_memory_positions.append(r['memory_positions'])
        batch_latent_counts.append(r['latent_counts'])
        batch_memory_token_ids.append(r['memory_token_ids'])

    return {
        'input_ids': torch.stack(batch_input_ids),
        'attention_mask': torch.stack(batch_attention_mask),
        'labels': torch.stack(batch_labels),
        'memory_positions': batch_memory_positions,
        'latent_counts': batch_latent_counts,
        'memory_token_ids': batch_memory_token_ids,
    }


def prepare_datasets(data_args, processor, training_args, accelerator=None, pooling: str = "mean"):
    """Dataset loader and collator setup.

    Args:
        data_args: Data arguments (use_packing, dataset_name, etc.)
        processor: LCLMProcessor
        training_args: Training arguments
        accelerator: Optional accelerator for distributed training
        pooling: Pooling strategy ('mean', 'eos', 'concat')

    Returns:
        train_dl: Training dataloader
        eval_dls: Dict of evaluation dataloaders
    """
    # Check if using packing (dynamic packing with StatefulDataLoader)
    use_packing = data_args.use_packing

    if use_packing:
        # Dynamic packing: uses pre-packed parquet files with StatefulDataLoader
        from data.dynamic_packing_dataset import DynamicPackedDataset

        packed_path = data_args.dataset_name
        if not os.path.exists(packed_path):
            raise FileNotFoundError(
                f"Packed dataset not found at: {packed_path}\n"
                f"Please preprocess first with:\n"
                f"  python data/preprocess_for_dynamic_packing.py \\\n"
                f"    --input_path <your_data.parquet> \\\n"
                f"    --output_dir {packed_path}"
            )

        # Get compression_ratio from training args or processor
        compression_ratio = training_args.compression_ratio or processor.compression_ratio

        # Get seed from training args
        seed = training_args.seed

        if accelerator:
            accelerator.print(f"Loading packed dataset from: {packed_path}")
            accelerator.print(f"  Runtime compression_ratio: {compression_ratio}")
            accelerator.print(f"  Seed: {seed}")

        # Get target_length for consistent tensor shapes (avoids torch.compile recompilation)
        # This should match max_packed_length used during preprocessing
        target_length = data_args.max_packed_length

        # Create dynamic dataset with shuffling
        train_dataset = DynamicPackedDataset(
            parquet_path=packed_path,
            decoder_tokenizer=processor.decoder_tokenizer,
            embed_tokenizer=processor.embed_tokenizer,
            compression_ratio=compression_ratio,
            num_processes=accelerator.num_processes if accelerator else 1,
            process_rank=accelerator.process_index if accelerator else 0,
            seed=seed,
            shuffle=True,  # Shuffle rows within each file
            shuffle_files=True,  # Shuffle file order
            drop_last_files=True,  # Drop extra files for even distribution (prevents hangs)
            pooling=pooling,
            target_length=target_length,  # Pad to consistent length for flex_attention
        )

        # Create StatefulDataLoader for auto-resume support
        if not HAS_STATEFUL_DATALOADER:
            raise ImportError(
                "torchdata is required for StatefulDataLoader. "
                "Install with: pip install torchdata"
            )

        # Get batch size from data_args
        train_batch_size = data_args.train_batch_size

        # Create collate function that handles batching multiple packed sequences
        def packed_collate_fn(batch_of_packed):
            """Collate multiple packed sequences into a batch."""
            return collate_multiple_packed_batches(
                batch_of_packed,
                target_length=target_length,
                pad_token_id=processor.decoder_tokenizer.pad_token_id or 0,
            )

        train_dl = StatefulDataLoader(
            train_dataset,
            batch_size=train_batch_size,  # Batch multiple packed sequences together
            shuffle=False,  # Dataset handles ordering
            num_workers=0,  # Required for stateful IterableDataset
            pin_memory=True,
            collate_fn=packed_collate_fn,
        )
        if accelerator:
            accelerator.print(f"  Using StatefulDataLoader (batch_size={train_batch_size}) for auto-resume")

    else:
        # Unpacked training: stateful dataset with collation
        from data.stateful_dataset import create_stateful_dataloader

        # Get seed from training args
        seed = training_args.seed

        train_collate_fn = lambda examples: collate_batch(
            examples,
            processor,
            data_args.max_memory_length,
        )

        train_dl = create_stateful_dataloader(
            parquet_path=data_args.dataset_name,
            collate_fn=train_collate_fn,
            batch_size=data_args.train_batch_size,
            accelerator=accelerator,
            seed=seed,
            shuffle=True,
            shuffle_files=True,
            drop_last_files=True,
            prompt_column="compression_prompt",
        )

        if accelerator:
            accelerator.print("  Using StatefulDataLoader for auto-resume")

    if accelerator:
        accelerator.print("Created dataloaders:")
        if use_packing:
            accelerator.print(f"  Train: Packed (streaming with shuffle)")
        else:
            accelerator.print(f"  Train: {len(train_dl)} batches")

    return train_dl
