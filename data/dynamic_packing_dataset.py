"""
Dynamic packing dataset with on-the-fly expansion.

Loads pre-packed batches (with single <|memory|> placeholder per region)
and expands them at runtime based on compression_ratio.

Works with StatefulDataLoader for auto-resume checkpointing.
Supports file-level and row-level shuffling with deterministic resume.
"""

import os
import glob
import math
import pickle
import random
from typing import Dict, List, Any, Optional, Tuple

import pyarrow.parquet as pq
import torch
from torch.utils.data import IterableDataset

from data.packing_utils import collate_packed_batch


class DynamicPackedDataset(IterableDataset):
    """
    Stateful dataset that loads pre-packed batches and expands on-the-fly.

    Features:
    - File-level shuffling (deterministic with seed)
    - Row-level shuffling within each file (with RNG state for resume)
    - StatefulDataLoader compatible for auto-resume checkpointing
    - Supports different pooling strategies (mean, eos, concat)

    With StatefulDataLoader:
    - state_dict() is called automatically when checkpointing
    - load_state_dict() is called automatically when resuming
    - No manual skipping needed

    Usage:
        from torchdata.stateful_dataloader import StatefulDataLoader

        dataset = DynamicPackedDataset(...)
        dataloader = StatefulDataLoader(dataset, batch_size=1, num_workers=0)

        # Accelerate handles save/load automatically
        dataloader = accelerator.prepare(dataloader)
    """

    def __init__(
        self,
        parquet_path: str,
        decoder_tokenizer,
        embed_tokenizer,
        compression_ratio: int,
        num_processes: int = 1,
        process_rank: int = 0,
        seed: int = 42,
        shuffle: bool = True,
        shuffle_files: bool = True,
        drop_last_files: bool = True,
        pooling: str = "mean",
        target_length: int = None,
    ):
        """
        Args:
            parquet_path: Path to directory containing parquet files
            decoder_tokenizer: LLM tokenizer (for special token IDs)
            embed_tokenizer: Embed tokenizer (for tokenizing memory_strings)
            compression_ratio: Number of embed tokens per chunk (runtime parameter!)
            num_processes: Total number of distributed processes
            process_rank: This process's rank
            seed: Random seed for reproducibility
            shuffle: Whether to shuffle rows within each file
            shuffle_files: Whether to shuffle file order
            drop_last_files: Drop extra files so each rank gets equal number (prevents hangs)
            pooling: Pooling strategy ('mean', 'eos', 'concat')
                All modes produce the same number of embeddings: ceil(num_tokens / compression_ratio)
            target_length: If provided, pad all batches to this length for consistent tensor shapes
                          (avoids torch.compile recompilation with flex_attention)
        """
        self.parquet_path = parquet_path
        self.decoder_tokenizer = decoder_tokenizer
        self.embed_tokenizer = embed_tokenizer
        self.compression_ratio = compression_ratio
        self.num_processes = num_processes
        self.process_rank = process_rank
        self.seed = seed
        self.shuffle = shuffle
        self.shuffle_files = shuffle_files
        self.drop_last_files = drop_last_files
        self.pooling = pooling

        # Target length for padding (ensures consistent tensor shapes for torch.compile)
        # If None, no padding is applied
        self.target_length = target_length

        # Get all parquet files
        all_files = sorted(glob.glob(os.path.join(parquet_path, "*.parquet")))
        if not all_files:
            raise ValueError(f"No parquet files found in {parquet_path}")

        # Shuffle files deterministically with seed (before dropping/sharding)
        if shuffle_files:
            random.Random(seed).shuffle(all_files)

        # Drop extra files to ensure even distribution across ranks
        # This prevents hangs when ranks finish at different times
        if drop_last_files and len(all_files) % num_processes != 0:
            original_count = len(all_files)
            # Keep only files that divide evenly
            files_to_keep = (len(all_files) // num_processes) * num_processes
            all_files = all_files[:files_to_keep]
            print(
                f"[Rank {process_rank}] Dropped {original_count - files_to_keep} files "
                f"for even distribution: {original_count} → {len(all_files)}"
            )

        # Shard across processes
        self.parquet_files = self._shard_files(all_files, num_processes, process_rank)

        # Log file assignment
        print(
            f"[Rank {process_rank}/{num_processes}] DynamicPackedDataset: "
            f"{len(self.parquet_files)}/{len(all_files)} parquet files"
        )
        if self.parquet_files:
            print(f"  Files: {self.parquet_files[0]} ... {self.parquet_files[-1]}")
        print(f"  Shuffle files: {shuffle_files}, Shuffle rows: {shuffle}, Drop last: {drop_last_files}")

        # Get special token IDs
        self.memory_start_id = decoder_tokenizer.convert_tokens_to_ids('<|memory_start|>')
        self.memory_end_id = decoder_tokenizer.convert_tokens_to_ids('<|memory_end|>')
        self.memory_id = decoder_tokenizer.convert_tokens_to_ids('<|memory|>')

        # Validate special tokens exist
        if self.memory_id is None or self.memory_start_id is None or self.memory_end_id is None:
            raise ValueError(
                "LLM tokenizer missing special tokens. "
                f"memory_start={self.memory_start_id}, memory={self.memory_id}, memory_end={self.memory_end_id}"
            )

        # Initialize RNG for row shuffling (per-rank seed for different shuffles)
        self._rng = random.Random(seed + process_rank)

        # State for checkpointing
        self._state = {
            'file_idx': 0,
            'row_idx': 0,  # Index into the shuffled row order
            'epoch': 0,
            'rng_state': self._rng.getstate(),  # Save RNG state for reproducible resume
        }

        # Current file's shuffled indices (populated when file is loaded)
        self._current_row_indices: List[int] = []

        # Count total rows (packed batches) for __len__
        self._total_rows = 0
        for pq_file in self.parquet_files:
            pq_meta = pq.read_metadata(pq_file)
            self._total_rows += pq_meta.num_rows
        print(f"  Total packed batches for this rank: {self._total_rows}")

        # Fingerprint for validation
        self._fingerprint = f"{parquet_path}_{len(self.parquet_files)}_{compression_ratio}_{seed}"

    def _shard_files(self, all_files: List[str], num_processes: int, process_rank: int) -> List[str]:
        """Distribute files across processes."""
        return [f for i, f in enumerate(all_files) if i % num_processes == process_rank]

    def __len__(self) -> int:
        """Return total number of packed batches for this rank."""
        return self._total_rows

    def _shuffle_indices(self, num_rows: int) -> List[int]:
        """Generate shuffled row indices for current file."""
        indices = list(range(num_rows))
        if self.shuffle:
            # Save RNG state before shuffle (for reproducibility on resume)
            self._state['rng_state'] = self._rng.getstate()
            self._rng.shuffle(indices)
        return indices

    def __iter__(self):
        """
        Iterate through packed batches, expanding memory regions on-the-fly.

        Uses while loop to check self._state on each iteration, allowing
        StatefulDataLoader to restore state after iterator creation.

        Yields:
            Collated batch dict ready for model forward pass
        """
        # Use while loop to read state dynamically (not cached at start)
        # This allows load_state_dict to be called after __iter__ starts
        while self._state['file_idx'] < len(self.parquet_files):
            file_idx = self._state['file_idx']
            pq_file = self.parquet_files[file_idx]
            table = pq.read_table(pq_file)
            num_rows = len(table)

            # Generate or restore shuffled indices for this file
            if self._state['row_idx'] == 0:
                # Starting a new file - generate fresh shuffle
                self._current_row_indices = self._shuffle_indices(num_rows)
            elif not self._current_row_indices:
                # Resuming mid-file - restore RNG state and regenerate same shuffle
                if self._state.get('rng_state'):
                    self._rng.setstate(self._state['rng_state'])
                self._current_row_indices = self._shuffle_indices(num_rows)

            # Iterate through shuffled indices
            while self._state['row_idx'] < len(self._current_row_indices):
                # Get the actual row index from shuffled order
                actual_row_idx = self._current_row_indices[self._state['row_idx']]

                # Load packed batch
                packed_batch_bytes = table['packed_batch_bytes'][actual_row_idx].as_py()
                packed_batch = pickle.loads(packed_batch_bytes)

                # Expand all examples in the packed batch
                expanded_examples = []
                for example in packed_batch:
                    expanded = self._expand_example(example)
                    if expanded is not None:
                        expanded_examples.append(expanded)

                # Update state BEFORE yielding (so checkpoint captures next position)
                self._state['row_idx'] += 1

                if not expanded_examples:
                    continue

                # Collate and yield (with optional padding for consistent shapes)
                yield collate_packed_batch(
                    expanded_examples,
                    target_length=self.target_length,
                    pad_token_id=self.decoder_tokenizer.pad_token_id or 0,
                )

            # Move to next file, reset row_idx
            self._state['file_idx'] += 1
            self._state['row_idx'] = 0
            self._current_row_indices = []  # Clear for next file

        # Reset state for next epoch
        self._state['file_idx'] = 0
        self._state['row_idx'] = 0
        self._state['epoch'] += 1
        self._current_row_indices = []

    def _expand_example(self, example: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Expand a single example's memory regions based on runtime compression_ratio.

        Base format (always START, M, END = 3 consecutive tokens per region):
            base_input_ids: [A, B, START, M, END, C, D, START, M, END, E]
            memory_positions: [2, 7]  # Just start indices

        Expanded format (e.g., 10 chunks and 5 chunks):
            input_ids: [A, B, START, M×10, END, C, D, START, M×5, END, E]
            memory_positions: [(3, 13), (18, 23)]  # positions of M tokens only

        Args:
            example: Dict with base_input_ids, base_labels, memory_strings, memory_positions

        Returns:
            Expanded example dict or None if invalid
        """
        base_input_ids = example['base_input_ids']
        base_labels = example['base_labels']
        memory_strings = example.get('memory_strings', [])
        memory_positions = example.get('memory_positions', [])  # List of start indices

        # Handle case with no memory regions
        if not memory_strings or not memory_positions:
            return {
                'processed': {
                    'input_ids': [list(base_input_ids)],
                    'labels': [list(base_labels)],
                    'memory_token_ids': [[]],
                    'memory_positions': [[]],
                    'latent_counts': [[]],
                },
                'seq_len': len(base_input_ids),
            }

        # Validate
        if len(memory_strings) != len(memory_positions):
            print(f"Warning: memory_strings ({len(memory_strings)}) != memory_positions ({len(memory_positions)})")
            return None

        new_input_ids = []
        new_labels = []
        memory_token_ids = []
        memory_positions = []
        latent_counts = []

        # Build set of memory region positions (START, M, END = 3 tokens each)
        memory_region_positions = set()
        for start_idx in memory_positions:
            memory_region_positions.add(start_idx)      # START
            memory_region_positions.add(start_idx + 1)  # M
            memory_region_positions.add(start_idx + 2)  # END

        memory_idx = 0
        i = 0

        while i < len(base_input_ids):
            # Check if we're at the start of a memory region
            if memory_idx < len(memory_positions) and i == memory_positions[memory_idx]:
                start_idx = memory_positions[memory_idx]

                # === Process this memory region ===

                # 1. Tokenize memory string with embed_tokenizer
                memory_str = memory_strings[memory_idx]
                embed_ids = self.embed_tokenizer.encode(memory_str, add_special_tokens=False)
                memory_token_ids.append(embed_ids)

                # 2. Calculate number of chunks (= number of embeddings for all pooling modes)
                num_chunks = max(1, math.ceil(len(embed_ids) / self.compression_ratio))
                latent_counts.append(num_chunks)

                # 3. Add START token
                new_input_ids.append(base_input_ids[start_idx])  # START token
                new_labels.append(-100)

                # 4. Add M × num_chunks
                memory_start_pos = len(new_input_ids)
                new_input_ids.extend([self.memory_id] * num_chunks)
                new_labels.extend([-100] * num_chunks)
                memory_end_pos = len(new_input_ids)
                memory_positions.append((memory_start_pos, memory_end_pos))

                # 5. Add END token
                new_input_ids.append(base_input_ids[start_idx + 2])  # END token
                new_labels.append(-100)

                # 6. Skip past this memory region (3 tokens: START, M, END)
                i = start_idx + 3
                memory_idx += 1
                continue

            # Regular token (not part of a memory region)
            if i not in memory_region_positions:
                new_input_ids.append(base_input_ids[i])
                new_labels.append(base_labels[i])

            i += 1

        return {
            'processed': {
                'input_ids': [new_input_ids],
                'labels': [new_labels],
                'memory_token_ids': [memory_token_ids],
                'memory_positions': [memory_positions],
                'latent_counts': [latent_counts],
            },
            'seq_len': len(new_input_ids),
        }

    # ==================== StatefulDataLoader Interface ====================
    # These methods are called automatically by StatefulDataLoader

    def state_dict(self) -> Dict[str, Any]:
        """
        Save state for checkpointing.

        Called automatically by StatefulDataLoader.state_dict()
        """
        file_idx = self._state['file_idx']
        current_file = self.parquet_files[file_idx] if file_idx < len(self.parquet_files) else None
        return {
            'file_idx': file_idx,
            'row_idx': self._state['row_idx'],
            'epoch': self._state['epoch'],
            'rng_state': self._state['rng_state'],  # RNG state for reproducible row shuffle
            'compression_ratio': self.compression_ratio,
            'fingerprint': self._fingerprint,
            # Save file names for validation
            'current_file': current_file,
            'all_files': self.parquet_files,
            # Save current shuffled indices for exact resume
            'current_row_indices': self._current_row_indices,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """
        Restore state from checkpoint.

        Called automatically by StatefulDataLoader.load_state_dict()
        """
        print(f"[DEBUG] DynamicPackedDataset.load_state_dict called with keys: {list(state_dict.keys())}")

        # Validate fingerprint
        if state_dict.get('fingerprint') != self._fingerprint:
            print(
                f"Warning: Dataset fingerprint mismatch. "
                f"Expected {self._fingerprint}, got {state_dict.get('fingerprint')}. "
                f"State will be loaded but may not resume correctly."
            )

        # Validate file list matches
        saved_files = state_dict.get('all_files', [])
        if saved_files and saved_files != self.parquet_files:
            print(f"Warning: File list mismatch!")
            print(f"  Saved: {len(saved_files)} files, first={saved_files[0] if saved_files else None}")
            print(f"  Current: {len(self.parquet_files)} files, first={self.parquet_files[0] if self.parquet_files else None}")

        # Validate current file matches
        file_idx = state_dict.get('file_idx', 0)
        saved_current = state_dict.get('current_file')
        if saved_current and file_idx < len(self.parquet_files):
            actual_current = self.parquet_files[file_idx]
            if saved_current != actual_current:
                print(f"Warning: Current file mismatch at file_idx={file_idx}!")
                print(f"  Saved: {saved_current}")
                print(f"  Actual: {actual_current}")

        # Warn if compression_ratio changed (still valid, just different expansion)
        if state_dict.get('compression_ratio') != self.compression_ratio:
            print(
                f"Note: compression_ratio changed from {state_dict.get('compression_ratio')} to {self.compression_ratio}. "
                f"Resuming at same data position but with new expansion."
            )

        # Restore state
        self._state['file_idx'] = file_idx
        self._state['row_idx'] = state_dict.get('row_idx', 0)
        self._state['epoch'] = state_dict.get('epoch', 0)

        # Restore RNG state for reproducible shuffling
        rng_state = state_dict.get('rng_state')
        if rng_state:
            self._state['rng_state'] = rng_state
            self._rng.setstate(rng_state)

        # Restore shuffled indices for exact resume mid-file
        self._current_row_indices = state_dict.get('current_row_indices', [])

        current_file = self.parquet_files[self._state['file_idx']] if self._state['file_idx'] < len(self.parquet_files) else "N/A"
        print(
            f"Restored dataset state: file_idx={self._state['file_idx']}, "
            f"row_idx={self._state['row_idx']}, epoch={self._state['epoch']}, "
            f"current_file={current_file}, "
            f"shuffled_indices_len={len(self._current_row_indices)}"
        )


def create_dynamic_dataloader(
    parquet_path: str,
    decoder_tokenizer,
    embed_tokenizer,
    compression_ratio: int,
    accelerator=None,
    seed: int = 42,
    shuffle: bool = True,
    shuffle_files: bool = True,
    drop_last_files: bool = True,
    pooling: str = "mean",
):
    """
    Create a StatefulDataLoader with DynamicPackedDataset.

    Args:
        parquet_path: Path to dynamic packed parquet files
        decoder_tokenizer: LLM tokenizer
        embed_tokenizer: Embed tokenizer
        compression_ratio: Compression chunk size (runtime parameter)
        accelerator: HuggingFace Accelerator instance
        seed: Random seed
        shuffle: Whether to shuffle rows within each file
        shuffle_files: Whether to shuffle file order
        drop_last_files: Drop extra files for even distribution across ranks
        pooling: Pooling strategy ('mean', 'eos', 'concat')

    Returns:
        StatefulDataLoader ready for training
    """
    from torchdata.stateful_dataloader import StatefulDataLoader

    # Get distributed info from accelerator
    if accelerator is not None:
        num_processes = accelerator.num_processes
        process_rank = accelerator.process_index
    else:
        num_processes = 1
        process_rank = 0

    # Create dataset
    dataset = DynamicPackedDataset(
        parquet_path=parquet_path,
        decoder_tokenizer=decoder_tokenizer,
        embed_tokenizer=embed_tokenizer,
        compression_ratio=compression_ratio,
        num_processes=num_processes,
        process_rank=process_rank,
        seed=seed,
        shuffle=shuffle,
        shuffle_files=shuffle_files,
        drop_last_files=drop_last_files,
        pooling=pooling,
    )

    # Create StatefulDataLoader
    # - batch_size=1 because each item is already a packed batch
    # - num_workers=0 required for stateful iteration with IterableDataset
    # - collate_fn just unwraps the single item
    dataloader = StatefulDataLoader(
        dataset,
        batch_size=1,
        shuffle=False,  # Dataset handles ordering
        num_workers=0,  # Required for stateful IterableDataset
        pin_memory=True,
        collate_fn=lambda x: x[0],  # Unwrap single item
    )

    return dataloader
