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
from data.parquet_row_sharding import build_row_shard_plan, row_shard_fingerprint


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
            drop_last_files: Drop the global row remainder when it cannot be split
                evenly across ranks. If false, pad deterministically from the global
                row prefix so every rank still has the same length.
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

        # Treat the ordered files as one global row stream, then assign an equal
        # contiguous row interval to each rank. Only parquet metadata is read here.
        row_plan = build_row_shard_plan(
            all_files,
            num_processes=num_processes,
            process_rank=process_rank,
            drop_remainder=drop_last_files,
        )
        self._source_manifest = row_plan["manifest"]
        self._row_assignments: List[Tuple[str, int, int]] = row_plan["assignments"]
        # Retain this public attribute for compatibility. A path can occur twice
        # when non-drop padding wraps to the beginning of the global row stream.
        self.parquet_files = [path for path, _, _ in self._row_assignments]
        self._total_rows = row_plan["rows_per_rank"]

        # Log file assignment
        print(
            f"[Rank {process_rank}/{num_processes}] DynamicPackedDataset: "
            f"{len(self._row_assignments)} row ranges across "
            f"{len(set(self.parquet_files))}/{len(all_files)} parquet files"
        )
        if self.parquet_files:
            print(f"  Files: {self.parquet_files[0]} ... {self.parquet_files[-1]}")
        print(f"  Shuffle files: {shuffle_files}, Shuffle rows: {shuffle}, Drop last: {drop_last_files}")
        if row_plan["dropped_rows"]:
            print(f"  Dropped global remainder: {row_plan['dropped_rows']} rows")
        if row_plan["padded_rows"]:
            print(f"  Deterministic global-prefix padding: {row_plan['padded_rows']} rows")

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
        self._current_shuffle_rng_state = None

        print(f"  Total packed batches for this rank: {self._total_rows}")

        # Fingerprint the source manifest and exact rank-local row ranges so a
        # checkpoint cannot silently resume against a different assignment.
        self._fingerprint = row_shard_fingerprint(
            dataset_kind="dynamic_packed",
            manifest=self._source_manifest,
            assignments=self._row_assignments,
            num_processes=num_processes,
            process_rank=process_rank,
            drop_remainder=drop_last_files,
            seed=seed,
            shuffle=shuffle,
        )

    def __len__(self) -> int:
        """Return total number of packed batches for this rank."""
        return self._total_rows

    def _shuffle_indices(self, row_start: int, row_stop: int) -> List[int]:
        """Generate shuffled indices for the current assigned row range."""
        indices = list(range(row_start, row_stop))
        if self.shuffle:
            self._current_shuffle_rng_state = self._rng.getstate()
            self._rng.shuffle(indices)
            # Save the post-shuffle state needed by the next assignment.
            self._state['rng_state'] = self._rng.getstate()
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
        while self._state['file_idx'] < len(self._row_assignments):
            file_idx = self._state['file_idx']
            pq_file, row_start, row_stop = self._row_assignments[file_idx]
            table = pq.read_table(pq_file)
            num_rows = len(table)
            if row_start < 0 or row_stop > num_rows or row_start >= row_stop:
                raise RuntimeError(
                    "Parquet row assignment is invalid after reading the file: "
                    f"file={pq_file!r}, range=[{row_start}, {row_stop}), "
                    f"num_rows={num_rows}"
                )

            # Generate or restore shuffled indices for this file
            if self._state['row_idx'] == 0:
                # Starting a new file - generate fresh shuffle
                self._current_row_indices = self._shuffle_indices(row_start, row_stop)
            elif not self._current_row_indices:
                # New checkpoints always carry the exact order. Retain a
                # deterministic fallback for checkpoint serializers that omit it.
                if self.shuffle and self._current_shuffle_rng_state is None:
                    raise ValueError(
                        "Cannot regenerate the current shuffled row range: "
                        "checkpoint is missing current_shuffle_rng_state"
                    )
                self._current_row_indices = list(range(row_start, row_stop))
                if self.shuffle:
                    replay_rng = random.Random()
                    replay_rng.setstate(self._current_shuffle_rng_state)
                    replay_rng.shuffle(self._current_row_indices)

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
                    raise ValueError(
                        "Packed parquet row contains no valid examples; skipping it "
                        "would make distributed ranks take different numbers of steps: "
                        f"file={pq_file!r}, row={actual_row_idx}"
                    )

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
            self._current_shuffle_rng_state = None

        # Reset state for next epoch
        self._state['file_idx'] = 0
        self._state['row_idx'] = 0
        self._state['epoch'] += 1
        self._current_row_indices = []
        self._current_shuffle_rng_state = None

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
        # Keep source positions separate from the expanded output positions. Reusing
        # this name for both used to clear the source list before expansion.
        source_memory_positions = example.get('memory_positions', [])  # List of START indices

        if len(base_input_ids) != len(base_labels):
            print(
                "Warning: base_input_ids and base_labels have different lengths "
                f"({len(base_input_ids)} != {len(base_labels)})"
            )
            return None

        has_memory_strings = bool(memory_strings)
        has_memory_positions = bool(source_memory_positions)

        # Both fields must either be empty (an intentionally uncompressed example)
        # or populated. Treat one-sided metadata as corruption instead of silently
        # converting the example into a no-memory example.
        if has_memory_strings != has_memory_positions:
            print(
                "Warning: one-sided memory metadata: "
                f"memory_strings={len(memory_strings)}, "
                f"memory_positions={len(source_memory_positions)}"
            )
            return None

        # Handle an intentionally uncompressed example.
        if not has_memory_strings:
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
        if len(memory_strings) != len(source_memory_positions):
            print(
                f"Warning: memory_strings ({len(memory_strings)}) != "
                f"memory_positions ({len(source_memory_positions)})"
            )
            return None

        # Validate the compact source representation before indexing into it. Each
        # source position must identify a non-overlapping START, M, END triplet.
        previous_region_end = -1
        for region_idx, start_idx in enumerate(source_memory_positions):
            if not isinstance(start_idx, int):
                print(
                    f"Warning: memory position {region_idx} is not an integer: "
                    f"{start_idx!r}"
                )
                return None
            if start_idx <= previous_region_end:
                print(
                    f"Warning: memory position {region_idx} ({start_idx}) is not "
                    "strictly ordered or overlaps the previous region"
                )
                return None
            if start_idx < 0 or start_idx + 2 >= len(base_input_ids):
                print(
                    f"Warning: memory position {region_idx} ({start_idx}) is out "
                    f"of bounds for sequence length {len(base_input_ids)}"
                )
                return None
            if (
                base_input_ids[start_idx] != self.memory_start_id
                or base_input_ids[start_idx + 1] != self.memory_id
                or base_input_ids[start_idx + 2] != self.memory_end_id
            ):
                print(
                    f"Warning: memory position {region_idx} ({start_idx}) does not "
                    "point to a START, M, END triplet"
                )
                return None
            previous_region_end = start_idx + 2

        if self.compression_ratio <= 0:
            raise ValueError(
                f"compression_ratio must be positive, got {self.compression_ratio}"
            )

        new_input_ids = []
        new_labels = []
        memory_token_ids = []
        expanded_memory_positions = []
        latent_counts = []

        # Build set of memory region positions (START, M, END = 3 tokens each)
        memory_region_positions = set()
        for start_idx in source_memory_positions:
            memory_region_positions.add(start_idx)      # START
            memory_region_positions.add(start_idx + 1)  # M
            memory_region_positions.add(start_idx + 2)  # END

        memory_idx = 0
        i = 0

        while i < len(base_input_ids):
            # Check if we're at the start of a memory region
            if (
                memory_idx < len(source_memory_positions)
                and i == source_memory_positions[memory_idx]
            ):
                start_idx = source_memory_positions[memory_idx]

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
                expanded_memory_positions.append((memory_start_pos, memory_end_pos))

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

        # These are internal expansion invariants. Malformed source data is rejected
        # above; reaching this point with inconsistent output indicates a code bug.
        num_regions = len(memory_strings)
        if not (
            memory_idx
            == len(memory_token_ids)
            == len(expanded_memory_positions)
            == len(latent_counts)
            == num_regions
        ):
            raise RuntimeError(
                "Memory expansion invariant failed: "
                f"consumed={memory_idx}, token_ids={len(memory_token_ids)}, "
                f"positions={len(expanded_memory_positions)}, "
                f"latent_counts={len(latent_counts)}, expected={num_regions}"
            )
        if len(new_input_ids) != len(new_labels):
            raise RuntimeError(
                "Memory expansion invariant failed: input_ids and labels have "
                f"different lengths ({len(new_input_ids)} != {len(new_labels)})"
            )
        for region_idx, ((start_pos, end_pos), latent_count) in enumerate(
            zip(expanded_memory_positions, latent_counts)
        ):
            if end_pos - start_pos != latent_count:
                raise RuntimeError(
                    f"Memory expansion invariant failed for region {region_idx}: "
                    f"position width {end_pos - start_pos} != latent count {latent_count}"
                )
            if any(
                token_id != self.memory_id
                for token_id in new_input_ids[start_pos:end_pos]
            ):
                raise RuntimeError(
                    f"Memory expansion invariant failed for region {region_idx}: "
                    "expanded span contains a non-memory token"
                )

        return {
            'processed': {
                'input_ids': [new_input_ids],
                'labels': [new_labels],
                'memory_token_ids': [memory_token_ids],
                'memory_positions': [expanded_memory_positions],
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
        current_assignment = (
            self._row_assignments[file_idx]
            if file_idx < len(self._row_assignments)
            else None
        )
        current_file = current_assignment[0] if current_assignment else None
        return {
            'file_idx': file_idx,
            'row_idx': self._state['row_idx'],
            'epoch': self._state['epoch'],
            'rng_state': self._state['rng_state'],  # RNG state for reproducible row shuffle
            'compression_ratio': self.compression_ratio,
            'fingerprint': self._fingerprint,
            # Save both compatibility file names and the exact row plan.
            'current_file': current_file,
            'current_assignment': current_assignment,
            'all_files': self.parquet_files,
            'source_manifest': self._source_manifest,
            'row_assignments': self._row_assignments,
            # Save current shuffled indices for exact resume
            'current_row_indices': self._current_row_indices,
            'current_shuffle_rng_state': self._current_shuffle_rng_state,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """
        Restore state from checkpoint.

        Called automatically by StatefulDataLoader.load_state_dict()
        """
        print(f"[DEBUG] DynamicPackedDataset.load_state_dict called with keys: {list(state_dict.keys())}")

        saved_fingerprint = state_dict.get('fingerprint')
        if saved_fingerprint != self._fingerprint:
            raise ValueError(
                "Cannot restore DynamicPackedDataset: row-assignment fingerprint "
                f"mismatch (expected {self._fingerprint}, got {saved_fingerprint}). "
                "The parquet manifest, world size, rank, or remainder policy changed."
            )

        saved_assignments = [
            tuple(assignment) for assignment in state_dict.get('row_assignments', [])
        ]
        if saved_assignments != self._row_assignments:
            raise ValueError(
                "Cannot restore DynamicPackedDataset: checkpoint row assignments do "
                "not match the current rank's assignments"
            )

        # Validate current assignment and cursor.
        file_idx = state_dict.get('file_idx', 0)
        if not isinstance(file_idx, int) or not 0 <= file_idx <= len(self._row_assignments):
            raise ValueError(f"Invalid checkpoint file_idx: {file_idx!r}")
        row_idx = state_dict.get('row_idx', 0)
        if not isinstance(row_idx, int) or row_idx < 0:
            raise ValueError(f"Invalid checkpoint row_idx: {row_idx!r}")
        saved_current = state_dict.get('current_assignment')
        if file_idx < len(self._row_assignments):
            actual_current = self._row_assignments[file_idx]
            if saved_current is None or tuple(saved_current) != actual_current:
                raise ValueError(
                    "Cannot restore DynamicPackedDataset: current row assignment does "
                    f"not match at index {file_idx}"
                )
            assignment_length = actual_current[2] - actual_current[1]
            if row_idx > assignment_length:
                raise ValueError(
                    f"Invalid checkpoint row_idx {row_idx} for assignment length "
                    f"{assignment_length}"
                )
        elif row_idx != 0:
            raise ValueError("Completed checkpoint must have row_idx=0")

        # Warn if compression_ratio changed (still valid, just different expansion)
        if state_dict.get('compression_ratio') != self.compression_ratio:
            print(
                f"Note: compression_ratio changed from {state_dict.get('compression_ratio')} to {self.compression_ratio}. "
                f"Resuming at same data position but with new expansion."
            )

        # Restore state
        self._state['file_idx'] = file_idx
        self._state['row_idx'] = row_idx
        self._state['epoch'] = state_dict.get('epoch', 0)

        # Restore RNG state for reproducible shuffling
        rng_state = state_dict.get('rng_state')
        if rng_state:
            self._state['rng_state'] = rng_state
            self._rng.setstate(rng_state)

        # Restore and validate the exact current row order.
        self._current_row_indices = state_dict.get('current_row_indices', [])
        if file_idx < len(self._row_assignments) and self._current_row_indices:
            _, row_start, row_stop = self._row_assignments[file_idx]
            if (
                len(self._current_row_indices) != row_stop - row_start
                or set(self._current_row_indices) != set(range(row_start, row_stop))
            ):
                raise ValueError(
                    "Cannot restore DynamicPackedDataset: current_row_indices is not "
                    "a permutation of the current assigned row range"
                )
        self._current_shuffle_rng_state = state_dict.get('current_shuffle_rng_state')

        current_file = (
            self._row_assignments[self._state['file_idx']][0]
            if self._state['file_idx'] < len(self._row_assignments)
            else "N/A"
        )
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
        drop_last_files: Drop the global row remainder for even distribution;
            otherwise pad deterministically from the global row prefix
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
