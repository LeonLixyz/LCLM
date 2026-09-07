"""
Stateful dataset for non-packed training with checkpoint resumption.

Streams raw parquet rows and aliases configured prompt/target columns.
Tokenization happens in collate function, not in dataset.

Works with StatefulDataLoader for auto-resume checkpointing.
Supports file-level and row-level shuffling with deterministic resume.
"""

import os
import glob
import random
from typing import Dict, List, Any, Optional, Tuple

import pyarrow.parquet as pq
from torch.utils.data import IterableDataset

from data.parquet_row_sharding import build_row_shard_plan, row_shard_fingerprint


class StatefulDataset(IterableDataset):
    """
    Stateful dataset that streams raw text data from parquet files.

    Features:
    - File-level shuffling (deterministic with seed)
    - Row-level shuffling within each file (with RNG state for resume)
    - StatefulDataLoader compatible for auto-resume checkpointing
    - Yields every parquet column plus canonical prompt/target aliases

    With StatefulDataLoader:
    - state_dict() is called automatically when checkpointing
    - load_state_dict() is called automatically when resuming
    - No manual skipping needed

    Usage:
        from torchdata.stateful_dataloader import StatefulDataLoader

        dataset = StatefulDataset(...)
        dataloader = StatefulDataLoader(dataset, batch_size=4, num_workers=0)

        # Accelerate handles save/load automatically
        dataloader = accelerator.prepare(dataloader)
    """

    def __init__(
        self,
        parquet_path: str,
        num_processes: int = 1,
        process_rank: int = 0,
        seed: int = 42,
        shuffle: bool = True,
        shuffle_files: bool = True,
        drop_last_files: bool = True,
        prompt_column: str = "prompt",
        target_column: str = "target",
    ):
        """
        Args:
            parquet_path: Path to directory containing parquet files, or single parquet file
            num_processes: Total number of distributed processes
            process_rank: This process's rank
            seed: Random seed for reproducibility
            shuffle: Whether to shuffle rows within each file
            shuffle_files: Whether to shuffle file order
            drop_last_files: Drop the global row remainder when it cannot be split
                evenly across ranks. If false, pad deterministically from the global
                row prefix so every rank still has the same length.
            prompt_column: Name of prompt column in parquet
            target_column: Name of target column in parquet
        """
        self.parquet_path = parquet_path
        self.num_processes = num_processes
        self.process_rank = process_rank
        self.seed = seed
        self.shuffle = shuffle
        self.shuffle_files = shuffle_files
        self.drop_last_files = drop_last_files
        self.prompt_column = prompt_column
        self.target_column = target_column

        # Get all parquet files
        if os.path.isfile(parquet_path):
            all_files = [parquet_path]
        else:
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
            f"[Rank {process_rank}/{num_processes}] StatefulDataset: "
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

        print(f"  Total rows for this rank: {self._total_rows}")

        # Fingerprint the source manifest and exact rank-local row ranges so a
        # checkpoint cannot silently resume against a different assignment.
        self._fingerprint = row_shard_fingerprint(
            dataset_kind="stateful",
            manifest=self._source_manifest,
            assignments=self._row_assignments,
            num_processes=num_processes,
            process_rank=process_rank,
            drop_remainder=drop_last_files,
            seed=seed,
            shuffle=shuffle,
        )

    def __len__(self) -> int:
        """Return total number of rows for this rank."""
        return self._total_rows

    def _shuffle_indices(self, row_start: int, row_stop: int) -> List[int]:
        """Generate shuffled indices for the current assigned row range."""
        indices = list(range(row_start, row_stop))
        if self.shuffle:
            self._current_shuffle_rng_state = self._rng.getstate()
            self._rng.shuffle(indices)
            # This is the state needed to shuffle the *next* assignment. Keeping
            # the pre-shuffle state here caused post-resume order to diverge at
            # the next file boundary.
            self._state['rng_state'] = self._rng.getstate()
        return indices

    def __iter__(self):
        """
        Iterate through rows, yielding raw prompt/target dicts.

        Uses while loop to check self._state on each iteration, allowing
        StatefulDataLoader to restore state after iterator creation.

        Yields:
            Full parquet row with configured columns also available under the
            canonical 'prompt' and 'target' keys
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

                # Preserve native records (for example agent messages/tools) while
                # retaining the canonical aliases expected by existing collators.
                row = {
                    column: table[column][actual_row_idx].as_py()
                    for column in table.column_names
                }
                row["prompt"] = row[self.prompt_column]
                row["target"] = row[self.target_column]

                # Update state BEFORE yielding (so checkpoint captures next position)
                self._state['row_idx'] += 1

                yield row

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
        print(f"[DEBUG] StatefulDataset.load_state_dict called with keys: {list(state_dict.keys())}")

        saved_fingerprint = state_dict.get('fingerprint')
        if saved_fingerprint != self._fingerprint:
            raise ValueError(
                "Cannot restore StatefulDataset: row-assignment fingerprint "
                f"mismatch (expected {self._fingerprint}, got {saved_fingerprint}). "
                "The parquet manifest, world size, rank, or remainder policy changed."
            )

        saved_assignments = [
            tuple(assignment) for assignment in state_dict.get('row_assignments', [])
        ]
        if saved_assignments != self._row_assignments:
            raise ValueError(
                "Cannot restore StatefulDataset: checkpoint row assignments do "
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
                    "Cannot restore StatefulDataset: current row assignment does "
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
                    "Cannot restore StatefulDataset: current_row_indices is not "
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


def create_stateful_dataloader(
    parquet_path: str,
    collate_fn,
    batch_size: int = 1,
    accelerator=None,
    seed: int = 42,
    shuffle: bool = True,
    shuffle_files: bool = True,
    drop_last_files: bool = True,
    prompt_column: str = "prompt",
    target_column: str = "target",
):
    """
    Create a StatefulDataLoader with StatefulDataset.

    Args:
        parquet_path: Path to parquet files (directory or single file)
        collate_fn: Collate function for tokenization
        batch_size: Batch size
        accelerator: HuggingFace Accelerator instance
        seed: Random seed
        shuffle: Whether to shuffle rows within each file
        shuffle_files: Whether to shuffle file order
        drop_last_files: Drop the global row remainder for even distribution;
            otherwise pad deterministically from the global row prefix
        prompt_column: Name of prompt column
        target_column: Name of target column

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
    dataset = StatefulDataset(
        parquet_path=parquet_path,
        num_processes=num_processes,
        process_rank=process_rank,
        seed=seed,
        shuffle=shuffle,
        shuffle_files=shuffle_files,
        drop_last_files=drop_last_files,
        prompt_column=prompt_column,
        target_column=target_column,
    )

    # Create StatefulDataLoader
    # - num_workers=0 required for stateful iteration with IterableDataset
    dataloader = StatefulDataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # Dataset handles ordering
        num_workers=0,  # Required for stateful IterableDataset
        pin_memory=True,
        collate_fn=collate_fn,
    )

    return dataloader
