"""
Stateful dataset for non-packed training with checkpoint resumption.

Streams raw parquet files with prompt/target columns.
Tokenization happens in collate function, not in dataset.

Works with StatefulDataLoader for auto-resume checkpointing.
Supports file-level and row-level shuffling with deterministic resume.
"""

import os
import glob
import random
from typing import Dict, List, Any, Optional

import pyarrow.parquet as pq
from torch.utils.data import IterableDataset


class StatefulDataset(IterableDataset):
    """
    Stateful dataset that streams raw text data from parquet files.

    Features:
    - File-level shuffling (deterministic with seed)
    - Row-level shuffling within each file (with RNG state for resume)
    - StatefulDataLoader compatible for auto-resume checkpointing
    - Yields raw prompt/target dicts (tokenization in collate)

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
            drop_last_files: Drop extra files so each rank gets equal number (prevents hangs)
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
            f"[Rank {process_rank}/{num_processes}] StatefulDataset: "
            f"{len(self.parquet_files)}/{len(all_files)} parquet files"
        )
        if self.parquet_files:
            print(f"  Files: {self.parquet_files[0]} ... {self.parquet_files[-1]}")
        print(f"  Shuffle files: {shuffle_files}, Shuffle rows: {shuffle}, Drop last: {drop_last_files}")

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

        # Count total rows for __len__
        self._total_rows = 0
        for pq_file in self.parquet_files:
            pq_meta = pq.read_metadata(pq_file)
            self._total_rows += pq_meta.num_rows
        print(f"  Total rows for this rank: {self._total_rows}")

        # Fingerprint for validation
        self._fingerprint = f"{parquet_path}_{len(self.parquet_files)}_{seed}"

    def _shard_files(self, all_files: List[str], num_processes: int, process_rank: int) -> List[str]:
        """Distribute files across processes."""
        return [f for i, f in enumerate(all_files) if i % num_processes == process_rank]

    def __len__(self) -> int:
        """Return total number of rows for this rank."""
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
        Iterate through rows, yielding raw prompt/target dicts.

        Uses while loop to check self._state on each iteration, allowing
        StatefulDataLoader to restore state after iterator creation.

        Yields:
            Dict with 'prompt' and 'target' keys (raw text)
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

                # Get raw data
                prompt = table[self.prompt_column][actual_row_idx].as_py()
                target = table[self.target_column][actual_row_idx].as_py()

                # Update state BEFORE yielding (so checkpoint captures next position)
                self._state['row_idx'] += 1

                yield {"prompt": prompt, "target": target}

            # Move to next file, reset row_idx
            self._state['file_idx'] += 1
            self._state['row_idx'] = 0
            self._current_row_indices = []  # Clear for next file

        # Reset state for next epoch
        self._state['file_idx'] = 0
        self._state['row_idx'] = 0
        self._state['epoch'] += 1
        self._current_row_indices = []

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
        print(f"[DEBUG] StatefulDataset.load_state_dict called with keys: {list(state_dict.keys())}")

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
        drop_last_files: Drop extra files for even distribution across ranks
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
