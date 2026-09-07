"""Deterministic, metadata-only row sharding for parquet datasets."""

from __future__ import annotations

import hashlib
import json
import math
from bisect import bisect_right
from typing import Any, Dict, List, Sequence, Tuple

import pyarrow.parquet as pq


RowManifest = List[Tuple[str, int]]
RowAssignment = Tuple[str, int, int]


def build_row_shard_plan(
    parquet_files: Sequence[str],
    num_processes: int,
    process_rank: int,
    drop_remainder: bool,
) -> Dict[str, Any]:
    """Build one rank's contiguous row assignment from parquet metadata.

    The ordered parquet files define a single global row stream. In drop mode,
    only the final ``total_rows % num_processes`` rows are omitted. Otherwise,
    the global prefix is repeated just enough to give every rank the same number
    of rows. Contiguous rank ranges minimize the number of parquet files each
    process must open.
    """
    if num_processes <= 0:
        raise ValueError(f"num_processes must be positive, got {num_processes}")
    if process_rank < 0 or process_rank >= num_processes:
        raise ValueError(
            f"process_rank must be in [0, {num_processes}), got {process_rank}"
        )

    manifest: RowManifest = []
    for parquet_file in parquet_files:
        num_rows = pq.read_metadata(parquet_file).num_rows
        manifest.append((parquet_file, num_rows))

    total_rows = sum(num_rows for _, num_rows in manifest)
    if total_rows < num_processes:
        raise ValueError(
            "Cannot create a non-empty, equal-length distributed parquet shard: "
            f"found {total_rows} total rows for {num_processes} ranks"
        )

    if drop_remainder:
        rows_per_rank = total_rows // num_processes
        effective_rows = rows_per_rank * num_processes
    else:
        rows_per_rank = math.ceil(total_rows / num_processes)
        effective_rows = rows_per_rank * num_processes

    dropped_rows = total_rows - effective_rows if drop_remainder else 0
    padded_rows = effective_rows - total_rows if not drop_remainder else 0
    rank_start = process_rank * rows_per_rank
    rank_stop = rank_start + rows_per_rank

    # Map the rank's effective global interval back to physical parquet ranges.
    # In non-drop mode, only the small padded suffix wraps to the global prefix.
    file_ends: List[int] = []
    running_total = 0
    for _, num_rows in manifest:
        running_total += num_rows
        file_ends.append(running_total)

    assignments: List[RowAssignment] = []
    cursor = rank_start
    while cursor < rank_stop:
        physical_row = cursor % total_rows
        file_idx = bisect_right(file_ends, physical_row)
        file_start = file_ends[file_idx - 1] if file_idx else 0
        parquet_file, file_num_rows = manifest[file_idx]
        local_start = physical_row - file_start
        available = file_num_rows - local_start
        take = min(rank_stop - cursor, available)
        assignments.append((parquet_file, local_start, local_start + take))
        cursor += take

    assigned_rows = sum(stop - start for _, start, stop in assignments)
    if assigned_rows != rows_per_rank:
        raise RuntimeError(
            "Row shard planning invariant failed: "
            f"assigned {assigned_rows}, expected {rows_per_rank}"
        )

    return {
        "manifest": manifest,
        "assignments": assignments,
        "total_rows": total_rows,
        "rows_per_rank": rows_per_rank,
        "dropped_rows": dropped_rows,
        "padded_rows": padded_rows,
    }


def row_shard_fingerprint(
    *,
    dataset_kind: str,
    manifest: Sequence[Tuple[str, int]],
    assignments: Sequence[RowAssignment],
    num_processes: int,
    process_rank: int,
    drop_remainder: bool,
    seed: int,
    shuffle: bool,
) -> str:
    """Return a stable fingerprint that includes the exact rank assignment."""
    payload = {
        "version": 1,
        "dataset_kind": dataset_kind,
        "manifest": [list(item) for item in manifest],
        "assignments": [list(item) for item in assignments],
        "num_processes": num_processes,
        "process_rank": process_rank,
        "drop_remainder": drop_remainder,
        "seed": seed,
        "shuffle": shuffle,
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()

