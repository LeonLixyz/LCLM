"""Recognize flat packs and the published multi-component packed snapshot."""
from pathlib import Path


def discover_packed_parquet_files(directory):
    root=Path(directory)
    flat=sorted(p for p in root.glob('*.parquet') if p.is_file())
    # Deliberately not a recursive glob: raw datasets, in-progress outputs and
    # quarantines must never accidentally enter the training row stream.
    release=sorted(p for p in root.glob('data/*/part-*/*.parquet') if p.is_file())
    if flat and release:
        raise ValueError('Ambiguous packed input: both flat and release-layout shards exist')
    selected=flat or release
    if not selected:raise ValueError(f'No packed parquet files found in {directory}')
    return [str(p) for p in selected]
