#!/usr/bin/env python3
"""Audit or rewrite Stage-3 reasoning targets on a Modal Volume.

Only ``reasoning_data`` and ``dolci_think`` rows are changed:

* ``compression_prompt`` is replaced with the uncompressed ``prompt``.
* the reasoning prefix in ``target`` is wrapped in LCLM memory markers.
* an explicit final-answer boundary is preferred; otherwise the final 128
  Qwen3-4B-Instruct tokens remain uncompressed.

The rewrite is immutable: output shards and reports live under a separate
directory on the same Volume.
"""

from __future__ import annotations

import collections
import itertools
import json
import os
from pathlib import Path
from typing import Any

import modal


APP_NAME = "lclm-stage3-reasoning-rewrite"
ALGORITHM_VERSION = 3
VOLUME_NAME = "lclm-stage3-data"
VOLUME_ROOT = Path("/data")
DEFAULT_SOURCE_DIR = "stage3-final-mixture"
DEFAULT_OUTPUT_DIR = "stage3-final-mixture-cot50-v1"
DEFAULT_REPORT_DIR = "stage3-cleaning/cot50-rewrite-v1"
REASONING_SUBDATASETS = frozenset({"reasoning_data", "dolci_think"})
REASONING_COMPRESSION_FRACTION = 0.5

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "pyarrow>=17,<22",
        "transformers>=4.51,<5",
        "huggingface-hub>=0.30",
    )
    .add_local_python_source("data")
)
volume = modal.Volume.from_name(VOLUME_NAME)
app = modal.App(APP_NAME)


def _empty_stats(shard: str) -> dict[str, Any]:
    return {
        "algorithm_version": ALGORITHM_VERSION,
        "shard": shard,
        "rows": 0,
        "selected_rows": 0,
        "assigned_compressed_rows": 0,
        "assigned_uncompressed_rows": 0,
        "changed_rows": 0,
        "short_uncompressed_rows": 0,
        "preexisting_target_memory_rows": 0,
        "methods": {},
        "subdatasets": {},
        "compressed_subdatasets": {},
        "uncompressed_subdatasets": {},
        "examples": {},
    }


def _merge_stats(parts: list[dict[str, Any]]) -> dict[str, Any]:
    total = _empty_stats("ALL")
    total["shards"] = len(parts)
    for part in parts:
        for key in (
            "rows",
            "selected_rows",
            "assigned_compressed_rows",
            "assigned_uncompressed_rows",
            "changed_rows",
            "short_uncompressed_rows",
            "preexisting_target_memory_rows",
        ):
            total[key] += part[key]
        for field in (
            "methods",
            "subdatasets",
            "compressed_subdatasets",
            "uncompressed_subdatasets",
        ):
            merged = collections.Counter(total[field])
            merged.update(part[field])
            total[field] = dict(sorted(merged.items()))
        for method, examples in part["examples"].items():
            bucket = total["examples"].setdefault(method, [])
            bucket.extend(examples[: max(0, 5 - len(bucket))])
            del bucket[5:]
    return total


@app.cls(
    image=image,
    cpu=4,
    memory=16384,
    timeout=60 * 60,
    max_containers=64,
    volumes={str(VOLUME_ROOT): volume},
    secrets=[modal.Secret.from_name("huggingface")],
)
class ReasoningRewriter:
    @modal.enter()
    def load_tokenizer(self) -> None:
        from transformers import AutoTokenizer

        from data.build_stage3_agent_mixture import DEFAULT_DECODER_TOKENIZER

        self.tokenizer = AutoTokenizer.from_pretrained(
            DEFAULT_DECODER_TOKENIZER,
            use_fast=True,
        )

    def _split(self, target: str):
        from data.build_stage3_agent_mixture import split_reasoning_target

        return split_reasoning_target(
            target,
            fallback_tokenizer=self.tokenizer,
            fallback_answer_tokens=128,
        )

    @staticmethod
    def _compress_row(relative_path: str, row_index: int) -> bool:
        from data.build_stage3_agent_mixture import reasoning_row_is_compressed

        return reasoning_row_is_compressed(
            f"{relative_path}:{row_index}",
            REASONING_COMPRESSION_FRACTION,
        )

    @modal.method()
    def audit_shard(self, relative_path: str, source_dir: str) -> dict[str, Any]:
        import pyarrow.parquet as pq

        source_path = VOLUME_ROOT / source_dir / relative_path
        stats = _empty_stats(relative_path)
        parquet = pq.ParquetFile(source_path)
        for batch in parquet.iter_batches(
            batch_size=128,
            columns=["sub_dataset", "target"],
        ):
            row_start = stats["rows"]
            subdatasets = batch.column(0).to_pylist()
            targets = batch.column(1).to_pylist()
            stats["rows"] += len(targets)
            for index, (subdataset, target) in enumerate(zip(subdatasets, targets)):
                if subdataset not in REASONING_SUBDATASETS:
                    continue
                stats["selected_rows"] += 1
                stats["subdatasets"][subdataset] = (
                    stats["subdatasets"].get(subdataset, 0) + 1
                )
                if not isinstance(target, str):
                    raise TypeError(
                        f"{relative_path}: selected target is {type(target).__name__}"
                    )
                if "<|memory_start|>" in target or "<|memory_end|>" in target:
                    stats["preexisting_target_memory_rows"] += 1
                    continue
                if not self._compress_row(relative_path, row_start + index):
                    stats["assigned_uncompressed_rows"] += 1
                    stats["uncompressed_subdatasets"][subdataset] = (
                        stats["uncompressed_subdatasets"].get(subdataset, 0) + 1
                    )
                    continue
                stats["assigned_compressed_rows"] += 1
                stats["compressed_subdatasets"][subdataset] = (
                    stats["compressed_subdatasets"].get(subdataset, 0) + 1
                )
                split = self._split(target)
                if split is None:
                    stats["short_uncompressed_rows"] += 1
                    continue
                stats["changed_rows"] += 1
                stats["methods"][split.method] = stats["methods"].get(split.method, 0) + 1
                examples = stats["examples"].setdefault(split.method, [])
                if len(examples) < 1:
                    examples.append(
                        {
                            "analysis_tail": split.analysis[-240:],
                            "answer_head": split.suffix[:240],
                        }
                    )
        return stats

    @modal.method()
    def rewrite_shard(
        self,
        relative_path: str,
        source_dir: str,
        output_dir: str,
        overwrite: bool,
    ) -> dict[str, Any]:
        import pyarrow as pa
        import pyarrow.parquet as pq

        from data.build_stage3_agent_mixture import wrap_reasoning_split

        source_path = VOLUME_ROOT / source_dir / relative_path
        output_path = VOLUME_ROOT / output_dir / relative_path
        stats_path = output_path.with_suffix(output_path.suffix + ".stats.json")
        if output_path.exists() and stats_path.exists() and not overwrite:
            with stats_path.open() as handle:
                cached_stats = json.load(handle)
            # Early audit versions treated any Markdown H4 (``####``) as an
            # answer boundary. Transparently repair those shards on resume.
            if (
                cached_stats.get("algorithm_version") == ALGORITHM_VERSION
                and "marker:####" not in cached_stats.get("methods", {})
            ):
                return cached_stats

        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
        temporary_stats = stats_path.with_suffix(stats_path.suffix + ".tmp")
        stats = _empty_stats(relative_path)
        parquet = pq.ParquetFile(source_path)
        schema = parquet.schema_arrow
        prompt_index = schema.get_field_index("prompt")
        compression_index = schema.get_field_index("compression_prompt")
        target_index = schema.get_field_index("target")
        subdataset_index = schema.get_field_index("sub_dataset")
        if min(prompt_index, compression_index, target_index, subdataset_index) < 0:
            raise ValueError(f"{relative_path}: missing a required Stage-3 column")

        writer = pq.ParquetWriter(
            temporary_path,
            schema,
            compression="zstd",
            use_dictionary=True,
        )
        try:
            for batch in parquet.iter_batches(batch_size=128):
                row_start = stats["rows"]
                stats["rows"] += batch.num_rows
                prompts = batch.column(prompt_index).to_pylist()
                compression_prompts = batch.column(compression_index).to_pylist()
                targets = batch.column(target_index).to_pylist()
                subdatasets = batch.column(subdataset_index).to_pylist()
                for index, (subdataset, target) in enumerate(zip(subdatasets, targets)):
                    if subdataset not in REASONING_SUBDATASETS:
                        continue
                    stats["selected_rows"] += 1
                    stats["subdatasets"][subdataset] = (
                        stats["subdatasets"].get(subdataset, 0) + 1
                    )
                    compression_prompts[index] = prompts[index]
                    if not isinstance(target, str):
                        raise TypeError(
                            f"{relative_path}: selected target is {type(target).__name__}"
                        )
                    if "<|memory_start|>" in target or "<|memory_end|>" in target:
                        stats["preexisting_target_memory_rows"] += 1
                        continue
                    if not self._compress_row(relative_path, row_start + index):
                        stats["assigned_uncompressed_rows"] += 1
                        stats["uncompressed_subdatasets"][subdataset] = (
                            stats["uncompressed_subdatasets"].get(subdataset, 0) + 1
                        )
                        continue
                    stats["assigned_compressed_rows"] += 1
                    stats["compressed_subdatasets"][subdataset] = (
                        stats["compressed_subdatasets"].get(subdataset, 0) + 1
                    )
                    split = self._split(target)
                    if split is None:
                        stats["short_uncompressed_rows"] += 1
                        continue
                    rewritten = wrap_reasoning_split(split)
                    if rewritten.replace("<|memory_start|>", "").replace(
                        "<|memory_end|>", ""
                    ) != target:
                        raise AssertionError(f"{relative_path}: target text changed")
                    targets[index] = rewritten
                    stats["changed_rows"] += 1
                    stats["methods"][split.method] = (
                        stats["methods"].get(split.method, 0) + 1
                    )

                arrays = list(batch.columns)
                arrays[compression_index] = pa.array(
                    compression_prompts,
                    type=schema.field(compression_index).type,
                )
                arrays[target_index] = pa.array(
                    targets,
                    type=schema.field(target_index).type,
                )
                writer.write_batch(pa.RecordBatch.from_arrays(arrays, schema=schema))
        except BaseException:
            writer.close()
            temporary_path.unlink(missing_ok=True)
            temporary_stats.unlink(missing_ok=True)
            raise
        else:
            writer.close()

        with temporary_stats.open("w") as handle:
            json.dump(stats, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary_path, output_path)
        os.replace(temporary_stats, stats_path)
        volume.commit()
        return stats

    @modal.method()
    def validate_shard(
        self,
        relative_path: str,
        source_dir: str,
        output_dir: str,
    ) -> dict[str, Any]:
        import pyarrow.parquet as pq

        from data.build_stage3_agent_mixture import wrap_reasoning_split

        source_path = VOLUME_ROOT / source_dir / relative_path
        output_path = VOLUME_ROOT / output_dir / relative_path
        source = pq.ParquetFile(source_path)
        output = pq.ParquetFile(output_path)
        if not source.schema_arrow.equals(output.schema_arrow, check_metadata=True):
            raise AssertionError(f"{relative_path}: output schema changed")
        if source.metadata.num_rows != output.metadata.num_rows:
            raise AssertionError(f"{relative_path}: output row count changed")

        schema = source.schema_arrow
        prompt_index = schema.get_field_index("prompt")
        compression_index = schema.get_field_index("compression_prompt")
        target_index = schema.get_field_index("target")
        subdataset_index = schema.get_field_index("sub_dataset")
        stats = _empty_stats(relative_path)
        source_batches = source.iter_batches(batch_size=128)
        output_batches = output.iter_batches(batch_size=128)
        sentinel = object()
        for source_batch, output_batch in itertools.zip_longest(
            source_batches,
            output_batches,
            fillvalue=sentinel,
        ):
            if source_batch is sentinel or output_batch is sentinel:
                raise AssertionError(f"{relative_path}: output batch count changed")
            if source_batch.num_rows != output_batch.num_rows:
                raise AssertionError(f"{relative_path}: output batch size changed")
            row_start = stats["rows"]
            stats["rows"] += source_batch.num_rows
            for column_index in range(source_batch.num_columns):
                if column_index in {compression_index, target_index}:
                    continue
                if not source_batch.column(column_index).equals(
                    output_batch.column(column_index)
                ):
                    raise AssertionError(
                        f"{relative_path}: column {schema.names[column_index]} changed"
                    )

            prompts = source_batch.column(prompt_index).to_pylist()
            old_compression = source_batch.column(compression_index).to_pylist()
            old_targets = source_batch.column(target_index).to_pylist()
            subdatasets = source_batch.column(subdataset_index).to_pylist()
            new_compression = output_batch.column(compression_index).to_pylist()
            new_targets = output_batch.column(target_index).to_pylist()
            for index, subdataset in enumerate(subdatasets):
                if subdataset not in REASONING_SUBDATASETS:
                    if new_compression[index] != old_compression[index]:
                        raise AssertionError(
                            f"{relative_path}: non-reasoning compression prompt changed"
                        )
                    if new_targets[index] != old_targets[index]:
                        raise AssertionError(
                            f"{relative_path}: non-reasoning target changed"
                        )
                    continue
                stats["selected_rows"] += 1
                if new_compression[index] != prompts[index]:
                    raise AssertionError(
                        f"{relative_path}: reasoning compression_prompt != prompt"
                    )
                compress_assigned = self._compress_row(relative_path, row_start + index)
                if not compress_assigned:
                    stats["assigned_uncompressed_rows"] += 1
                    split = None
                else:
                    stats["assigned_compressed_rows"] += 1
                    split = self._split(old_targets[index])
                expected = old_targets[index] if split is None else wrap_reasoning_split(split)
                if new_targets[index] != expected:
                    raise AssertionError(f"{relative_path}: rewritten target mismatch")
                if compress_assigned:
                    if split is None:
                        stats["short_uncompressed_rows"] += 1
                    else:
                        stats["changed_rows"] += 1
                        stats["methods"][split.method] = (
                            stats["methods"].get(split.method, 0) + 1
                        )
        stats["validated"] = True
        return stats


@app.function(volumes={str(VOLUME_ROOT): volume}, timeout=60 * 10)
def list_shards(source_dir: str) -> list[str]:
    source = VOLUME_ROOT / source_dir
    return sorted(str(path.relative_to(source)) for path in source.rglob("*.parquet"))


@app.function(volumes={str(VOLUME_ROOT): volume}, timeout=60 * 10)
def write_report(report_dir: str, name: str, report: dict[str, Any]) -> str:
    path = VOLUME_ROOT / report_dir / name
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)
    volume.commit()
    return str(path)


@app.local_entrypoint()
def main(
    mode: str = "audit",
    source_dir: str = DEFAULT_SOURCE_DIR,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    report_dir: str = DEFAULT_REPORT_DIR,
    max_shards: int = 0,
    overwrite: bool = False,
) -> None:
    if mode not in {"audit", "rewrite", "validate"}:
        raise ValueError("mode must be 'audit', 'rewrite', or 'validate'")
    shards = list_shards.remote(source_dir)
    if max_shards > 0:
        shards = shards[:max_shards]
    worker = ReasoningRewriter()
    if mode == "audit":
        parts = list(worker.audit_shard.map(shards, kwargs={"source_dir": source_dir}))
    elif mode == "rewrite":
        parts = list(
            worker.rewrite_shard.map(
                shards,
                kwargs={
                    "source_dir": source_dir,
                    "output_dir": output_dir,
                    "overwrite": overwrite,
                },
            )
        )
    else:
        parts = list(
            worker.validate_shard.map(
                shards,
                kwargs={"source_dir": source_dir, "output_dir": output_dir},
            )
        )
    report = _merge_stats(parts)
    report.update(
        {
            "mode": mode,
            "source_dir": source_dir,
            "output_dir": output_dir if mode in {"rewrite", "validate"} else None,
            "fallback_answer_tokens": 128,
            "reasoning_compression_fraction": REASONING_COMPRESSION_FRACTION,
            "decoder_tokenizer": "Qwen/Qwen3-4B-Instruct-2507",
        }
    )
    name = f"{mode}-report.json"
    report_path = write_report.remote(report_dir, name, report)
    print(json.dumps({key: value for key, value in report.items() if key != "examples"}, indent=2))
    print(f"report: {report_path}")

