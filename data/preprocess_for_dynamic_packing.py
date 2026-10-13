#!/usr/bin/env python3
"""
Preprocess dataset for DYNAMIC packed training.

Unlike regular packing (preprocess_for_packing.py), this stores:
- Raw memory_strings (NOT embed-tokenized)
- Memory positions for each region
- Single <|memory|> placeholder per region

At training time, DynamicPackedDataset will:
- Tokenize memory_strings with runtime embed_tokenizer
- Expand to N <|memory|> tokens based on runtime chunk_size

This allows changing chunk_size and embed_tokenizer without re-preprocessing.

Usage:
    python data/preprocess_for_dynamic_packing.py \
        --input_dir data/merged/mid_train_merged \
        --output_dir data/dynamic_packed/all_stages \
        --llm_tokenizer Qwen/Qwen3-4B-Instruct-2507 \
        --reference_chunk_size 16 \
        --max_packed_length 32768 \
        --num_workers 64
"""

import argparse
import bisect
import heapq
import os
import pickle
import json
import random
import re
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

import pyarrow as pa
import pyarrow.parquet as pq
from transformers import AutoTokenizer
from tqdm import tqdm

# Support both ``python -m data.preprocess_for_dynamic_packing`` and the
# documented direct script invocation from the repository root.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.chat_utils import build_prompt_and_target_text


# ==================== Stats Tracking (from preprocess_for_packing.py) ====================

class RunningStats:
    """Track running statistics without storing the full history."""

    def __init__(self):
        self.count = 0
        self.total = 0.0
        self.minimum: Optional[float] = None
        self.maximum: Optional[float] = None
        self._lower: List[float] = []
        self._upper: List[float] = []

    def update(self, value: float) -> None:
        self.count += 1
        self.total += value
        self.minimum = value if self.minimum is None else min(self.minimum, value)
        self.maximum = value if self.maximum is None else max(self.maximum, value)

        if not self._lower or value <= -self._lower[0]:
            heapq.heappush(self._lower, -value)
        else:
            heapq.heappush(self._upper, value)

        if len(self._lower) > len(self._upper) + 1:
            heapq.heappush(self._upper, -heapq.heappop(self._lower))
        elif len(self._upper) > len(self._lower):
            heapq.heappush(self._lower, -heapq.heappop(self._upper))

    def average(self) -> float:
        return self.total / self.count if self.count else 0.0

    def median(self) -> float:
        if not self.count:
            return 0.0
        return float(-self._lower[0])

    def min_value(self) -> float:
        return float(self.minimum) if self.minimum is not None else 0.0

    def max_value(self) -> float:
        return float(self.maximum) if self.maximum is not None else 0.0


# ==================== Streaming Packer (from preprocess_for_packing.py) ====================

class StreamingPacker:
    """Streaming best-fit packer with limited active batches."""

    def __init__(
        self,
        max_packed_length: int,
        shuffle_buffer_size: int,
        random_seed: int,
        max_active_batches: int,
        max_memory_tokens_per_batch: Optional[int] = None,
        isolate_memory_threshold: Optional[int] = None,
    ):
        self.max_packed_length = max_packed_length
        self.max_memory_tokens_per_batch = max_memory_tokens_per_batch
        self.isolate_memory_threshold = isolate_memory_threshold
        self.shuffle_buffer_size = max(1, shuffle_buffer_size)
        self.rng = random.Random(random_seed)
        self.buffer: List[Dict[str, Any]] = []
        self.max_active_batches = max(1, max_active_batches)

        self._next_batch_id = 0
        self._active_batches: Dict[int, List[Dict[str, Any]]] = {}
        self._active_lengths: Dict[int, int] = {}
        self._active_memory_tokens: Dict[int, int] = {}  # Track memory tokens per batch
        self._active_bins: List[Tuple[int, int]] = []

        self.num_processed = 0
        self.num_batches = 0
        self.num_isolated = 0  # Samples isolated due to high memory
        self.total_tokens = 0
        self.total_memory_tokens = 0
        self.examples_packed = 0
        self.skipped = 0
        self.skipped_memory_limit = 0

        self.length_stats = RunningStats()
        self.sample_stats = RunningStats()
        self.memory_stats = RunningStats()

    def add_example(self, example: Dict[str, Any]) -> Iterator[List[Dict[str, Any]]]:
        self.num_processed += 1
        self.buffer.append(example)
        if len(self.buffer) >= self.shuffle_buffer_size:
            yield from self._flush_buffer()

    def finalize(self) -> Iterator[List[Dict[str, Any]]]:
        yield from self._flush_buffer()
        remaining_ids = list(self._active_batches.keys())
        for batch_id in remaining_ids:
            batch = self._finalize_batch(batch_id)
            if batch:
                yield batch

    def packing_efficiency(self) -> float:
        if self.num_batches == 0 or self.max_packed_length == 0:
            return 0.0
        return self.total_tokens / (self.num_batches * self.max_packed_length)

    def _flush_buffer(self) -> Iterator[List[Dict[str, Any]]]:
        if not self.buffer:
            return
        self.rng.shuffle(self.buffer)
        buffer = sorted(self.buffer, key=lambda ex: ex['estimated_seq_len'], reverse=True)
        self.buffer = []
        for example in buffer:
            batch = self._place_example(example)
            if batch:
                yield batch

    def _place_example(self, example: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
        seq_len = example['estimated_seq_len']
        mem_tokens = example.get('memory_tokens', 0)

        if seq_len > self.max_packed_length:
            self.skipped += 1
            sub_dataset = example.get('_sub_dataset', 'unknown')
            tqdm.write(f"Skipped: estimated_seq_len={seq_len} > max_packed_length={self.max_packed_length}, sub_dataset={sub_dataset}")
            return None

        # Check if single sample exceeds memory limit
        if self.max_memory_tokens_per_batch and mem_tokens > self.max_memory_tokens_per_batch:
            self.skipped_memory_limit += 1
            sub_dataset = example.get('_sub_dataset', 'unknown')
            tqdm.write(f"Skipped: memory_tokens={mem_tokens} > max_memory_tokens_per_batch={self.max_memory_tokens_per_batch}, sub_dataset={sub_dataset}")
            return None

        # Isolate high-memory samples into their own batch (don't pack with others)
        if self.isolate_memory_threshold and mem_tokens >= self.isolate_memory_threshold:
            self.num_isolated += 1
            self.num_batches += 1
            self.examples_packed += 1
            self.total_tokens += seq_len
            self.total_memory_tokens += mem_tokens
            self.length_stats.update(seq_len)
            self.sample_stats.update(1)
            self.memory_stats.update(mem_tokens)
            return [example]  # Single-sample batch

        # Try to find a batch that fits both seq_len and memory constraints
        idx = bisect.bisect_left(self._active_bins, (seq_len, -1))
        while idx < len(self._active_bins):
            remaining, batch_id = self._active_bins[idx]
            # Check memory constraint
            if self.max_memory_tokens_per_batch:
                current_mem = self._active_memory_tokens.get(batch_id, 0)
                if current_mem + mem_tokens > self.max_memory_tokens_per_batch:
                    idx += 1
                    continue
            # Found a valid batch
            self._active_bins.pop(idx)
            batch = self._active_batches[batch_id]
            batch.append(example)
            self._active_lengths[batch_id] += seq_len
            self._active_memory_tokens[batch_id] = self._active_memory_tokens.get(batch_id, 0) + mem_tokens
            remaining -= seq_len
            if remaining > 0:
                bisect.insort(self._active_bins, (remaining, batch_id))
            else:
                completed = self._finalize_batch(batch_id)
                if completed:
                    return completed
            return None

        # No existing batch fits, create a new one
        batch_id = self._next_batch_id
        self._next_batch_id += 1
        self._active_batches[batch_id] = [example]
        self._active_lengths[batch_id] = seq_len
        self._active_memory_tokens[batch_id] = mem_tokens
        remaining = self.max_packed_length - seq_len
        if remaining > 0:
            bisect.insort(self._active_bins, (remaining, batch_id))
        else:
            return self._finalize_batch(batch_id)

        if len(self._active_bins) > self.max_active_batches:
            remaining_space, full_batch_id = self._active_bins.pop(0)
            return self._finalize_batch(full_batch_id)

        return None

    def _finalize_batch(self, batch_id: int) -> Optional[List[Dict[str, Any]]]:
        batch = self._active_batches.pop(batch_id, None)
        if batch is None:
            return None

        batch_len = self._active_lengths.pop(batch_id, 0)
        batch_mem = self._active_memory_tokens.pop(batch_id, 0)
        self._remove_bin(batch_id)

        self.num_batches += 1
        num_examples = len(batch)
        self.examples_packed += num_examples
        self.total_tokens += batch_len
        self.total_memory_tokens += batch_mem
        self.length_stats.update(batch_len)
        self.sample_stats.update(num_examples)
        self.memory_stats.update(batch_mem)
        return batch

    def _remove_bin(self, batch_id: int) -> None:
        for idx, (_, bid) in enumerate(self._active_bins):
            if bid == batch_id:
                self._active_bins.pop(idx)
                break


# ==================== Parquet Writer (from preprocess_for_packing.py) ====================

class ParquetBatchWriter:
    """Streaming Parquet writer that buffers and emits fixed-size shards."""

    def __init__(
        self,
        output_dir: str,
        base_name: str = "packed_batches",
        shard_size: int = 512,
        compression: str = "snappy",
    ):
        self.output_dir = output_dir
        self.base_name = base_name
        self.shard_size = max(1, shard_size)
        self.compression = compression

        self._buffer: List[bytes] = []
        self._writer: Optional[pq.ParquetWriter] = None
        self._rows_in_current_file = 0
        self._file_index = 0
        self._files: List[str] = []

    @property
    def files(self) -> List[str]:
        return list(self._files)

    def write(self, payload: bytes) -> None:
        self._buffer.append(payload)
        if len(self._buffer) >= self.shard_size:
            self._flush(force=False)

    def close(self) -> None:
        self._flush(force=True)
        self._close_current_writer()

    def _flush(self, force: bool) -> None:
        if not self._buffer:
            return

        while self._buffer and (force or len(self._buffer) >= self.shard_size):
            remaining = self.shard_size - self._rows_in_current_file
            if self._writer is None or remaining == 0:
                self._close_current_writer()
                self._open_new_writer()
                remaining = self.shard_size
            chunk_size = min(len(self._buffer), remaining)
            chunk = self._buffer[:chunk_size]
            array = pa.array(chunk, type=pa.binary())
            table = pa.table({'packed_batch_bytes': array})
            if self._writer is None:
                self._open_new_writer(table.schema)
            self._writer.write_table(table)
            self._rows_in_current_file += chunk_size
            del self._buffer[:chunk_size]

        if force and self._buffer:
            remaining = self.shard_size - self._rows_in_current_file
            if self._writer is None or remaining == 0:
                self._close_current_writer()
                self._open_new_writer()
            chunk = self._buffer
            array = pa.array(chunk, type=pa.binary())
            table = pa.table({'packed_batch_bytes': array})
            if self._writer is None:
                self._open_new_writer(table.schema)
            self._writer.write_table(table)
            self._rows_in_current_file += len(chunk)
            self._buffer.clear()

    def _build_file_path(self, index: int) -> str:
        return os.path.join(self.output_dir, f"{self.base_name}.{index:05d}.parquet")

    def _open_new_writer(self, schema: Optional[pa.Schema] = None) -> None:
        path = self._build_file_path(self._file_index)
        if schema is None:
            schema = pa.schema([('packed_batch_bytes', pa.binary())])
        self._writer = pq.ParquetWriter(path, schema, compression=self.compression)
        self._files.append(path)
        self._rows_in_current_file = 0
        self._file_index += 1

    def _close_current_writer(self) -> None:
        if self._writer is not None:
            self._writer.close()
            self._writer = None
        self._rows_in_current_file = 0


def rename_parquet_files(
    files: List[str],
    prefix: str,
    digits: int,
    output_dir: str,
) -> List[str]:
    """Rename shard files to prefix-xxxxx-of-yyyyy.parquet."""
    total = len(files)
    if total == 0:
        return []

    base_dir = Path(output_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    digits = max(1, digits)
    prefix = prefix or "packed"

    renamed: List[str] = []
    for idx, old_path in enumerate(files, start=1):
        old_path_obj = Path(old_path)
        new_name = f"{prefix}-{idx:0{digits}d}-of-{total:0{digits}d}.parquet"
        new_path = base_dir / new_name
        os.replace(old_path_obj, new_path)
        renamed.append(new_name)
    return renamed


# ==================== Dynamic Packing Specific ====================

# Regex pattern for memory regions (same as LCLMProcessor)
MEMORY_PATTERN = re.compile(
    r'<\|memory_start\|>(.*?)<\|memory_end\|>',
    re.DOTALL
)
MEMORY_MARKERS = ('<|memory_start|>', '<|memory_end|>', '<|memory|>')


def contains_memory_marker(value: Any) -> bool:
    """Return whether any nested text field contains an LCLM memory marker."""
    if isinstance(value, str):
        return any(marker in value for marker in MEMORY_MARKERS)
    if isinstance(value, dict):
        return any(contains_memory_marker(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(contains_memory_marker(item) for item in value)
    return False


def has_empty_memory_block(text: str) -> bool:
    """Check if text has any empty memory blocks (empty or whitespace only)."""
    for match in MEMORY_PATTERN.finditer(text):
        if not match.group(1).strip():
            return True
    return False


def has_unbalanced_memory_tags(text: str) -> bool:
    """Check if memory start/end tags are unbalanced."""
    start_count = text.count('<|memory_start|>')
    end_count = text.count('<|memory_end|>')
    return start_count != end_count


def extract_memory_strings(text: str) -> Tuple[str, List[str], List[Tuple[int, int]]]:
    """
    Extract code from memory regions and replace with single placeholder.

    Input:  "hello <|memory_start|>def foo()<|memory_end|> world"
    Output: ("hello <|memory_start|><|memory|><|memory_end|> world", ["def foo()"], [(6, 44)])

    Returns:
        - modified_text: Text with memory content replaced by placeholder
        - memory_strings: List of extracted memory content strings
        - original_spans: List of (start, end) char positions of each region in ORIGINAL text
    """
    memory_strings = []
    original_spans = []

    # Find all matches first to get positions
    for match in MEMORY_PATTERN.finditer(text):
        memory_strings.append(match.group(1))
        original_spans.append((match.start(), match.end()))

    # Replace all matches
    modified_text = MEMORY_PATTERN.sub('<|memory_start|><|memory|><|memory_end|>', text)

    return modified_text, memory_strings, original_spans


def find_memory_positions(
    input_ids: List[int],
    memory_start_id: int,
    memory_id: Optional[int] = None,
    memory_end_id: Optional[int] = None,
) -> List[int]:
    """
    Find start index for each memory region in tokenized sequence.

    Since base format is always START, M, END (3 consecutive tokens),
    we only need to track the START position.

    Returns list of start indices.
    """
    positions = []
    for i, tok_id in enumerate(input_ids):
        if tok_id == memory_start_id:
            if memory_id is not None and memory_end_id is not None:
                if i + 2 >= len(input_ids) or input_ids[i + 1 : i + 3] != [memory_id, memory_end_id]:
                    raise ValueError(
                        f"Memory START at token {i} is not followed by the compact "
                        "<|memory|>, <|memory_end|> representation"
                    )
            positions.append(i)
    return positions


def _canonicalize_agent_messages(example: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
    """Select one native trajectory field without truthiness/array ambiguity."""
    messages = example.get("messages")
    conversations = example.get("conversations")
    if isinstance(messages, str):
        messages = json.loads(messages)
    if isinstance(conversations, str):
        conversations = json.loads(conversations)
    if messages is not None and conversations is not None:
        if messages != conversations:
            raise ValueError("Agent row has conflicting messages and conversations fields")
        conversations = None
    selected = messages if messages is not None else conversations
    if selected is None:
        return None
    if not isinstance(selected, list) or not selected:
        raise ValueError("Agent trajectory must be a nonempty list")
    selected = [json.loads(message) if isinstance(message, str) else message for message in selected]
    if not all(isinstance(message, dict) for message in selected):
        raise ValueError("Agent trajectory messages must be dictionaries")
    return selected


def _compact_agent_memory_regions(
    messages: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Extract memory bodies from native messages without dropping tool fields.

    Memory markers are valid only in a message's text ``content``. Structured
    fields such as ``tool_calls`` must remain literal and marker-free. The
    returned messages retain every native field while replacing each memory
    body with the single dynamic-packing placeholder.
    """

    compacted: List[Dict[str, Any]] = []
    memory_strings: List[str] = []
    for index, message in enumerate(messages):
        copied = dict(message)
        content = copied.get("content")
        if content is not None:
            if not isinstance(content, str):
                raise TypeError(
                    f"Agent message {index} content must be a string or null"
                )
            modified_content, memories, _ = extract_memory_strings(content)
            copied["content"] = modified_content
            memory_strings.extend(memories)

        structured_fields = {
            key: value for key, value in copied.items() if key != "content"
        }
        if contains_memory_marker(structured_fields):
            raise ValueError(
                "Memory markers are allowed only in agent message content"
            )
        compacted.append(copied)
    return compacted, memory_strings


def estimate_expanded_length(
    base_len: int,
    memory_strings: List[str],
    reference_chunk_size: int,
    embed_tokenizer=None,
    chars_per_token: float = 4.0,
) -> int:
    """
    Estimate final sequence length after expansion with reference_chunk_size.

    Base has 3 tokens per region (START, M, END).
    After expansion: START, M*num_chunks, END = num_chunks + 2.
    Delta per region: (num_chunks + 2) - 3 = num_chunks - 1.

    If embed_tokenizer is provided, uses actual tokenization for accurate count.
    Otherwise falls back to chars_per_token estimation.
    """
    total_delta = 0
    for memory_str in memory_strings:
        if embed_tokenizer is not None:
            # Use actual tokenizer for accurate count
            embed_tokens = len(embed_tokenizer.encode(memory_str, add_special_tokens=False))
            est_embed_tokens = max(1, embed_tokens)
        else:
            # Fallback to char-based estimation
            est_embed_tokens = max(1, len(memory_str) / chars_per_token)
        num_chunks = max(1, math.ceil(est_embed_tokens / reference_chunk_size))
        delta = num_chunks - 1
        total_delta += delta
    return base_len + total_delta


def process_sft_example(
    example: Dict[str, Any],
    tokenizer,
    prompt_column: str = "compression_prompt",
) -> Optional[Tuple[List[int], List[int], List[str], int]]:
    """
    Process SFT (chat format) example using token-based approach.

    Returns:
        Tuple of (input_ids, labels, memory_strings, prompt_len) or None if invalid
    """
    prompt_data = example.get(prompt_column)
    target_data = example.get("target", "")

    if not isinstance(prompt_data, list) or not prompt_data:
        return None

    # 1. Extract memory from prompt messages without dropping structured fields
    # such as tool_calls. Memory regions may also appear in the target for latent
    # CoT training; their metadata is appended after prompt memory in serialized
    # sequence order.
    memory_strings = []
    prompt_messages = []
    for msg in prompt_data:
        if not isinstance(msg, dict) or "role" not in msg:
            raise TypeError("Each prompt message must be a dict containing 'role'")
        copied_message = dict(msg)
        content = copied_message.get("content")
        if content is not None:
            if not isinstance(content, str):
                raise TypeError("Only text message content is supported")
            modified_content, memories, _ = extract_memory_strings(content)
            copied_message["content"] = modified_content
            memory_strings.extend(memories)
        prompt_messages.append(copied_message)

    target_message = example.get("target_message")
    if target_message is not None:
        if not isinstance(target_message, dict) or target_message.get("role") != "assistant":
            raise TypeError("target_message must be an assistant message dict")
        target_message = dict(target_message)
    else:
        if not isinstance(target_data, str):
            raise TypeError("target must be a string when target_message is absent")
        target_message = {"role": "assistant", "content": target_data}

    target_content = target_message.get("content")
    if target_content is not None:
        if not isinstance(target_content, str):
            raise TypeError("Only text target content is supported")
        modified_target, target_memories, _ = extract_memory_strings(target_content)
        target_message["content"] = modified_target
        memory_strings.extend(target_memories)

    tools = example.get("tools")
    template_kwargs = {"tools": tools} if tools else {}

    # 2. Tokenize prompt with add_generation_prompt=True
    prompt_ids = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=True,
        add_special_tokens=False,
        add_generation_prompt=True,
        **template_kwargs,
    )

    # 3. Tokenize full conversation (prompt + assistant response)
    full_messages = prompt_messages + [target_message]
    full_ids = tokenizer.apply_chat_template(
        full_messages,
        tokenize=True,
        add_special_tokens=False,
        add_generation_prompt=False,
        **template_kwargs,
    )

    # 4. Compute labels: prompt = -100, target = trainable
    prompt_len = len(prompt_ids)
    prefix_matches = full_ids[:prompt_len] == prompt_ids
    if example.get('_recover_prefix_only') and prefix_matches:
        return None
    if not prefix_matches:
        if example.get('_legacy_sft_prefix_strict'):
            # The baseline/recovery release migration keeps these rows in its
            # separate recovery shard group, including on baseline retries.
            return None
        if not getattr(tokenizer,'is_fast',False):
            raise ValueError("Generation-ready prompt tokens are not a prefix of the full chat")
        prompt_text=tokenizer.apply_chat_template(prompt_messages,tokenize=False,
            add_generation_prompt=True,**template_kwargs)
        full_text=tokenizer.apply_chat_template(full_messages,tokenize=False,
            add_generation_prompt=False,**template_kwargs)
        if not full_text.startswith(prompt_text):
            raise ValueError('Generation-ready prompt text is not a prefix of the full chat')
        encoded=tokenizer(full_text,add_special_tokens=False,return_offsets_mapping=True)
        if list(encoded['input_ids'])!=full_ids:
            raise ValueError('Full chat tokenization disagrees with offset tokenization')
        boundary=len(prompt_text)
        prompt_len=next((i for i,(start,end) in enumerate(encoded['offset_mapping'])
                         if start>=boundary and end>start),len(full_ids))
    labels = [-100] * prompt_len + full_ids[prompt_len:]

    # 5. Find and mask memory regions (START, M, END)
    memory_start_id = tokenizer.convert_tokens_to_ids('<|memory_start|>')
    memory_positions = find_memory_positions(
        full_ids,
        memory_start_id,
        tokenizer.convert_tokens_to_ids('<|memory|>'),
        tokenizer.convert_tokens_to_ids('<|memory_end|>'),
    )
    for start_idx in memory_positions:
        for i in range(start_idx, min(start_idx + 3, len(labels))):
            labels[i] = -100

    return full_ids, labels, memory_strings, prompt_len


def process_continual_example(
    example: Dict[str, Any],
    tokenizer,
    prompt_column: str = "compression_prompt",
) -> Optional[Tuple[List[int], List[int], List[str]]]:
    """
    Process continual pretraining example.

    Returns:
        Tuple of (input_ids, labels, memory_strings) or None if invalid
    """
    prompt_data = example.get(prompt_column)

    # Extract content from user message
    if isinstance(prompt_data, list) and len(prompt_data) > 0:
        full_text = prompt_data[0].get("content", "")
    else:
        full_text = str(prompt_data) if prompt_data else ""

    if not full_text or not full_text.strip():
        return None

    # 1. Extract memory strings
    text_with_placeholders, memory_strings, _ = extract_memory_strings(full_text)

    # 2. Tokenize
    input_ids = tokenizer.encode(text_with_placeholders, add_special_tokens=False)
    if not input_ids:
        return None

    # 3. Labels: all trainable except memory regions
    labels = list(input_ids)

    # 4. Mask memory regions (START, M, END)
    memory_start_id = tokenizer.convert_tokens_to_ids('<|memory_start|>')
    memory_positions = find_memory_positions(input_ids, memory_start_id)
    for start_idx in memory_positions:
        for i in range(start_idx, min(start_idx + 3, len(labels))):
            labels[i] = -100

    return input_ids, labels, memory_strings


# Global worker state
_worker_tokenizer = None
_worker_embed_tokenizer = None
_worker_memory_start_id = None
_worker_memory_end_id = None
_worker_memory_id = None


def worker_init(tokenizer_name: str, embed_tokenizer_name: str = None,
                tokenizer_revision: str = None, embed_tokenizer_revision: str = None):
    """Initialize tokenizers in worker process."""
    global _worker_tokenizer, _worker_embed_tokenizer, _worker_memory_start_id, _worker_memory_end_id, _worker_memory_id

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    decoder_options = {'revision': tokenizer_revision} if tokenizer_revision else {}
    _worker_tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True, **decoder_options)
    if _worker_tokenizer.pad_token is None:
        _worker_tokenizer.pad_token = _worker_tokenizer.eos_token

    # Add special memory tokens
    new_tokens = ['<|memory_start|>', '<|memory_end|>', '<|memory|>']
    _worker_tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})

    _worker_memory_start_id = _worker_tokenizer.convert_tokens_to_ids('<|memory_start|>')
    _worker_memory_end_id = _worker_tokenizer.convert_tokens_to_ids('<|memory_end|>')
    _worker_memory_id = _worker_tokenizer.convert_tokens_to_ids('<|memory|>')

    # Load embed tokenizer for accurate code length estimation
    if embed_tokenizer_name:
        encoder_options = {'revision': embed_tokenizer_revision} if embed_tokenizer_revision else {}
        _worker_embed_tokenizer = AutoTokenizer.from_pretrained(embed_tokenizer_name, use_fast=True, **encoder_options)
    else:
        _worker_embed_tokenizer = None


def worker_process_example(args: Tuple[Dict[str, Any], int, str, Optional[int], Optional[int]]) -> Optional[Dict[str, Any]]:
    """
    Process single example for dynamic packing.

    Returns dict with:
    - base_input_ids: tokenized sequence with single <|memory|> per region
    - base_labels: labels (-100 for memory regions and prompt)
    - memory_strings: raw code strings for each region
    - memory_positions: start indices for each region
    - estimated_seq_len: estimated length after expansion
    """
    global _worker_tokenizer, _worker_embed_tokenizer, _worker_memory_start_id, _worker_memory_end_id, _worker_memory_id

    example, reference_chunk_size, prompt_column, max_seq_len, max_memory_tokens = args
    # Parquet transports arbitrary per-row tool schemas as JSON, avoiding Arrow
    # unions/casts that can silently erase parameters from heterogeneous tools.
    example = dict(example)
    sub_dataset = example.get('sub_dataset', 'unknown')

    try:
        if isinstance(example.get('tools'), str):
            example['tools'] = json.loads(example['tools'])
        # Native agent trajectories use messages/conversations and intentionally
        # have no prompt/target split. They are handled below without compression.
        agent_messages = _canonicalize_agent_messages(example)

        # Early check: skip empty inputs
        prompt_data = example.get(prompt_column)
        if not agent_messages and (
            prompt_data is None
            or (hasattr(prompt_data, '__len__') and len(prompt_data) == 0)
        ):
            return None

        # Early check: skip rows with empty memory blocks or unbalanced tags
        # Extract all text content from prompt for validation
        validation_messages = agent_messages or prompt_data or []
        if hasattr(validation_messages, '__iter__') and not isinstance(validation_messages, str):
            for turn in validation_messages:
                if isinstance(turn, dict):
                    content_value = turn.get('content')
                    content = content_value if isinstance(content_value, str) else ""
                    if has_unbalanced_memory_tags(content):
                        return None
                    if has_empty_memory_block(content):
                        return None

        target_for_validation = example.get("target_message", {}).get("content") if isinstance(
            example.get("target_message"), dict
        ) else example.get("target", "")
        if isinstance(target_for_validation, str):
            if has_unbalanced_memory_tags(target_for_validation):
                return None
            if has_empty_memory_block(target_for_validation):
                return None

        # Determine if this is continual pretraining or SFT
        target_data = example.get("target", "")
        is_continual = (target_data == "NA_string_only")

        if agent_messages:
            # Imported lazily so the legacy prompt/target path remains usable by
            # older callers. Native fields survive while memory bodies in any
            # message content are compacted for the encoder/adapter path.
            from data.chat_utils import tokenize_qwen_agent_conversation

            if contains_memory_marker(example.get("tools")):
                raise ValueError("Agent tool schemas must not contain memory markers")
            compacted_messages, memory_strings = _compact_agent_memory_regions(
                agent_messages
            )

            trajectory = tokenize_qwen_agent_conversation(
                compacted_messages,
                tokenizer=_worker_tokenizer,
                tools=example.get("tools"),
            )
            input_ids = trajectory["input_ids"]
            labels = trajectory["labels"]
        elif is_continual:
            # Continual pretraining: all tokens trainable except memory regions
            result = process_continual_example(example, _worker_tokenizer, prompt_column)
            if result is None:
                return None
            input_ids, labels, memory_strings = result
        else:
            # SFT: prompt masked, target trainable (except memory regions)
            result = process_sft_example(example, _worker_tokenizer, prompt_column)
            if result is None:
                return None
            input_ids, labels, memory_strings, prompt_len = result

        # Find memory positions for output
        memory_positions = find_memory_positions(
            input_ids,
            _worker_memory_start_id,
            _worker_memory_id,
            _worker_memory_end_id,
        )

        # Validate memory count - skip if mismatch (broken tokenization)
        if len(memory_strings) != len(memory_positions):
            # This means regex found memory blocks but tokenizer didn't preserve them
            tqdm.write(f"Warning: memory_strings ({len(memory_strings)}) != memory_positions ({len(memory_positions)}), sub_dataset={sub_dataset}")
            # Save problematic example for debugging
            try:
                import time
                debug_dir = Path("data/debug_memory_mismatch")
                debug_dir.mkdir(parents=True, exist_ok=True)
                debug_file = debug_dir / f"mismatch_{int(time.time()*1000)}_{sub_dataset[:30]}.json"
                debug_data = {
                    "sub_dataset": sub_dataset,
                    "memory_strings_count": len(memory_strings),
                    "memory_positions_count": len(memory_positions),
                    "memory_strings": memory_strings,
                    "prompt_data": [
                        {"role": m.get("role"), "content": str(m.get("content", ""))}
                        for m in example.get(prompt_column, [])
                        if isinstance(m, dict)
                    ],
                    "target": str(example.get("target", "")),
                }
                with open(debug_file, "w") as f:
                    json.dump(debug_data, f, indent=2, ensure_ascii=False)
            except Exception as e:
                tqdm.write(f"  Failed to save debug: {e}")
            return None

        # Check for trainable tokens
        trainable_count = sum(1 for l in labels if l != -100)
        if trainable_count == 0:
            return None

        # 7. Estimate expanded length (using embed tokenizer if available)
        estimated_seq_len = estimate_expanded_length(
            len(input_ids), memory_strings, reference_chunk_size,
            embed_tokenizer=_worker_embed_tokenizer
        )

        # 8. Check max length
        if max_seq_len and estimated_seq_len > max_seq_len:
            return {'_skipped_max_seq_len': True, 'estimated_seq_len': estimated_seq_len, '_sub_dataset': sub_dataset}

        # 9. Compute memory tokens (encoder input) and check limit
        # Pre-compression: use actual tokenizer if available, else estimate
        if _worker_embed_tokenizer is not None:
            original_code_tokens = sum(
                len(_worker_embed_tokenizer.encode(s, add_special_tokens=False))
                for s in memory_strings
            )
        else:
            original_code_tokens = sum(len(s) / 4.0 for s in memory_strings)

        # Check max memory tokens (encoder input limit)
        if max_memory_tokens and original_code_tokens > max_memory_tokens:
            return {'_skipped_max_memory_tokens': True, 'memory_tokens': int(original_code_tokens), '_sub_dataset': sub_dataset}

        pre_compression_tokens = len(input_ids) - len(memory_positions) * 3 + int(original_code_tokens)

        return {
            # Core data needed for training
            'base_input_ids': input_ids,
            'base_labels': labels,
            'memory_strings': memory_strings,
            'memory_positions': memory_positions,
            'estimated_seq_len': estimated_seq_len,
            'memory_tokens': int(original_code_tokens),  # For per-batch memory limit
            # Metadata for stats tracking (not stored in final output)
            '_sub_dataset': sub_dataset,
            '_trainable_tokens': trainable_count,
            '_pre_compression_tokens': int(pre_compression_tokens),
        }

    except Exception as e:
        tqdm.write(f"Error processing example: {type(e).__name__}: {e}, sub_dataset={sub_dataset}")
        return None


def iterate_processed_examples(
    dataset: Iterable[Dict[str, Any]],
    num_workers: int,
    llm_tokenizer_name: str,
    embed_tokenizer_name: Optional[str],
    reference_chunk_size: int,
    prompt_column: str,
    max_seq_len: Optional[int],
    max_memory_tokens: Optional[int],
    total: Optional[int],
) -> Iterator[Dict[str, Any]]:
    """Yield processed examples using multiprocessing."""
    desc = "Processing examples"

    # Prepare args for each example
    def make_args(example):
        return (dict(example), reference_chunk_size, prompt_column, max_seq_len, max_memory_tokens)

    if num_workers > 1:
        from multiprocessing import Pool

        optimal_chunksize = max(16, min(128, (total or 10000) // (num_workers * 4)))

        with Pool(
            processes=num_workers,
            initializer=worker_init,
            initargs=(llm_tokenizer_name, embed_tokenizer_name),
            maxtasksperchild=1000,
        ) as pool:
            args_iter = (make_args(ex) for ex in dataset)
            iterator = pool.imap_unordered(worker_process_example, args_iter, chunksize=optimal_chunksize)
            for result in tqdm(iterator, total=total, desc=desc):
                if result is not None:
                    yield result
    else:
        # Single-threaded
        worker_init(llm_tokenizer_name, embed_tokenizer_name)
        for example in tqdm(dataset, total=total, desc=desc):
            result = worker_process_example(make_args(example))
            if result is not None:
                yield result


def main():
    parser = argparse.ArgumentParser(description="Preprocess data for dynamic packing")
    parser.add_argument("--input_dir", type=str, default=None, help="Parquet/JSONL file or directory")
    parser.add_argument("--input_path", type=str, default=None, help="Alias for --input_dir")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--llm_tokenizer", type=str, required=True, help="LLM tokenizer name")
    parser.add_argument("--embed_tokenizer", type=str, default="Qwen/Qwen3-Embedding-0.6B", help="Embed tokenizer for accurate code length estimation")
    parser.add_argument("--reference_chunk_size", type=int, default=16, help="Reference chunk size for packing estimation")
    parser.add_argument("--max_packed_length", type=int, default=32768, help="Maximum packed sequence length")
    parser.add_argument("--max_seq_len", type=int, default=None, help="Maximum single sequence length")
    parser.add_argument("--max_memory_tokens", type=int, default=None, help="Maximum memory tokens (encoder input) per sample")
    parser.add_argument("--max_memory_tokens_per_batch", type=int, default=None, help="Maximum memory tokens per packed batch")
    parser.add_argument("--isolate_memory_threshold", type=int, default=None, help="Isolate samples with memory_tokens >= threshold into single-sample batches")
    parser.add_argument("--prompt_column", type=str, default="compression_prompt", help="Prompt column name")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of workers")
    parser.add_argument("--packing_seed", type=int, default=42, help="Random seed")
    parser.add_argument("--shuffle_buffer_size", type=int, default=8192, help="Shuffle buffer size")
    parser.add_argument("--shard_size", type=int, default=512, help="Packed batches per shard")
    parser.add_argument("--max_active_batches", type=int, default=32, help="Max active batches for packing")
    parser.add_argument("--parquet_file_prefix", type=str, default="train", help="Parquet file prefix")
    parser.add_argument("--parquet_filename_digits", type=int, default=5, help="Digits for shard numbering")

    # Legacy alias
    parser.add_argument("--seed", type=int, default=None, help="Alias for --packing_seed")

    args = parser.parse_args()

    # Keep helpers and --help usable in lightweight environments. Actual CLI
    # execution still requires the project's declared `datasets` dependency.
    try:
        from datasets import load_dataset
    except ImportError as exc:
        parser.error("the preprocessing CLI requires the 'datasets' package")

    # Handle aliases
    if args.input_path and not args.input_dir:
        args.input_dir = args.input_path
    if args.seed is not None:
        args.packing_seed = args.seed

    if not args.input_dir:
        parser.error("--input_dir is required")

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 80)
    print("DYNAMIC PACKING PREPROCESSING")
    print("=" * 80)
    print(f"Input: {args.input_dir}")
    print(f"Output: {args.output_dir}")
    print(f"LLM tokenizer: {args.llm_tokenizer}")
    print(f"Embed tokenizer: {args.embed_tokenizer}")
    print(f"Reference chunk size: {args.reference_chunk_size}")
    print(f"Max packed length: {args.max_packed_length}")
    print(f"Max seq len: {args.max_seq_len}")
    print(f"Max memory tokens (per sample): {args.max_memory_tokens}")
    print(f"Max memory tokens (per batch): {args.max_memory_tokens_per_batch}")
    print(f"Isolate memory threshold: {args.isolate_memory_threshold}")
    print(f"Num workers: {args.num_workers}")
    print("")
    print("NOTE: This creates a dataset that works with ANY embed_tokenizer")
    print("      and ANY chunk_size at training time!")
    print("=" * 80)

    # Load tokenizers (for validation)
    print("\n[1/3] Loading tokenizers...")
    if args.num_workers > 1:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Pre-cache tokenizers
    from huggingface_hub import snapshot_download
    try:
        snapshot_download(args.llm_tokenizer, allow_patterns=["*.json", "*.txt", "*.model"])
        print(f"  LLM tokenizer cached: {args.llm_tokenizer}")
    except Exception as e:
        print(f"  Warning: Could not pre-cache LLM tokenizer: {e}")

    if args.embed_tokenizer:
        try:
            snapshot_download(args.embed_tokenizer, allow_patterns=["*.json", "*.txt", "*.model"])
            print(f"  Embed tokenizer cached: {args.embed_tokenizer}")
        except Exception as e:
            print(f"  Warning: Could not pre-cache embed tokenizer: {e}")

    tokenizer = AutoTokenizer.from_pretrained(args.llm_tokenizer, use_fast=True)
    new_tokens = ['<|memory_start|>', '<|memory_end|>', '<|memory|>']
    tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})
    print(f"  Added special tokens: {new_tokens}")

    memory_id = tokenizer.convert_tokens_to_ids('<|memory|>')
    memory_start_id = tokenizer.convert_tokens_to_ids('<|memory_start|>')
    memory_end_id = tokenizer.convert_tokens_to_ids('<|memory_end|>')
    print(f"  Special token IDs: memory={memory_id}, start={memory_start_id}, end={memory_end_id}")

    # Load dataset
    print(f"\n[2/3] Loading dataset from {args.input_dir}...")
    input_path = args.input_dir
    path_obj = Path(input_path)

    if path_obj.is_dir():
        parquet_files = list(path_obj.glob("*.parquet"))
        if parquet_files:
            dataset = load_dataset('parquet', data_files=str(path_obj / "*.parquet"), split='train')
        else:
            json_files = [
                *path_obj.glob("*.json"),
                *path_obj.glob("*.jsonl"),
                *path_obj.glob("*.json.gz"),
                *path_obj.glob("*.jsonl.gz"),
            ]
            if not json_files:
                raise ValueError(f"No parquet or JSON/JSONL files found in: {input_path}")
            dataset = load_dataset(
                'json', data_files=[str(path) for path in sorted(json_files)], split='train'
            )
    elif input_path.endswith('.parquet'):
        dataset = load_dataset('parquet', data_files=input_path, split='train')
    elif input_path.endswith(('.json', '.jsonl', '.json.gz', '.jsonl.gz')):
        dataset = load_dataset('json', data_files=input_path, split='train')
    else:
        raise ValueError(f"Unsupported format: {input_path}; expected parquet or JSONL")

    # Shuffle
    print(f"  Applying global shuffle (seed={args.packing_seed})...")
    dataset = dataset.shuffle(seed=args.packing_seed)

    dataset_size = len(dataset)
    print(f"  Loaded {dataset_size:,} examples")

    # Process and pack
    print(f"\n[3/3] Processing examples...")
    print(f"  Using {args.num_workers} worker(s)")
    print(f"  Shuffle buffer: {args.shuffle_buffer_size}")
    print(f"  Max active batches: {args.max_active_batches}")

    processed_iterator = iterate_processed_examples(
        dataset=dataset,
        num_workers=args.num_workers,
        llm_tokenizer_name=args.llm_tokenizer,
        embed_tokenizer_name=args.embed_tokenizer,
        reference_chunk_size=args.reference_chunk_size,
        prompt_column=args.prompt_column,
        max_seq_len=args.max_seq_len,
        max_memory_tokens=args.max_memory_tokens,
        total=dataset_size,
    )

    # Stats tracking
    from collections import defaultdict
    import numpy as np

    sub_dataset_stats = defaultdict(lambda: {
        "num_samples": 0,
        "num_skipped_too_long": 0,
        "pre_compression_tokens": 0,
        "post_compression_tokens": 0,
        "trainable_tokens": 0,
    })

    samples_skipped_no_trainable = 0
    samples_skipped_too_short = 0
    samples_skipped_too_long = 0
    samples_skipped_max_memory = 0
    total_pre_compression = 0
    total_post_compression = 0

    # Track individual sample lengths for quantile stats
    all_sample_lengths = []
    skipped_sample_lengths = []

    # Create output directories
    all_samples_dir = os.path.join(args.output_dir, "all_samples")
    max_length_dir = os.path.join(args.output_dir, "max_length_only")
    os.makedirs(all_samples_dir, exist_ok=True)
    os.makedirs(max_length_dir, exist_ok=True)

    # Create single packer
    packer = StreamingPacker(
        max_packed_length=args.max_packed_length,
        shuffle_buffer_size=args.shuffle_buffer_size,
        random_seed=args.packing_seed,
        max_active_batches=args.max_active_batches,
        max_memory_tokens_per_batch=args.max_memory_tokens_per_batch,
        isolate_memory_threshold=args.isolate_memory_threshold,
    )

    # Create TWO writers - we filter output batches by length
    writer_all = ParquetBatchWriter(
        output_dir=all_samples_dir,
        base_name="packed_batches",
        shard_size=args.shard_size,
        compression="snappy",
    )

    writer_max = ParquetBatchWriter(
        output_dir=max_length_dir,
        base_name="packed_batches",
        shard_size=args.shard_size,
        compression="snappy",
    )

    batches_all = 0
    batches_max = 0

    # Track packed batch lengths for distribution
    all_batch_lengths = []
    max_batch_lengths = []

    # Threshold for "max length" batches (batches at or very near max_packed_length)
    max_length_threshold = args.max_packed_length  # Exactly at max

    print("\nStreaming: processing, filtering, and packing...")

    for processed_example in processed_iterator:
        # Check if this was a max_seq_len skip from the worker
        if processed_example.get('_skipped_max_seq_len'):
            samples_skipped_too_long += 1
            sub_dataset = processed_example.get('_sub_dataset', 'unknown')
            sub_dataset_stats[sub_dataset]["num_skipped_too_long"] += 1
            skipped_sample_lengths.append(processed_example.get('estimated_seq_len', 0))
            continue

        # Check if this was a max_memory_tokens skip from the worker
        if processed_example.get('_skipped_max_memory_tokens'):
            samples_skipped_max_memory += 1
            sub_dataset = processed_example.get('_sub_dataset', 'unknown')
            memory_tokens = processed_example.get('memory_tokens', 0)
            tqdm.write(f"Skipped: memory_tokens={memory_tokens:,} > max_memory_tokens, sub_dataset={sub_dataset}")
            continue

        estimated_len = processed_example['estimated_seq_len']
        sub_dataset = processed_example.get('_sub_dataset', 'unknown')
        trainable = processed_example.get('_trainable_tokens', 0)

        # Filter: must have trainable tokens
        if trainable == 0:
            samples_skipped_no_trainable += 1
            continue

        # Filter: minimum length
        if estimated_len < 18:
            samples_skipped_too_short += 1
            continue

        # Track sample length for stats
        all_sample_lengths.append(estimated_len)

        # Track stats
        stats = sub_dataset_stats[sub_dataset]
        stats["num_samples"] += 1
        stats["pre_compression_tokens"] += processed_example.get('_pre_compression_tokens', 0)
        stats["post_compression_tokens"] += len(processed_example['base_input_ids'])
        stats["trainable_tokens"] += trainable

        total_pre_compression += processed_example.get('_pre_compression_tokens', 0)
        total_post_compression += len(processed_example['base_input_ids'])

        # Strip metadata before packing (only keep core training data + sub_dataset for logging)
        example_to_pack = {
            'base_input_ids': processed_example['base_input_ids'],
            'base_labels': processed_example['base_labels'],
            'memory_strings': processed_example['memory_strings'],
            'memory_positions': processed_example['memory_positions'],
            'estimated_seq_len': processed_example['estimated_seq_len'],
            'memory_tokens': processed_example['memory_tokens'],  # For per-batch memory limit
            '_sub_dataset': sub_dataset,  # Keep for skip logging
        }

        # Add to packer
        for packed_batch in packer.add_example(example_to_pack):
            batch_len = sum(ex.get('estimated_seq_len', 0) for ex in packed_batch)
            all_batch_lengths.append(batch_len)
            writer_all.write(pickle.dumps(packed_batch))
            batches_all += 1

            # If batch is at max length, also write to max_length_only
            if batch_len >= max_length_threshold:
                max_batch_lengths.append(batch_len)
                writer_max.write(pickle.dumps(packed_batch))
                batches_max += 1

    # Finalize packer
    print("\nFinalizing packer...")

    for packed_batch in packer.finalize():
        batch_len = sum(ex.get('estimated_seq_len', 0) for ex in packed_batch)
        all_batch_lengths.append(batch_len)
        writer_all.write(pickle.dumps(packed_batch))
        batches_all += 1

        # If batch is at max length, also write to max_length_only
        if batch_len >= max_length_threshold:
            max_batch_lengths.append(batch_len)
            writer_max.write(pickle.dumps(packed_batch))
            batches_max += 1

    writer_all.close()
    writer_max.close()

    # Rename files
    files_all = rename_parquet_files(
        writer_all.files,
        prefix=args.parquet_file_prefix,
        digits=args.parquet_filename_digits,
        output_dir=all_samples_dir,
    )

    files_max = rename_parquet_files(
        writer_max.files,
        prefix=args.parquet_file_prefix,
        digits=args.parquet_filename_digits,
        output_dir=max_length_dir,
    )

    packed_batches_written = batches_all

    print(f"\n  All samples saved to: {args.output_dir}/all_samples/")
    print(f"    Batches: {batches_all}, Files: {len(files_all) if files_all else 0}")
    print(f"  Max-length samples saved to: {args.output_dir}/max_length_only/")
    print(f"    Batches: {batches_max}, Files: {len(files_max) if files_max else 0}")

    # Compute sample length statistics
    sample_length_stats = {}
    length_distribution = {}
    if all_sample_lengths:
        lengths_arr = np.array(all_sample_lengths)
        sample_length_stats = {
            'count': len(all_sample_lengths),
            'min': int(np.min(lengths_arr)),
            'max': int(np.max(lengths_arr)),
            'mean': float(np.mean(lengths_arr)),
            'median': float(np.median(lengths_arr)),
            'std': float(np.std(lengths_arr)),
            'p25': float(np.percentile(lengths_arr, 25)),
            'p75': float(np.percentile(lengths_arr, 75)),
            'p90': float(np.percentile(lengths_arr, 90)),
            'p95': float(np.percentile(lengths_arr, 95)),
            'p99': float(np.percentile(lengths_arr, 99)),
        }

        # Create length distribution histogram (512-token bins)
        bin_size = 512
        max_len = int(np.max(lengths_arr))
        bins = list(range(0, max_len + bin_size + 1, bin_size))
        hist, bin_edges = np.histogram(lengths_arr, bins=bins)

        # Format as dict: "0-512": count, "512-1024": count, etc.
        length_distribution = {}
        for i, count in enumerate(hist):
            if count > 0:
                bin_start = int(bin_edges[i])
                bin_end = int(bin_edges[i + 1])
                length_distribution[f"{bin_start}-{bin_end}"] = int(count)

    skipped_length_stats = {}
    if skipped_sample_lengths:
        skipped_arr = np.array(skipped_sample_lengths)
        skipped_length_stats = {
            'count': len(skipped_sample_lengths),
            'min': int(np.min(skipped_arr)),
            'max': int(np.max(skipped_arr)),
            'mean': float(np.mean(skipped_arr)),
            'median': float(np.median(skipped_arr)),
        }

    # Compute PACKED BATCH length statistics
    def compute_batch_stats(batch_lengths, max_packed_len):
        if not batch_lengths:
            return {}, {}
        arr = np.array(batch_lengths)
        stats = {
            'count': len(batch_lengths),
            'min': int(np.min(arr)),
            'max': int(np.max(arr)),
            'mean': float(np.mean(arr)),
            'median': float(np.median(arr)),
            'std': float(np.std(arr)),
            'p25': float(np.percentile(arr, 25)),
            'p75': float(np.percentile(arr, 75)),
            'p90': float(np.percentile(arr, 90)),
            'p95': float(np.percentile(arr, 95)),
            'p99': float(np.percentile(arr, 99)),
        }
        # Distribution with 512-token bins
        # Use max_packed_len + 1 as final edge so values at exactly max_packed_len
        # fall into the last normal bin, not a separate overflow bin
        bin_size = 512
        bins = list(range(0, max_packed_len + 1, bin_size))
        if bins[-1] != max_packed_len + 1:
            bins.append(max_packed_len + 1)  # Ensure max value is captured
        hist, bin_edges = np.histogram(arr, bins=bins)
        dist = {}
        for i, count in enumerate(hist):
            if count > 0:
                bin_start = int(bin_edges[i])
                bin_end = int(bin_edges[i + 1]) - 1  # Show inclusive end
                dist[f"{bin_start}-{bin_end}"] = int(count)
        return stats, dist

    all_batch_stats, all_batch_distribution = compute_batch_stats(all_batch_lengths, args.max_packed_length)
    max_batch_stats, max_batch_distribution = compute_batch_stats(max_batch_lengths, args.max_packed_length)

    # Save metadata for all_samples
    base_metadata = {
        'dataset_path': args.input_dir,
        'llm_tokenizer': args.llm_tokenizer,
        'embed_tokenizer': args.embed_tokenizer,
        'reference_chunk_size': args.reference_chunk_size,
        'max_packed_length': args.max_packed_length,
        'max_seq_len': args.max_seq_len,
        'packing_seed': args.packing_seed,
        'total_samples_collected': len(all_sample_lengths),
        'samples_skipped_no_trainable': samples_skipped_no_trainable,
        'samples_skipped_too_short': samples_skipped_too_short,
        'samples_skipped_exceed_max_seq_len': samples_skipped_too_long,
        'samples_skipped_exceed_max_memory_tokens': samples_skipped_max_memory,
        'total_pre_compression_tokens': total_pre_compression,
        'total_post_compression_tokens': total_post_compression,
        'sample_length_stats': sample_length_stats,
        'sample_length_distribution': length_distribution,
        'skipped_sample_length_stats': skipped_length_stats,
    }

    # Save to all_samples subfolder
    all_samples_metadata = {
        **base_metadata,
        'num_packed_batches': packer.num_batches,
        'samples_skipped_exceed_max_packed': packer.skipped,
        'packing_efficiency': packer.packing_efficiency(),
        'batch_length_stats': all_batch_stats,
        'batch_length_distribution': all_batch_distribution,
        'parquet_files': files_all,
    }
    metadata_path = os.path.join(args.output_dir, "all_samples", "metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(all_samples_metadata, f, indent=2)
    print(f"  Metadata: {metadata_path}")

    stats_path = os.path.join(args.output_dir, "all_samples", "sub_dataset_stats.json")
    with open(stats_path, 'w') as f:
        json.dump(dict(sub_dataset_stats), f, indent=2)

    # Save to max_length_only subfolder (filtered by packed batch length)
    if batches_max > 0:
        max_length_metadata = {
            **base_metadata,
            'num_packed_batches': batches_max,
            'max_length_threshold': max_length_threshold,
            'batch_length_stats': max_batch_stats,
            'batch_length_distribution': max_batch_distribution,
            'parquet_files': files_max,
            'note': f'Only packed batches with total length >= {max_length_threshold}',
        }
        metadata_path = os.path.join(args.output_dir, "max_length_only", "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(max_length_metadata, f, indent=2)
        print(f"  Metadata: {metadata_path}")

    # Print summary
    print("\n" + "=" * 80)
    print("PACKING STATISTICS")
    print("=" * 80)
    print(f"Total samples collected: {len(all_sample_lengths):,}")
    print(f"Skipped (no trainable): {samples_skipped_no_trainable:,}")
    print(f"Skipped (< 18 tokens): {samples_skipped_too_short:,}")
    print(f"Skipped (> max_seq_len): {samples_skipped_too_long:,}")
    print(f"Skipped (> max_memory_tokens): {samples_skipped_max_memory:,}")

    print(f"\nPacking stats:")
    print(f"  Total packed batches: {packer.num_batches:,}")
    print(f"  Isolated (high memory): {packer.num_isolated:,}")
    print(f"  Skipped (> max_packed): {packer.skipped:,}")
    print(f"  Skipped (> max_memory_per_batch): {packer.skipped_memory_limit:,}")
    print(f"  Packing efficiency: {packer.packing_efficiency():.1%}")
    print(f"  Samples per batch: avg={packer.sample_stats.average():.2f}, min={packer.sample_stats.min_value():.0f}, max={packer.sample_stats.max_value():.0f}")
    print(f"  Length per batch: avg={packer.length_stats.average():.1f}, min={packer.length_stats.min_value():.0f}, max={packer.length_stats.max_value():.0f}")
    if packer.memory_stats.count > 0:
        print(f"  Memory per batch: avg={packer.memory_stats.average():.1f}, min={packer.memory_stats.min_value():.0f}, max={packer.memory_stats.max_value():.0f}")

    print(f"\nMax-length batches (>= {max_length_threshold} tokens):")
    print(f"  Count: {batches_max:,} ({100*batches_max/max(1,packer.num_batches):.1f}% of all batches)")

    print(f"\nToken stats:")
    print(f"  Pre-compression: {total_pre_compression:,}")
    print(f"  Post-compression: {total_post_compression:,}")

    if sample_length_stats:
        print(f"\nIndividual sample length stats (n={sample_length_stats['count']:,}):")
        print(f"  Min: {sample_length_stats['min']:,}")
        print(f"  Max: {sample_length_stats['max']:,}")
        print(f"  Mean: {sample_length_stats['mean']:,.1f}")
        print(f"  Std: {sample_length_stats['std']:,.1f}")
        print(f"  Median (p50): {sample_length_stats['median']:,.1f}")
        print(f"  p25: {sample_length_stats['p25']:,.1f}")
        print(f"  p75: {sample_length_stats['p75']:,.1f}")
        print(f"  p90: {sample_length_stats['p90']:,.1f}")
        print(f"  p95: {sample_length_stats['p95']:,.1f}")
        print(f"  p99: {sample_length_stats['p99']:,.1f}")

    if length_distribution:
        print(f"\nSample length distribution (512-token bins):")
        total_samples = sample_length_stats['count']
        cumulative = 0
        for bin_range, count in sorted(length_distribution.items(), key=lambda x: int(x[0].split('-')[0])):
            pct = 100.0 * count / total_samples
            cumulative += count
            cum_pct = 100.0 * cumulative / total_samples
            bar_len = int(pct / 2)  # Scale bar to ~50 chars max
            bar = '█' * bar_len
            print(f"  {bin_range:>12}: {count:>8,} ({pct:>5.1f}%) {bar}")

    if skipped_length_stats:
        print(f"\nSkipped sample length stats (n={skipped_length_stats['count']:,}):")
        print(f"  Min: {skipped_length_stats['min']:,}")
        print(f"  Max: {skipped_length_stats['max']:,}")
        print(f"  Mean: {skipped_length_stats['mean']:,.1f}")
        print(f"  Median: {skipped_length_stats['median']:,.1f}")

    print("\n" + "=" * 80)
    print("PREPROCESSING COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    main()
