"""Category-isolated Stage-3 packs with stable raw-row provenance.

The three independent bin packers only meet after they emit complete sequences.
Both length variants consume the same processed rows; overlength examples are
excluded whole, never truncated. This module does not import or select generated
expansion data, whose separate review/export gates are still pending.
"""
from __future__ import annotations

import copy
import hashlib
import json
import pickle
import random
from collections import Counter
from pathlib import Path

REASONING_SOURCES = frozenset({'reasoning_data', 'dolci_think'})
OTHER_SOURCES = frozenset({
    'Nemotron_MCQ', 'Nemotron_RQA', 'chatqa2_long_sft', 'chatqa2_narrativeqa_long',
    'chatqa_v1', 'code-instruct', 'dolci_instruct', 'finewiki_en',
    'helpsteer_v1_train', 'helpsteer_v2_train', 'helpsteer_v3_edit_quality_train',
    'helpsteer_v3_edit_train', 'helpsteer_v3_feedback_train', 'helpsteer_v3_preference_train',
    'hotpotqa_distractor', 'latex_formulas_en_10m', 'lmsys_chat_1m', 'long_data_booksum',
    'nemotron_math_4plus',
    'long_data_nq', 'nemotron_cc_v2_hq_synth', 'nemotron_instruction_following',
    'pubmedqa_artificial', 'pubmedqa_labeled', 'pubmedqa_unlabeled', 'rag_instruct',
    'redpajama_arxiv', 'redpajama_github', 'repo_forward', 'repo_reverse',
    'repo_summarize', 'stack_v2_repo_recon', 'triviaqa_rc', 'tulu3_sft_mixture', 'wildchat',
})
CATEGORIES = ('agent', 'reasoning', 'other')
LENGTHS = (16384, 32768)
MEMORY_START = '<|memory_start|>'
MEMORY_END = '<|memory_end|>'


def category_for(source_kind, sub_dataset):
    if source_kind == 'native':
        return 'agent'
    if source_kind != 'base':
        raise ValueError(f'Unapproved source kind: {source_kind}')
    if sub_dataset in REASONING_SOURCES:
        return 'reasoning'
    if sub_dataset in OTHER_SOURCES:
        return 'other'
    raise ValueError(f'Unclassified base subdataset: {sub_dataset!r}')


def has_memory(value):
    if isinstance(value, str):
        return any(token in value for token in (MEMORY_START, MEMORY_END, '<|memory|>'))
    if isinstance(value, dict):
        return any(has_memory(v) for v in value.values())
    if isinstance(value, (tuple, list)):
        return any(has_memory(v) for v in value)
    return False


def apply_reasoning_policy(row, row_id, tokenizer=None, rewrite_math=False):
    """Preserve verified CoT50 rows and leave passage-reconstruction data intact."""
    sub = row['sub_dataset']
    if sub not in REASONING_SOURCES:
        return row, 'not_reasoning'
    if has_memory(row['prompt']):
        raise ValueError(f'Reasoning original prompt contains memory markers: {row_id}')
    row = copy.deepcopy(row)
    if row['compression_prompt'] != row['prompt']:
        raise ValueError(f'Previously corrected reasoning prompt differs: {row_id}')
    return row, 'preserved_cot_memory' if has_memory(row['target']) else 'preserved_uncompressed'


def process_raw_item(item):
    """Multiprocessing target. Return metadata even for processing exclusions."""
    from data import preprocess_for_dynamic_packing as pre
    source_kind, row_id, row, rewrite_math = item
    sub = row['sub_dataset']
    category = category_for(source_kind, sub)
    policy = 'native_uncompressed' if source_kind == 'native' else 'not_reasoning'
    if source_kind == 'base':
        row, policy = apply_reasoning_policy(row, row_id, pre._worker_tokenizer, rewrite_math)
    elif has_memory(row):
        raise ValueError(f'Native source unexpectedly compressed: {row_id}')
    output = pre.worker_process_example((row, 16, 'compression_prompt', 32768, None))
    info = {'row_id': row_id, 'sub_dataset': sub, 'category': category, 'policy': policy}
    if output is not None:
        output['_source_row_id'] = row_id
        output['_packing_category'] = category
        output['_reasoning_policy'] = policy
        if not output.get('_skipped_max_seq_len'):
            validate_example(output)
    return info, output


def exclusion_reason(example, length):
    if example is None:
        return 'processing_rejected'
    if example['estimated_seq_len'] < 18:
        return 'under_18'
    if example['estimated_seq_len'] > length:
        return 'overlength'
    assert not example.get('_skipped_max_seq_len')
    return None


def proportional_interleave(streams, sizes):
    """Weighted interleave without changing any source's file/row identity."""
    iterators = {key: iter(stream) for key, stream in streams.items()}
    seen = Counter()
    while iterators:
        key = min(iterators, key=lambda name: seen[name] / max(1, sizes[name]))
        try:
            item = next(iterators[key])
        except StopIteration:
            del iterators[key]
            continue
        seen[key] += 1
        yield item


def validate_example(row):
    assert len(row['base_input_ids']) == len(row['base_labels'])
    assert len(row['memory_strings']) == len(row['memory_positions'])
    assert any(v != -100 for v in row['base_labels'])
    for position in row['memory_positions']:
        assert row['base_labels'][position:position + 3] == [-100, -100, -100]
    if row['_packing_category'] == 'agent':
        assert not row['memory_strings']
    assert row['estimated_seq_len'] >= len(row['base_input_ids'])


def validate_pack(batch, length):
    assert batch
    categories = {row['_packing_category'] for row in batch}
    assert len(categories) == 1, categories
    category = next(iter(categories))
    assert category in CATEGORIES
    actual_length = sum(row['estimated_seq_len'] for row in batch)
    assert 0 < actual_length <= length, (actual_length, length)
    assert len({row['_source_row_id'] for row in batch}) == len(batch)
    return category, actual_length


class ShuffledSequenceWriter:
    """Shuffle whole category-pure sequences into mixed Parquet shards."""
    def __init__(self, output_dir, length, seed, buffer_size=512):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.length = length
        self.rng = random.Random(seed)
        self.buffer_size = buffer_size
        self.buffer = []
        self.files = []
        self.counts = {category: Counter() for category in CATEGORIES}
        self.digest = hashlib.sha256()

    def write(self, batch):
        category, length = validate_pack(batch, self.length)
        self.buffer.append({
            'packed_batch_bytes': pickle.dumps(batch, protocol=5),
            'packing_category': category, 'expanded_seq_len_cs16': length,
            'example_count': len(batch),
        })
        count = self.counts[category]
        count['packed_sequences'] += 1
        count['packed_rows'] += len(batch)
        count['decoder_tokens_cs16'] += length
        count['labeled_tokens'] += sum(row['_trainable_tokens'] for row in batch)
        count['memory_blocks'] += sum(len(row['memory_strings']) for row in batch)
        count['max_sequence_length'] = max(count['max_sequence_length'], length)
        if len(self.buffer) >= self.buffer_size:
            self.flush()

    def flush(self):
        if not self.buffer:
            return
        import pyarrow as pa
        import pyarrow.parquet as pq
        self.rng.shuffle(self.buffer)
        name = f'packed-{len(self.files):06d}.parquet'
        path = self.output_dir / name
        pq.write_table(pa.Table.from_pylist(self.buffer), path, compression='snappy')
        self.digest.update(path.read_bytes())
        self.files.append(name)
        self.buffer = []


def make_packers(length, seed):
    from data.preprocess_for_dynamic_packing import StreamingPacker
    return {category: StreamingPacker(length, 128, seed + index, 32)
            for index, category in enumerate(CATEGORIES)}


def canonical_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
