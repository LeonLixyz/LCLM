"""HFBackend on a tiny random decoder and a fake encoder (CPU)."""
import math
import re
from types import SimpleNamespace

import pytest

torch = pytest.importorskip('torch')
transformers = pytest.importorskip('transformers')

from compaction import HFBackend, generate_full_context, generate_with_compaction  # noqa: E402

SPECIAL = {'<|memory_start|>': 256, '<|memory_end|>': 257, '<|memory|>': 258}
VOCAB = 259
EOS = 1000  # Outside the vocabulary: generation always runs to the output budget.


class CharTokenizer:
    eos_token_id = EOS
    pad_token_id = 0

    def encode(self, text, add_special_tokens=False):
        ids = []
        for part in re.split('(' + '|'.join(map(re.escape, SPECIAL)) + ')', text):
            ids.extend([SPECIAL[part]] if part in SPECIAL else [ord(c) for c in part])
        return ids

    def decode(self, ids, skip_special_tokens=False, clean_up_tokenization_spaces=False):
        names = {v: k for k, v in SPECIAL.items()}
        return ''.join(('' if skip_special_tokens else names[i]) if i in names else chr(i) for i in ids)

    def convert_tokens_to_ids(self, token):
        return SPECIAL[token]

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return ''.join(f"<{m['role']}>{m['content']}" for m in messages) + '<assistant>'


class FakeLCLM:
    def __init__(self, ratio=16):
        torch.manual_seed(0)
        config = transformers.Qwen3Config(vocab_size=VOCAB, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                          num_attention_heads=4, num_key_value_heads=2, head_dim=8,
                                          max_position_embeddings=4096, eos_token_id=EOS)
        self.decoder = transformers.Qwen3ForCausalLM(config).eval()
        self.decoder.generation_config.eos_token_id = EOS
        self.encoder = SimpleNamespace(compression_ratio=ratio, embed_tokenizer=CharTokenizer())
        self.encoded = []

    def _process_latent_embeddings(self, memory_token_ids):
        self.encoded.append([len(ids) for ids in memory_token_ids])
        return [torch.full((max(1, math.ceil(len(ids) / self.encoder.compression_ratio)), 32), 0.01 * len(ids))
                for ids in memory_token_ids]


def make_backend(temperature=0.0, max_context=4096):
    model = FakeLCLM()
    sampling = dict(temperature=temperature, top_p=0.8, top_k=20, min_p=0.0, seed=0)
    return model, HFBackend(model, CharTokenizer(), max_context=max_context, sampling=sampling)


def test_build_splices_latents_between_delimiters():
    model, backend = make_backend()
    generated = [ord('a') + i % 26 for i in range(100)]
    memory = dict(source_start=0, source_end=64, token_ids=generated[:64], text=backend.decode(generated[:64]))
    request = dict(prefix_ids=[1, 2, 3], memories=[memory], recent_ids=generated[64:])
    with torch.inference_mode():
        (embeds, meta), = backend.build([('t', request)])
        table = model.decoder.get_input_embeddings().weight
        ids = [1, 2, 3, SPECIAL['<|memory_start|>']] + [SPECIAL['<|memory|>']] * 4 + [SPECIAL['<|memory_end|>']] + generated[64:]
        assert meta['prompt_positions'] == len(ids) == 3 + 4 + 2 + 36
        assert meta['memory_blocks'] == [dict(encoder_tokens=64, latent_tokens=4)]
        assert torch.equal(embeds[4:8], torch.full((4, 32), 0.64))
        keep = [i for i in range(len(ids)) if not 4 <= i < 8]
        assert torch.equal(embeds[keep], table[torch.tensor(ids)][keep])


def test_generate_with_compaction_reuses_cached_latents():
    model, backend = make_backend()
    row = generate_with_compaction(backend, 'question', block_size=64, keep_recent=32, max_tokens=300)
    assert row['generated_tokens'] == 300 and row['stop_reason'] == 'output_cap'
    assert [c['at_generated_tokens'] for c in row['compactions']] == [96, 160, 224, 288]
    # Each block is encoded exactly once although later requests include it again.
    assert model.encoded == [[64]] * 4
    prompt = row['prompt_tokens']
    assert [r['prompt_positions'] for r in row['requests']] == [prompt] + [prompt + 6 * k + 32 for k in range(1, 5)]
    assert backend.caches == {}


def test_full_context_and_sampling_paths_run():
    _, backend = make_backend(temperature=0.7)
    row = generate_full_context(backend, 'question', max_tokens=40)
    assert row['generated_tokens'] == 40 and len(row['requests']) == 1
    assert row['avg_context_positions'] == row['prompt_tokens'] + 41 / 2


def test_context_limit_is_enforced():
    _, backend = make_backend(max_context=64)
    row = generate_full_context(backend, 'question')
    assert row['peak_context_positions'] == 64
