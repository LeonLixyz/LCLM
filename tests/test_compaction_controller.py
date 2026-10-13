import pytest

from compaction.controller import (
    KEEP_RECENT,
    MAX_CONTEXT,
    context_stats,
    generate_full_context,
    generate_with_compaction,
    run,
    safe_cut,
)


class FakeBackend:
    """Emits ``length`` letters in total, then stops; counts latent blocks as len/16 + 2 delimiters."""

    def __init__(self, length=2600):
        self.length = length
        self.emitted = 0
        self.requests = []

    def prefix(self, messages):
        assert len(messages) == 1 and messages[0]['role'] == 'user'
        return [901, 902]

    def decode(self, ids, final=False):
        return ''.join(chr(x) for x in ids)

    def generate(self, request, max_tokens):
        self.requests.append(request)
        n = min(max_tokens, self.length - self.emitted)
        self.emitted += n
        tokens = [65 + i % 20 for i in range(self.emitted - n, self.emitted)]
        positions = (len(request['prefix_ids']) + len(request['recent_ids'])
                     + sum(len(m['token_ids']) // 16 + 2 for m in request['memories']))
        return dict(token_ids=tokens, finish_reason='stop' if self.emitted == self.length else 'length',
                    prompt_positions=positions)


def test_prompt_intact_and_only_generated_tokens_compacted():
    backend = FakeBackend()
    row = run(backend, [{'role': 'user', 'content': 'question'}], compact=True)
    assert row['stop_reason'] == 'eos'
    assert len(row['compactions']) == 4 and row['compacted_tokens'] == 2048
    for request in backend.requests:
        assert request['prefix_ids'] == [901, 902]
        if request['memories']:
            assert len(request['recent_ids']) >= KEEP_RECENT
    # The model sees memories + recent tokens that exactly reconstruct its own output.
    for request, info in zip(backend.requests, row['requests']):
        rebuilt = [t for m in request['memories'] for t in m['token_ids']] + request['recent_ids']
        assert rebuilt == row['generated_ids'][:info['generated_before']]


def test_compaction_schedule():
    row = generate_with_compaction(FakeBackend(), 'question')
    assert [c['at_generated_tokens'] for c in row['compactions']] == [768, 1280, 1792, 2304]
    assert [r['allowance'] for r in row['requests']] == [768, 512, 512, 512, 512]
    assert all(c['source_end'] <= c['at_generated_tokens'] - KEEP_RECENT for c in row['compactions'])
    assert all(c['recent_tokens'] == KEEP_RECENT for c in row['compactions'])


def test_short_solution_identical_before_compaction():
    full = generate_full_context(FakeBackend(700), 'q')
    compact = generate_with_compaction(FakeBackend(700), 'q')
    assert full['generated_ids'] == compact['generated_ids']
    assert compact['compactions'] == []
    assert full['avg_context_positions'] == compact['avg_context_positions']


@pytest.mark.parametrize('compact', [False, True])
def test_32k_total_budget_is_identical_in_both_modes(compact):
    row = run(FakeBackend(40000), [{'role': 'user', 'content': 'q'}], compact=compact)
    assert row['output_budget'] == MAX_CONTEXT - 2
    assert row['generated_tokens'] == MAX_CONTEXT - 2
    assert row['peak_context_positions'] <= MAX_CONTEXT
    assert row['stop_reason'] == 'output_cap'
    if not compact:
        assert row['peak_context_positions'] == MAX_CONTEXT and len(row['requests']) == 1


def test_budget_applies_to_all_generated_tokens():
    row = generate_with_compaction(FakeBackend(9999), 'q', max_tokens=2100)
    assert row['generated_tokens'] == 2100 and row['stop_reason'] == 'output_cap'


def test_unicode_cut_is_lossless():
    def decode(ids):
        return bytes(ids).decode('utf-8', errors='replace')
    ids = list(('a' * 511 + '界' + 'b' * 300).encode())
    cut = safe_cut(ids, 512, decode)
    assert cut == 511
    assert decode(ids[:cut]) + decode(ids[cut:]) == decode(ids)


def test_context_stats_formula():
    # Two requests: contexts [11, 12, 13], then after compaction [6, 7].
    stats = context_stats([dict(prompt_positions=10, generated_tokens=3), dict(prompt_positions=5, generated_tokens=2)])
    assert stats['position_token_sum'] == sum([11, 12, 13, 6, 7]) == 49
    assert stats['avg_context_positions'] == 49 / 5
    assert stats['peak_context_positions'] == 13


def test_full_context_average_is_prompt_plus_half_output():
    row = generate_full_context(FakeBackend(3000), 'q')
    assert row['avg_context_positions'] == row['prompt_tokens'] + (row['generated_tokens'] + 1) / 2
    assert row['peak_context_positions'] == row['prompt_tokens'] + row['generated_tokens']


def test_compaction_peak_matches_request_layout():
    row = generate_with_compaction(FakeBackend(2600), 'q')
    # Request k (k >= 1): prompt 2 + k blocks * (512/16 latents + 2 delimiters) + 256 recent.
    assert [r['prompt_positions'] for r in row['requests']] == [2] + [2 + 34 * k + 256 for k in range(1, 5)]
    # Peaks per request: 770, 804, 838, 872, 690 (last request emits 296 tokens).
    assert row['peak_context_positions'] == 872
