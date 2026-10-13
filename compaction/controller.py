"""Online latent compaction of a model's own generated tokens.

A trajectory is a sequence of decoder requests. Each request is

    [prompt] + [<|memory_start|> latents <|memory_end|>] * k + [recent raw tokens]

With compaction enabled, the first request may generate
``block_size + keep_recent`` tokens. Afterwards, whenever at least
``block_size + keep_recent`` generated tokens are still raw, the oldest
``block_size`` of them are moved into a new latent block, leaving
``keep_recent`` raw tokens, and the next request generates ``block_size``
more tokens. Earlier latent blocks are reused unchanged. The full-context
reference generates the whole answer in a single request.

The controller is backend-agnostic: ``trajectory`` yields
``(request, max_new_tokens)`` pairs and receives backend outputs, so the
same logic drives serial generation (``run``) and the continuous-batching
scheduler (``compaction.scheduler.serve``).
"""
import hashlib
import json

BLOCK_SIZE = 512
KEEP_RECENT = 256
MAX_CONTEXT = 32768


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def safe_cut(ids, block, decode):
    """Largest cut <= ``block`` that does not split a multi-token Unicode character."""
    whole = decode(ids)
    for n in range(min(block, len(ids)), max(0, min(block, len(ids)) - 32), -1):
        left, right = decode(ids[:n]), decode(ids[n:])
        if left + right == whole and not left.endswith('�') and not right.startswith('�'):
            return n
    raise ValueError('No lossless Unicode boundary near the requested compaction cut')


def trajectory(prefix_ids, decode, *, compact=True, block_size=BLOCK_SIZE, keep_recent=KEEP_RECENT,
               max_context=MAX_CONTEXT, max_tokens=None):
    """Generator driving one generation. Yields ``(request, max_new_tokens)``; returns the result dict.

    ``request`` is ``dict(prefix_ids, memories, recent_ids)``; each memory is
    ``dict(source_start, source_end, token_ids, text)`` over generated tokens.
    The backend must answer with at least ``token_ids``, ``finish_reason``
    (``'stop'`` or ``'length'``) and ``prompt_positions`` (decoder positions of
    the request, latents and delimiters included).

    The output budget is ``max_context - len(prefix_ids)`` (optionally capped by
    ``max_tokens``) in both modes, so compaction never buys extra output tokens.
    """
    prefix_ids = list(prefix_ids)
    budget = max_context - len(prefix_ids)
    if max_tokens is not None:
        budget = min(budget, max_tokens)
    assert budget > 0, 'Prompt alone exceeds the context limit'
    block = block_size if compact else None
    generated, memories, requests, compactions = [], [], [], []
    consumed = 0
    while len(generated) < budget:
        remaining = budget - len(generated)
        allowance = remaining if block is None else min(remaining, block + keep_recent if not generated else block)
        request = dict(prefix_ids=prefix_ids, memories=[dict(m) for m in memories], recent_ids=generated[consumed:])
        output = yield request, allowance
        tokens = list(output['token_ids'])
        assert tokens and len(tokens) <= allowance
        assert output['finish_reason'] in ('stop', 'length'), output['finish_reason']
        requests.append(dict(generated_before=len(generated), consumed_before=consumed, allowance=allowance,
                             prompt_positions=output['prompt_positions'], generated_tokens=len(tokens),
                             finish_reason=output['finish_reason'],
                             memory_blocks=output.get('memory_blocks', [])))
        generated.extend(tokens)
        if output['finish_reason'] == 'stop' or len(generated) == budget:
            break
        if compact:
            while len(generated) - consumed >= block + keep_recent:
                n = safe_cut(generated[consumed:], block, decode)
                ids = generated[consumed:consumed + n]
                text = decode(ids)
                assert text and decode(generated[:consumed]) + text + decode(generated[consumed + n:]) == decode(generated)
                memories.append(dict(source_start=consumed, source_end=consumed + n, token_ids=ids, text=text))
                consumed += n
                compactions.append(dict(at_generated_tokens=len(generated), source_start=consumed - n,
                                        source_end=consumed, recent_tokens=len(generated) - consumed))
        assert sum(len(m['token_ids']) for m in memories) == consumed
        assert [x for m in memories for x in m['token_ids']] + generated[consumed:] == generated
    return dict(mode='compaction' if compact else 'full', prompt_tokens=len(prefix_ids), output_budget=budget,
                max_context=max_context, generated_ids=generated, text=decode(generated, final=True),
                stop_reason='eos' if requests[-1]['finish_reason'] == 'stop' else 'output_cap',
                generated_tokens=len(generated), compacted_tokens=consumed, compactions=compactions,
                requests=requests, **context_stats(requests))


def context_stats(requests):
    """Active decoder positions after each generated token.

    A request with ``p`` prompt positions that emits ``n`` tokens occupies
    ``p+1, ..., p+n`` positions, summing to ``n*p + n*(n+1)/2``. The average is
    this sum over all requests divided by the generated-token count; the peak is
    the largest ``p+n``. Prompt, latent tokens, memory delimiters and recent raw
    tokens all count.
    """
    count = sum(r['generated_tokens'] for r in requests)
    area = sum(r['generated_tokens'] * r['prompt_positions'] + r['generated_tokens'] * (r['generated_tokens'] + 1) // 2
               for r in requests)
    return dict(position_token_sum=area, avg_context_positions=area / count,
                peak_context_positions=max(r['prompt_positions'] + r['generated_tokens'] for r in requests),
                total_prefill_positions=sum(r['prompt_positions'] for r in requests))


def run(backend, messages, **kwargs):
    """Run one trajectory serially on ``backend`` (see ``trajectory`` for kwargs).

    ``max_context`` defaults to the backend's context limit.
    """
    if kwargs.get('max_context') is None:
        kwargs['max_context'] = getattr(backend, 'max_context', MAX_CONTEXT)
    stream = trajectory(backend.prefix(messages), backend.decode, **kwargs)
    request, allowance = next(stream)
    try:
        while True:
            request, allowance = stream.send(backend.generate(request, allowance))
    except StopIteration as done:
        return done.value
    finally:
        if hasattr(backend, 'clear_cache'):
            backend.clear_cache()


def _as_messages(prompt):
    return [{'role': 'user', 'content': prompt}] if isinstance(prompt, str) else prompt


def generate_with_compaction(backend, prompt, *, block_size=BLOCK_SIZE, keep_recent=KEEP_RECENT,
                             max_context=None, max_tokens=None):
    """Generate with online latent compaction.

    ``prompt`` is a user message string or a list of chat messages. Returns a
    dict with ``text``, ``generated_tokens``, ``stop_reason``,
    ``avg_context_positions``, ``peak_context_positions``, the list of
    ``compactions`` and per-request details.
    """
    return run(backend, _as_messages(prompt), compact=True, block_size=block_size, keep_recent=keep_recent,
               max_context=max_context, max_tokens=max_tokens)


def generate_full_context(backend, prompt, *, max_context=None, max_tokens=None):
    """Full-context reference: one request, same output budget and statistics."""
    return run(backend, _as_messages(prompt), compact=False, max_context=max_context, max_tokens=max_tokens)
