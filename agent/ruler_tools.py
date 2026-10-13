"""Segmented RULER NIAH with a native ``expand`` tool over compressed segments.

No GPU imports: task preparation, the episode loops and scoring run on CPU with
any backend exposing ``generate(messages, tools) -> dict``. Answers are used only
after the final generation, for grading.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
import json
import re

from agent.tool_eval import parse_tool_calls
from agent.tool_tasks import TOOLS, fingerprint

SUBTASKS = ('niah_single_1', 'niah_single_2', 'niah_single_3',
            'niah_multikey_1', 'niah_multikey_2', 'niah_multikey_3',
            'niah_multivalue', 'niah_multiquery')
MODES = ('plain', 'compressed', 'compressed_expand')
START, END = '<|memory_start|>', '<|memory_end|>'
SEED = 2026092001
SEGMENT_TOKENS = 1024
# Expansion-call caps used in the paper, per RULER context length.
MAX_CALLS = {4096: 16, 8192: 16, 16384: 32, 32768: 16}

STOP_RULES = (
    '\nUse expand selectively to verify the exact values requested in the question. '
    'Issue at most one tool call per assistant turn. Read each result before choosing another action. '
    'Do not scan every segment or expand segments in numerical order. '
    'Do not expand the same segment again. Only use segment IDs shown in the context. '
    'After each result, check whether you can answer the question; once you have the needed '
    'evidence, give the final answer immediately. If a value is cut off at a segment boundary, '
    'read the adjacent segment before copying that value. '
    'You have at most {max_calls} calls in total. After the {ordinal} call, give your best supported final '
    'answer with the values you found; do not request another call or invent missing values. '
    'A final answer must contain the requested values, not a plan to look for them.'
)
FINAL_NOTICE = ('Expansion is now disabled. Answer the original question using the original '
                'context and the tool results already available. Return only the requested '
                'answer values you can support. Do not call tools, describe a plan, or invent '
                'missing values. Give your final answer now.')
_ORDINALS = {8: 'eighth', 16: 'sixteenth', 32: 'thirty-second'}


def ordinal(n):
    if n in _ORDINALS:
        return _ORDINALS[n]
    suffix = 'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')
    return f'{n}{suffix}'


def extract_prompt(prompt):
    """Strictly unpack the single-system/single-user Qwen chat template.

    Two NIAH types wrap the terminal question in memory. Move that question out,
    preserving every source character. Never infer it from the ground truth.
    """
    match = re.fullmatch(r'<\|im_start\|>system\n(.*?)<\|im_end\|>\n'
                         r'<\|im_start\|>user\n(.*?)<\|im_end\|>\n'
                         r'<\|im_start\|>assistant\n', prompt, re.S)
    if not match:
        raise ValueError('Unexpected source chat template')
    system, user = match.groups()
    if user.count(START) != 1 or user.count(END) != 1:
        raise ValueError('Expected one source memory span')
    prefix, tail = user.split(START)
    body, suffix = tail.split(END)
    moved = not suffix.strip()
    if moved:
        question = re.search(r'\nWhat (?:is|are) (?:all )?the special magic (?:numbers?|uuids?) for [^\n]+\n?$', body)
        if question is None:
            raise ValueError('Cannot locate terminal question inside source memory')
        body, suffix = body[:question.start()], body[question.start():]
    if not re.match(r'\s*What (?:is|are) (?:all )?the special magic (?:numbers?|uuids?) for ', suffix):
        raise ValueError('Unexpected NIAH question')
    if prefix + body + suffix != user.replace(START, '').replace(END, ''):
        raise AssertionError('Source text changed during prompt extraction')
    if not body.strip():
        raise ValueError('Empty haystack')
    return dict(system=system, prefix=prefix, body=body, question=suffix,
                question_moved_out_of_memory=moved)


def segment_text(text, tokenizer, chunk_tokens=SEGMENT_TOKENS):
    """Lossless character slices of at most ``chunk_tokens`` encoder tokens.

    Cuts prefer a nearby newline. Character offsets avoid splitting multibyte
    characters; boundaries depend only on the text and tokenizer.
    """
    if chunk_tokens < 2 or not text:
        raise ValueError('Need nonempty text and at least two tokens per segment')
    segments = []
    cursor = 0
    while cursor < len(text):
        remaining = text[cursor:]
        encoded = tokenizer(remaining, add_special_tokens=False, return_offsets_mapping=True)
        if len(encoded['input_ids']) <= chunk_tokens:
            cut = len(remaining)
        else:
            cut = encoded['offset_mapping'][chunk_tokens][0]
            if cut <= 0:
                raise ValueError('Tokenizer cannot supply a safe character boundary')
            newline = remaining.rfind('\n', max(1, cut - 256), cut)
            if newline >= 0:
                cut = newline + 1
            # Tokenization can change at the new boundary; enforce the actual cap.
            while cut > 0 and len(tokenizer.encode(remaining[:cut], add_special_tokens=False)) > chunk_tokens:
                cut -= 1
        if cut <= 0:
            raise ValueError('Unable to make progress while segmenting')
        chunk = remaining[:cut]
        segments.append(dict(segment_id=f'seg_{len(segments)+1}', text=chunk,
                             char_start=cursor, char_end=cursor+cut,
                             encoder_tokens=len(tokenizer.encode(chunk, add_special_tokens=False))))
        cursor += cut
    assert ''.join(s['text'] for s in segments) == text
    assert all(0 < s['encoder_tokens'] <= chunk_tokens for s in segments)
    return segments


def sample_rank(row, seed=SEED):
    """Selection key from the task type and source index only (shared across lengths)."""
    extra = row['extra_info']
    return fingerprint([seed, extra['ruler_task'], extra['ground_truth']['index']])


def select_rows(rows, n, seed=SEED, indices=None):
    """First ``n`` rows per subtask, by seeded hash rank or by an explicit index list."""
    by_task = defaultdict(list)
    for row in rows:
        by_task[row['extra_info']['ruler_task']].append(row)
    selected = []
    for subtask, candidates in by_task.items():
        if indices is None:
            chosen = sorted(candidates, key=lambda r: sample_rank(r, seed))[:n]
        else:
            by_index = {r['extra_info']['ground_truth']['index']: r for r in candidates}
            chosen = [by_index[i] for i in indices[subtask][:n]]
        selected.extend(chosen)
    return selected


def prepare_task(row, tokenizer, chunk_tokens=SEGMENT_TOKENS):
    extra = row['extra_info']
    assert extra['ruler_task'] in SUBTASKS
    assert extra['scoring_function'] == 'ruler_string_match_all'
    parts = extract_prompt(row['prompt'])
    segments = segment_text(parts.pop('body'), tokenizer, chunk_tokens)
    answers = extra['ground_truth']['answers']
    assert answers and all(isinstance(a, str) and a for a in answers)
    index = extra['ground_truth']['index']
    return dict(task_id=f"r{extra['context_length']}-{extra['ruler_task']}-{index:05d}",
                length=extra['context_length'], subtask=extra['ruler_task'], source_index=index,
                segments=segments, answers=answers, **parts)


def initial_messages(task, mode, max_calls=8):
    if mode not in MODES:
        raise ValueError(mode)
    instruction = ('\nThe context below is split into numbered segments in its original order. '
                   'Answer the question using that context. Return only the requested answer values, '
                   'without repeating the context.')
    if mode == 'compressed_expand':
        instruction += (f' You may call expand to read the exact original text of a selected segment. '
                        f'Use at most {max_calls} calls, then give your answer. '
                        'Select segments based on their content; only expand what you need.')
    blocks = []
    for segment in task['segments']:
        body = segment['text']
        if mode != 'plain':
            body = START + body + END
        blocks.append(f"[Segment {segment['segment_id']}]\n{body}\n[/Segment {segment['segment_id']}]")
    return [dict(role='system', content=task['system'] + instruction),
            dict(role='user', content=task['prefix'] + '\n'.join(blocks) + task['question'])], \
           ([deepcopy(TOOLS['expand'])] if mode == 'compressed_expand' else [])


def agent_messages(task, max_calls):
    """Initial agent prompt: compressed segments, the expand tool and a visible call budget."""
    messages, tools = initial_messages(task, 'compressed_expand', max_calls)
    messages[0]['content'] += STOP_RULES.format(max_calls=max_calls, ordinal=ordinal(max_calls))
    messages[0]['content'] += (f'\nExpansion budget remaining: {max_calls} of {max_calls}. '
                               'You may answer now without expanding.')
    return messages, tools


def execute_expand(call, segments, enabled):
    args = call['arguments']
    valid = (enabled and call['name'] == 'expand' and isinstance(args, dict)
             and set(args) == {'segment_id'} and isinstance(args['segment_id'], str)
             and re.fullmatch(r'seg_[1-9][0-9]*', args['segment_id']) is not None)
    if not valid:
        return {'error': 'invalid_tool_or_arguments'}, False, False
    if args['segment_id'] not in segments:
        return {'error': 'unknown_segment_id'}, True, False
    return segments[args['segment_id']], True, True


def ruler_match(response, answers):
    """Mean case-insensitive substring coverage of the reference answers (RULER string_match_all)."""
    return 100.0 * sum(answer.lower() in response.lower() for answer in answers) / len(answers)


def _result(task, mode, final, reason, generations, events, messages, initial_encoder_tokens):
    # Answers enter only here, after generation, for grading and diagnostics.
    score = ruler_match(final, task['answers']) if reason == 'final' else 0.0
    expanded = {e['arguments']['segment_id'] for e in events if e['success']}
    original = ''.join(s['text'] for s in task['segments'])
    # Character-level coverage handles an answer straddling two segments.
    coverage = []
    for answer in task['answers']:
        occurrences = list(re.finditer(re.escape(answer), original, flags=re.I))
        assert occurrences, 'Ground truth answer missing from original haystack'
        coverage.append(any(all(s['segment_id'] in expanded for s in task['segments']
                                if s['char_start'] < m.end() and s['char_end'] > m.start())
                            for m in occurrences))
    return dict(task_id=task['task_id'], length=task['length'], subtask=task['subtask'], mode=mode,
                source_index=task['source_index'],
                ruler_match=score, all_answers_correct=score == 100, stop_reason=reason,
                final=final, expected=task['answers'], turns=len(generations), tool_calls=len(events),
                schema_valid_calls=sum(e['schema_valid'] for e in events),
                successful_tool_calls=sum(e['success'] for e in events),
                expanded_segments=sorted(expanded), expanded_answer_coverage=sum(coverage)/len(coverage),
                output_tokens=sum(g['output_tokens'] for g in generations),
                total_prompt_positions=sum(g['prompt_positions'] for g in generations),
                first_prompt_positions=generations[0]['prompt_positions'],
                total_uncompressed_prompt_tokens=sum(g['uncompressed_prompt_tokens'] for g in generations),
                initial_encoder_tokens=initial_encoder_tokens,
                generations=generations, events=events, messages=messages)


def _append_tool_turn(text, calls, turn, messages, events, segments):
    native = [dict(id=f'call_{turn}_{i}', type='function', function=c) for i, c in enumerate(calls)]
    prose = re.sub(r'<tool_call>.*?</tool_call>', '', text, flags=re.S).strip()
    messages.append(dict(role='assistant', content=prose, tool_calls=native))
    for call in native:
        result, valid, success = execute_expand(call['function'], segments, True)
        events.append(dict(**call['function'], schema_valid=valid, success=success, result=result))
        messages.append(dict(role='tool', name=call['function']['name'], tool_call_id=call['id'],
                             content=result if success else json.dumps(result)))


def run_episode(task, mode, backend, max_turns=9, max_calls=8):
    """One episode with a fixed prompt; ``compressed`` is the no-expansion baseline."""
    messages, tools = initial_messages(task, mode, max_calls)
    segments = {s['segment_id']: s['text'] for s in task['segments']}
    generations, events = [], []
    final, reason = '', 'turn_limit'
    for turn in range(max_turns):
        generation = backend.generate(messages, tools)
        generations.append(generation)
        text = generation['text']
        if generation['finish_reason'] == 'length':
            reason = 'output_cap'
            break
        try:
            calls = parse_tool_calls(text)
        except (ValueError, TypeError) as exc:
            reason = 'protocol_error:' + str(exc)
            break
        if not calls:
            final, reason = text, 'final'
            messages.append(dict(role='assistant', content=text))
            break
        if not tools:
            reason = 'tool_unavailable'
            break
        if len(events) + len(calls) > max_calls:
            reason = 'tool_call_limit'
            break
        _append_tool_turn(text, calls, turn, messages, events, segments)
    encoder_tokens = sum(s['encoder_tokens'] for s in task['segments']) if mode != 'plain' else 0
    return _result(task, mode, final, reason, generations, events, messages, encoder_tokens)


def run_agent(task, backend, max_calls):
    """The paper's agent: expand with a visible budget, then a forced tool-free final answer.

    When the budget is spent, or a turn requests more calls than remain (no call
    of that turn is executed), the model is told expansion is disabled and must
    answer without tools. Unfinished or malformed episodes score zero.
    """
    messages, tools = agent_messages(task, max_calls)
    segments = {s['segment_id']: s['text'] for s in task['segments']}
    generations, events = [], []
    final, reason, trigger = '', 'turn_limit', None

    def finish(why):
        nonlocal final, reason, trigger
        trigger = why
        messages.append(dict(role='user', content=FINAL_NOTICE))
        generation = backend.generate(messages, [])
        generations.append(generation)
        if generation['finish_reason'] == 'length':
            reason = 'output_cap'
            return
        try:
            calls = parse_tool_calls(generation['text'])
        except (ValueError, TypeError) as exc:
            reason = 'protocol_error:' + str(exc)
            return
        if calls:
            reason = 'tool_unavailable'
            return
        final, reason = generation['text'], 'final'
        messages.append(dict(role='assistant', content=final))

    for turn in range(max_calls + 1):
        if len(events) == max_calls:
            finish('budget_exhausted')
            break
        generation = backend.generate(messages, tools)
        generations.append(generation)
        text = generation['text']
        if generation['finish_reason'] == 'length':
            reason = 'output_cap'
            break
        try:
            calls = parse_tool_calls(text)
        except (ValueError, TypeError) as exc:
            reason = 'protocol_error:' + str(exc)
            break
        if not calls:
            final, reason = text, 'final'
            messages.append(dict(role='assistant', content=text))
            break
        if len(events) + len(calls) > max_calls:
            finish('over_budget_batch_rejected')
            break
        _append_tool_turn(text, calls, turn, messages, events, segments)
        if len(events) < max_calls:
            remaining = max_calls - len(events)
            messages.append(dict(role='user', content=f'Expansion budget remaining: {remaining} of {max_calls}. '
                                 'Answer the original question now if you have enough evidence. '
                                 'Otherwise select only the next needed segment.'))
    row = _result(task, 'compressed_expand', final, reason, generations, events, messages,
                  sum(s['encoder_tokens'] for s in task['segments']))
    row.update(max_calls=max_calls, forced_final=trigger is not None, finalization_trigger=trigger)
    return row


def aggregate(rows):
    """Mean score per ``length/condition/subtask`` (and ``.../all``)."""
    groups = defaultdict(list)
    for row in rows:
        for subtask in (row['subtask'], 'all'):
            groups[f"{row['length']}/{row['condition']}/{subtask}"].append(row)
    return {key: dict(questions=len(items),
                      score=sum(r['ruler_match'] for r in items)/len(items),
                      all_answers_correct=sum(r['all_answers_correct'] for r in items),
                      mean_tool_calls=sum(r['tool_calls'] for r in items)/len(items),
                      mean_expanded_answer_coverage=sum(r['expanded_answer_coverage'] for r in items)/len(items),
                      stop_reasons=dict(Counter(r['stop_reason'] for r in items)))
            for key, items in sorted(groups.items())}
