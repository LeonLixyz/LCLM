"""Native tool-use diagnostic: Qwen tool-call protocol, closed-loop execution, grading.

    python -m agent.tool_eval --output-dir runs/native_tools
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re

from agent.tool_tasks import DEFAULT_SEED, ToolEnvironment, initial_messages


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'duplicate JSON key: {key}')
        result[key] = value
    return result


def strict_json(text):
    def reject(value):
        raise ValueError(f'nonfinite JSON constant: {value}')
    return json.loads(text, object_pairs_hook=_unique_object, parse_constant=reject)


def parse_tool_calls(text):
    """No regex-to-JSON repair, coercion, Python eval or forced tool decoding."""
    if '<think>' in text or '</think>' in text:
        raise ValueError('unexpected_thinking_tags')
    if any(marker in text for marker in ('<|memory_start|>', '<|memory_end|>', '<|memory|>', '<|im_start|>')):
        raise ValueError('unexpected_control_tokens')
    matches = list(re.finditer(r'<tool_call>\s*(.*?)\s*</tool_call>', text, re.S))
    outside = re.sub(r'<tool_call>.*?</tool_call>', '', text, flags=re.S)
    if '<tool_call' in outside or '</tool_call' in outside:
        raise ValueError('malformed_tool_call_delimiters')
    calls = []
    for match in matches:
        value = strict_json(match.group(1))
        if not isinstance(value, dict) or set(value) != {'name', 'arguments'}:
            raise ValueError('invalid_tool_call_envelope')
        if not isinstance(value['name'], str) or not value['name']:
            raise ValueError('invalid_tool_name')
        calls.append(value)
    return calls


def parse_final(text):
    """Extract one terminal FINAL value, independently of its answer wrapper.

    The instruction says *finish* with FINAL, so preceding prose is allowed.
    An unwrapped JSON value can have correct content while failing the requested
    wrapper format. Never search the expected answer or select among candidates.
    """
    text = text.strip()
    markers = list(re.finditer(r'(?m)^\s*FINAL:\s*', text))
    if len(markers) > 1:
        return None, False
    explicit = bool(markers)
    payload = text[markers[0].end():].strip() if explicit else text
    try:
        obj = strict_json(payload)
    except (ValueError, TypeError):
        return None, False
    wrapped = isinstance(obj, dict) and set(obj) == {'answer'}
    return (obj['answer'] if wrapped else obj), explicit and wrapped


def grade(task, mode, environment, final, stop_reason, turns, generations):
    answer, format_ok = parse_final(final)
    answer_correct = answer == task['expected']
    events = environment.events
    succeeded = {e['name'] for e in events if e['success']}
    required_tools = set(task['required_tools']) <= succeeded
    state_correct = True
    if task['subtype'] == 'stateful_reservation':
        state_correct = (len(environment.reservations) == 1
                         and environment.reservations[0]['item_code'] == task['reserve_item']
                         and environment.reservations[0]['quantity'] == task['quantity']
                         and environment.inventory[task['reserve_item']]['stock'] == task['expected']['remaining_stock'])
    restraint_ok = not task['no_tool'] or not events
    expanded = {e['arguments']['segment_id'] for e in events if e['name'] == 'expand' and e['success']}
    support = set(task['support'])
    success = answer_correct and required_tools and state_correct and restraint_ok and stop_reason == 'final'
    return dict(task_id=task['task_id'], family=task['family'], subtype=task['subtype'], mode=mode,
                task_success=bool(success), answer_correct=answer_correct, final_format_correct=format_ok,
                required_tools_used=required_tools, state_correct=state_correct, tool_restraint_correct=restraint_ok,
                stop_reason=stop_reason, turns=turns, tool_calls=len(events),
                schema_valid_calls=sum(e['schema_valid'] for e in events),
                successful_tool_calls=sum(e['success'] for e in events),
                injected_errors=environment.injected_errors,
                expansion_support_hits=len(expanded & support), expansion_support_total=len(support),
                expanded_segments=len(expanded), expansion_precision=(len(expanded & support)/len(expanded) if expanded else None),
                expansion_recall=(len(expanded & support)/len(support) if support else None),
                output_tokens=sum(g['output_tokens'] for g in generations),
                total_prompt_positions=sum(g['prompt_positions'] for g in generations),
                total_uncompressed_prompt_tokens=sum(g['uncompressed_prompt_tokens'] for g in generations),
                final=final, expected=task['expected'])


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False))
    tmp.replace(path)


def run_episode(task, mode, backend, *, max_turns=8, max_calls=12):
    messages, tools = initial_messages(task, mode)
    env = ToolEnvironment(task, tools)
    generations = []
    final = ''
    reason = 'turn_limit'
    for turn in range(max_turns):
        # The backend receives only public conversation and tool schemas.
        generation = backend.generate(messages, tools)
        generations.append(generation)
        text = generation['text']
        if generation['finish_reason'] == 'length':
            messages.append({'role': 'assistant', 'content': text})
            reason = 'output_cap'
            break
        try:
            calls = parse_tool_calls(text)
        except (ValueError, TypeError) as exc:
            messages.append({'role': 'assistant', 'content': text})
            reason = 'protocol_error:' + str(exc)
            break
        if not calls:
            final = text
            messages.append({'role': 'assistant', 'content': text})
            reason = 'final'
            break
        if len(env.events) + len(calls) > max_calls:
            reason = 'tool_call_limit'
            break
        native = [{'id': f'call_{turn}_{index}', 'type': 'function', 'function': call}
                  for index, call in enumerate(calls)]
        content = re.sub(r'<tool_call>.*?</tool_call>', '', text, flags=re.S).strip()
        messages.append({'role': 'assistant', 'content': content, 'tool_calls': native})
        for call in native:
            function = call['function']
            result = env.execute(function['name'], function['arguments'])
            content = (result['text'] if function['name'] == 'expand' and 'text' in result
                       else json.dumps(result, ensure_ascii=False))
            messages.append({'role': 'tool', 'name': function['name'],
                             'tool_call_id': call['id'], 'content': content})
    result = grade(task, mode, env, final, reason, len(generations), generations)
    result.update(messages=messages, tool_events=env.events, generations=generations)
    return result


def aggregate(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row['family'] + '/' + row['mode']].append(row)
        grouped['subtype/' + row['subtype'] + '/' + row['mode']].append(row)
    output = {}
    for name, group in grouped.items():
        calls = sum(r['tool_calls'] for r in group)
        output[name] = dict(episodes=len(group),
                           task_success=sum(r['task_success'] for r in group)/len(group),
                           answer_accuracy=sum(r['answer_correct'] for r in group)/len(group),
                           final_format_accuracy=sum(r['final_format_correct'] for r in group)/len(group),
                           tool_calls=calls,
                           schema_valid_rate=sum(r['schema_valid_calls'] for r in group)/calls if calls else None,
                           execution_success_rate=sum(r['successful_tool_calls'] for r in group)/calls if calls else None,
                           mean_turns=sum(r['turns'] for r in group)/len(group),
                           mean_output_tokens=sum(r['output_tokens'] for r in group)/len(group),
                           total_prompt_positions=sum(r['total_prompt_positions'] for r in group),
                           stop_reasons=dict(Counter(r['stop_reason'] for r in group)))
        if name == 'memory/compressed_expand':
            expanded = sum(r['expanded_segments'] for r in group)
            hits = sum(r['expansion_support_hits'] for r in group)
            output[name].update(expansion_precision=hits/expanded if expanded else None,
                                expansion_recall=hits/sum(r['expansion_support_total'] for r in group),
                                mean_expanded_segments=expanded/len(group))
    # Matched task IDs keep expansion gains separate from native tool skills.
    memory = defaultdict(dict)
    for row in rows:
        if row['family'] == 'memory':
            memory[row['task_id']][row['mode']] = row
    paired = [arms for arms in memory.values() if len(arms) == 3]
    output['paired_memory'] = dict(tasks=len(paired),
        expansion_rescues=sum(not p['compressed']['task_success'] and p['compressed_expand']['task_success'] for p in paired),
        expansion_regressions=sum(p['compressed']['task_success'] and not p['compressed_expand']['task_success'] for p in paired))
    return output


MEMORY_MODES = ('plain', 'compressed', 'compressed_expand')


def modes(task):
    return MEMORY_MODES if task['family'] == 'memory' else ('native',)


def run_task(task, backend):
    rows = []
    for mode in modes(task):
        row = run_episode(task, mode, backend)
        row['episode_id'] = task['task_id'] + '-' + mode
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--checkpoint', default='latent-context/0.6b-4b-LCLM-16x')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--per-family', type=int, default=50)
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED)
    parser.add_argument('--max-tokens', type=int, default=1024)
    parser.add_argument('--max-model-len', type=int, default=32768)
    parser.add_argument('--gpus', type=int, default=1, help='Independent model replicas, one per GPU')
    args = parser.parse_args()
    from agent.backend import resolve_checkpoint, run_jobs
    from agent.tool_tasks import make_tasks
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    tasks = make_tasks(args.per_family, args.seed)
    rows = []
    with (output/'episodes.jsonl').open('w') as stream:
        def save(task_rows):
            for row in task_rows:
                stream.write(json.dumps(row, ensure_ascii=False) + '\n')
                rows.append(row)
            stream.flush()
            print(f'{len(rows)} episodes', flush=True)
        run_jobs(tasks, run_task, dict(checkpoint=resolve_checkpoint(args.checkpoint), max_tokens=args.max_tokens,
                                       max_model_len=args.max_model_len), args.gpus, save)
    metrics = aggregate(rows)
    atomic_json(output/'metrics.json', metrics)
    print(json.dumps({k: v for k, v in metrics.items() if not k.startswith('subtype/')}, indent=2))


if __name__ == '__main__':
    main()
