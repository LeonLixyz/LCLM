"""RULER NIAH with latent context: compressed-only vs. the expand agent (paper Figure 6).

Every question runs twice on the same compressed segments: once without tools
(``compressed``) and once with the native ``expand`` tool (``agent``).

    # smoke test: 2 questions of one subtask at 4k
    python -m agent.run_niah_agent --context-length 4096 --subtasks niah_single_3 \\
        --n-questions 2 --output-dir runs/niah_agent_smoke/4096
    # one Figure 6 bar pair (400 questions, 800 episodes)
    python -m agent.run_niah_agent --context-length 16384 --output-dir runs/niah_agent/16384
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from agent.ruler_tools import (MAX_CALLS, SEED, SEGMENT_TOKENS, SUBTASKS, aggregate, prepare_task,
                               run_agent, run_episode, select_rows)

DEFAULT_CHECKPOINT = 'latent-context/0.6b-4b-LCLM-16x'
DATASET = ('latent-context/lclm-eval', 'ruler', 'test')
PANEL = Path(__file__).with_name('figure6_questions.json')
CONDITIONS = ('compressed', 'agent')
TRANSCRIPT_KEYS = ('messages', 'generations', 'events')


def load_rows(context_length, subtasks):
    from datasets import load_dataset
    name, config, split = DATASET
    categories = {f'memwrap/ruler/{t}_{context_length}' for t in subtasks}
    rows = []
    for row in load_dataset(name, config, split=split).filter(lambda r: r['category'] in categories):
        row = dict(row)
        if isinstance(row['extra_info'], str):
            row['extra_info'] = json.loads(row['extra_info'])
        rows.append(row)
    found = {r['extra_info']['ruler_task'] for r in rows}
    if found != set(subtasks):
        raise ValueError(f'Missing subtasks at {context_length}: {sorted(set(subtasks) - found)}')
    return rows


def run_question(job, backend):
    """Both conditions for one question; the job pool calls this on each replica."""
    task, max_calls = job
    rows = []
    for condition in CONDITIONS:
        if condition == 'compressed':
            row = run_episode(task, 'compressed', backend, max_turns=9, max_calls=8)
        else:
            row = run_agent(task, backend, max_calls)
        row.update(condition=condition, episode_id=f"{task['task_id']}-{condition}")
        rows.append(row)
    return rows


def print_summary(metrics, length):
    print(f'{"subtask":<18} {"n":>4} {"compressed":>11} {"agent":>8} {"calls":>6}')
    for subtask in (*SUBTASKS, 'all'):
        base, agent = (metrics.get(f'{length}/{c}/{subtask}') for c in CONDITIONS)
        if base and agent:
            print(f'{subtask:<18} {base["questions"]:>4} {base["score"]:>11.2f} {agent["score"]:>8.2f} '
                  f'{agent["mean_tool_calls"]:>6.2f}')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--checkpoint', default=DEFAULT_CHECKPOINT, help='Local directory or Hugging Face repo id')
    parser.add_argument('--context-length', type=int, required=True, help='4096, 8192, 16384 or 32768')
    parser.add_argument('--subtasks', default=','.join(SUBTASKS))
    parser.add_argument('--n-questions', type=int, default=50, help='Questions per subtask')
    parser.add_argument('--max-calls', type=int, help='Expansion-call cap (default: paper value for the length)')
    parser.add_argument('--selection', choices=('figure6', 'hash'), default='figure6',
                        help='figure6: the exact questions behind Figure 6; hash: rank by --seed')
    parser.add_argument('--seed', type=int, default=SEED)
    parser.add_argument('--segment-tokens', type=int, default=SEGMENT_TOKENS)
    parser.add_argument('--max-tokens', type=int, default=512, help='Generated tokens per assistant turn')
    parser.add_argument('--max-model-len', type=int, default=65536)
    parser.add_argument('--gpu-memory-utilization', type=float, default=0.65)
    parser.add_argument('--gpus', type=int, default=1, help='Independent model replicas, one per GPU')
    parser.add_argument('--save-transcripts', action='store_true', help='Also write full message histories')
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()

    from transformers import AutoTokenizer
    from agent.backend import resolve_checkpoint
    from agent.tool_eval import atomic_json

    length = args.context_length
    subtasks = [t for t in args.subtasks.split(',') if t]
    assert set(subtasks) <= set(SUBTASKS), subtasks
    max_calls = args.max_calls or MAX_CALLS[length]
    if args.selection == 'figure6':
        panel = json.loads(PANEL.read_text())
        if str(length) not in panel or args.n_questions > 50:
            raise ValueError('The Figure 6 panel has 50 questions per subtask at 4096/8192/16384/32768; use --selection hash')
        indices = panel[str(length)]
    else:
        indices = None

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    config = dict(checkpoint=args.checkpoint, context_length=length, subtasks=subtasks,
                  n_questions=args.n_questions, max_calls=max_calls, selection=args.selection,
                  seed=args.seed, segment_tokens=args.segment_tokens, max_tokens=args.max_tokens,
                  max_model_len=args.max_model_len)
    if (output/'config.json').exists() and json.loads((output/'config.json').read_text()) != config:
        raise ValueError(f'{output} holds a run with a different configuration')
    atomic_json(output/'config.json', config)

    checkpoint = resolve_checkpoint(args.checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(str(Path(checkpoint)/'encoder'))
    rows = select_rows(load_rows(length, subtasks), args.n_questions, args.seed, indices)
    tasks = [prepare_task(row, tokenizer, args.segment_tokens) for row in rows]
    print(f'{len(tasks)} questions, {sum(len(t["segments"]) for t in tasks)} segments', flush=True)

    episodes = output/'episodes.jsonl'
    saved = [json.loads(line) for line in episodes.read_text().splitlines()] if episodes.exists() else []
    done = {r['task_id'] for r in saved}
    todo = [(t, max_calls) for t in tasks if t['task_id'] not in done]
    print(f'{len(done)} questions already done, {len(todo)} to run', flush=True)

    with episodes.open('a') as stream, (output/'transcripts.jsonl').open('a') as transcripts:
        def save(question_rows):
            for row in question_rows:
                summary = {k: v for k, v in row.items() if k not in TRANSCRIPT_KEYS}
                stream.write(json.dumps(summary, ensure_ascii=False) + '\n')
                saved.append(summary)
                if args.save_transcripts:
                    transcripts.write(json.dumps(row, ensure_ascii=False) + '\n')
            stream.flush()
            transcripts.flush()
            print(f'[{len(saved)//2}/{len(tasks)}] {question_rows[0]["task_id"]}: '
                  + ', '.join(f'{r["condition"]}={r["ruler_match"]:.0f} ({r["stop_reason"]})' for r in question_rows),
                  flush=True)
        from agent.backend import run_jobs
        run_jobs(todo, run_question, dict(checkpoint=checkpoint, max_tokens=args.max_tokens,
                                          max_model_len=args.max_model_len,
                                          gpu_memory_utilization=args.gpu_memory_utilization),
                 args.gpus, save)
    if not args.save_transcripts and not (output/'transcripts.jsonl').stat().st_size:
        (output/'transcripts.jsonl').unlink()

    metrics = aggregate(saved)
    atomic_json(output/'summary.json', metrics)
    print_summary(metrics, length)


if __name__ == '__main__':
    main()
