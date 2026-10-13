"""MATH test-set evaluation of online latent compaction vs. full-context generation.

Runs both modes on the same prompts and output budget, one data-parallel vLLM
worker per GPU, then scores with the last ``\\boxed{}`` answer + math-verify.

    python -m compaction.eval_math --checkpoint latent-context/0.6b-4b-LCLM-16x \\
        --out runs/math_compaction --limit 20 --num-gpus 1

Outputs in ``--out``:
    generations.rank*.jsonl  raw per-trajectory results (resumable)
    scored.jsonl             one row per (problem, mode) with correctness and KV statistics
    metrics.json             accuracy and context-length statistics per mode

Re-running the same command resumes unfinished trajectories, or only rescores
when generation is complete.
"""
import argparse
import hashlib
import json
import multiprocessing as mp
import os
import sys
import traceback
from pathlib import Path

from .controller import BLOCK_SIZE, KEEP_RECENT, MAX_CONTEXT
from .math_answers import correct, last_boxed

DATASET = 'EleutherAI/hendrycks_math'
REVISION = '21a5633873b6a120296cce3e2df9d5550074f4a3'
# (config, rows, sha256 of test-00000-of-00001.parquet) at REVISION.
SPLITS = (
    ('algebra', 1187, '0e04a850dac9cfc758eac0c80afd2d0cc65e29ea72f916c8ac1045b12af937f5'),
    ('counting_and_probability', 474, 'c324818949c75a6227c0328a8499d58023cbbf6757cfc1736f5ab24cc8d87966'),
    ('geometry', 479, '98720f3a8134ad9784fd8951195245264495df03ffc5c9485a9a54ff5ae2431a'),
    ('intermediate_algebra', 903, 'e859f72486c951c2017da880541ca729940eaa986b0364eb23e238983df9bfeb'),
    ('number_theory', 540, 'ba156c9c9fa2040836ba12135fd964e6e67d6bccd8c469ec4a3ad941f8f6fbe9'),
    ('prealgebra', 871, '301e5d8ade5c7b5038032532e498bdc9dca799e7f1061a63f2e7d980803240cc'),
    ('precalculus', 546, 'da37474ba3761ae5727af3844a7ba963a735bd6a14c65e6137593669a4fe3b9c'),
)
INSTRUCTION = 'Solve the problem step by step. Finish with your final answer in \\boxed{}.'
MODES = ('full', 'compaction')


def sha256(text):
    return hashlib.sha256(text.encode()).hexdigest()


def load_tasks():
    """All 5,000 MATH test problems; the reference answer is the last boxed answer of the solution."""
    import pandas as pd
    from huggingface_hub import hf_hub_download
    tasks = []
    for config, rows, digest in SPLITS:
        path = hf_hub_download(DATASET, f'{config}/test-00000-of-00001.parquet', repo_type='dataset', revision=REVISION)
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest, path
        frame = pd.read_parquet(path)
        assert len(frame) == rows
        for i, row in enumerate(frame.itertuples(index=False)):
            tasks.append(dict(task_id=f'{config}/{i:04d}', problem=row.problem, answer=last_boxed(row.solution),
                              subject=row.type, level=row.level))
    assert len(tasks) == 5000 and all(t['answer'] for t in tasks)
    return tasks


def messages(problem):
    return [{'role': 'user', 'content': problem + '\n\n' + INSTRUCTION}]


def select(tasks, limit):
    """Fixed pseudo-random subset (by task-id hash) spanning all subjects."""
    if not limit:
        return tasks
    return sorted(tasks, key=lambda t: sha256(t['task_id']))[:limit]


def job_id(task_id, mode):
    return f'{task_id}|{mode}'


def read_generations(out):
    rows = {}
    for path in sorted(out.glob('generations.rank*.jsonl')):
        for line in path.read_text().splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:  # Partial last line of an interrupted run.
                continue
            rows[job_id(row['task_id'], row['mode'])] = row
    return rows


def worker(rank, jobs, gate, args):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(rank)
    os.environ.setdefault('VLLM_WORKER_MULTIPROC_METHOD', 'spawn')
    out = Path(args.out)
    try:
        from .backend import VLLMBackend
        from .scheduler import serve
        sampling = dict(temperature=args.temperature, top_p=args.top_p, top_k=args.top_k, min_p=args.min_p, seed=args.seed)
        with gate:
            backend = VLLMBackend(args.checkpoint, max_context=args.max_context, max_active=args.max_active,
                                  sampling=sampling, gpu_memory_utilization=args.gpu_memory_utilization)
        with (out / f'generations.rank{rank}.jsonl').open('a') as stream:
            def next_job():
                job = jobs.get()
                if job is None:
                    return None
                task_id, mode, problem = job
                return job_id(task_id, mode), messages(problem), mode == 'compaction'

            def on_complete(jid, result):
                task_id, mode = jid.split('|')
                result.pop('generated_ids')
                stream.write(json.dumps(dict(task_id=task_id, **result), ensure_ascii=False) + '\n')
                stream.flush()
                print(f"[gpu{rank}] {jid} {result['stop_reason']} tokens={result['generated_tokens']} "
                      f"compactions={len(result['compactions'])}", flush=True)

            serve(backend, next_job, on_complete, max_active=args.max_active, block_size=args.block_size,
                  keep_recent=args.keep_recent, max_context=args.max_context)
    except BaseException:
        (out / f'worker{rank}.error').write_text(traceback.format_exc())
        raise


def summarize(rows):
    n = len(rows)
    peaks = [r['peak_context_positions'] for r in rows]
    return dict(n=n, accuracy=100 * sum(r['correct'] for r in rows) / n,
                avg_context_positions=sum(r['avg_context_positions'] for r in rows) / n,
                mean_peak_context_positions=sum(peaks) / n, max_peak_context_positions=max(peaks),
                mean_generated_tokens=sum(r['generated_tokens'] for r in rows) / n,
                output_cap_hits=sum(r['stop_reason'] == 'output_cap' for r in rows),
                unparsed=sum(r['predicted_answer'] is None for r in rows),
                compaction_activated=sum(bool(r['compactions']) for r in rows))


def score(tasks, modes, generations, out):
    by_id = {t['task_id']: t for t in tasks}
    scored = []
    for task in tasks:
        for mode in modes:
            row = dict(generations[job_id(task['task_id'], mode)])
            row.update(subject=task['subject'], level=task['level'], answer=task['answer'],
                       predicted_answer=last_boxed(row['text']))
            row['correct'] = correct(row['answer'], row['predicted_answer'])
            scored.append(row)
    with (out / 'scored.jsonl').open('w') as stream:
        for row in scored:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
    metrics = dict(questions=len(by_id), modes={m: summarize([r for r in scored if r['mode'] == m]) for m in modes},
                   per_subject={s: {m: summarize([r for r in scored if r['mode'] == m and r['subject'] == s])
                                    for m in modes} for s in sorted({t['subject'] for t in tasks})})
    (out / 'metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')
    return metrics


def print_table(metrics):
    print(f"\n{'mode':<12}{'accuracy':>10}{'avg KV':>10}{'mean peak':>11}{'max peak':>10}{'compacted':>11}")
    for mode, g in metrics['modes'].items():
        print(f"{mode:<12}{g['accuracy']:>9.2f}%{g['avg_context_positions']:>10.1f}"
              f"{g['mean_peak_context_positions']:>11.1f}{g['max_peak_context_positions']:>10}"
              f"{g['compaction_activated']:>6}/{g['n']}")


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--checkpoint', default='latent-context/0.6b-4b-LCLM-16x', help='LCLM checkpoint dir or HF repo id')
    ap.add_argument('--out', required=True, help='Output directory')
    ap.add_argument('--limit', type=int, default=0, help='Evaluate a fixed subset of this many problems (0 = all 5,000)')
    ap.add_argument('--modes', default=','.join(MODES), help='Comma-separated subset of: full,compaction')
    ap.add_argument('--num-gpus', type=int, default=None, help='Data-parallel workers, one per GPU (default: all visible)')
    ap.add_argument('--max-active', type=int, default=16, help='Concurrent trajectories per GPU')
    ap.add_argument('--gpu-memory-utilization', type=float, default=0.70, help='vLLM share; the rest holds the encoder')
    ap.add_argument('--block-size', type=int, default=BLOCK_SIZE)
    ap.add_argument('--keep-recent', type=int, default=KEEP_RECENT)
    ap.add_argument('--max-context', type=int, default=MAX_CONTEXT)
    ap.add_argument('--temperature', type=float, default=0.7)
    ap.add_argument('--top-p', type=float, default=0.8)
    ap.add_argument('--top-k', type=int, default=20)
    ap.add_argument('--min-p', type=float, default=0.0)
    ap.add_argument('--seed', type=int, default=0)
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    modes = tuple(args.modes.split(','))
    assert modes and set(modes) <= set(MODES), modes
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    config = {k: v for k, v in vars(args).items() if k not in ('out', 'num_gpus', 'max_active', 'gpu_memory_utilization')}
    config_path = out / 'config.json'
    if config_path.exists():
        assert json.loads(config_path.read_text()) == config, f'{out} holds a run with different settings'
    else:
        config_path.write_text(json.dumps(config, indent=2) + '\n')

    tasks = select(load_tasks(), args.limit)
    done = read_generations(out)
    jobs = [(t['task_id'], m, t['problem']) for t in tasks for m in modes if job_id(t['task_id'], m) not in done]
    # Fixed order mixing long and short jobs; answers never reach the workers.
    jobs.sort(key=lambda j: sha256(job_id(j[0], j[1])))
    print(f'{len(tasks)} problems x {len(modes)} modes; {len(jobs)} trajectories to generate', flush=True)
    if jobs:
        if args.num_gpus is None:
            import torch
            args.num_gpus = torch.cuda.device_count()
        assert args.num_gpus > 0, 'No GPU available'
        ctx = mp.get_context('spawn')
        queue = ctx.Queue()
        gate = ctx.BoundedSemaphore(4)
        for job in jobs:
            queue.put(job)
        for _ in range(args.num_gpus):
            queue.put(None)
        procs = [ctx.Process(target=worker, args=(rank, queue, gate, args)) for rank in range(args.num_gpus)]
        for p in procs:
            p.start()
        for p in procs:
            p.join()
        failed = [rank for rank, p in enumerate(procs) if p.exitcode != 0]
        if failed:
            sys.exit(f'GPU worker(s) {failed} failed; see {out}/worker*.error. Re-run to resume.')
        done = read_generations(out)
    missing = [(t['task_id'], m) for t in tasks for m in modes if job_id(t['task_id'], m) not in done]
    if missing:
        sys.exit(f'{len(missing)} trajectories missing; re-run to resume.')
    metrics = score(tasks, modes, done, out)
    print_table(metrics)
    print(f'\nWrote {out}/scored.jsonl and {out}/metrics.json')
    return metrics


if __name__ == '__main__':
    main()
