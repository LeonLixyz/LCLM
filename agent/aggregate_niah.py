"""Per-length NIAH averages, compressed-only vs. with expansion, next to the paper's Figure 6.

    python -m agent.aggregate_niah runs/niah_agent/4096 runs/niah_agent/8192 runs/niah_agent/16384
    python -m agent.aggregate_niah runs/niah_agent --per-task
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from agent.ruler_tools import SUBTASKS, aggregate

# Figure 6: average over the 8 NIAH subtasks, 50 questions each (LCLM 0.6b-4b 16x).
PAPER = {4096: (80.94, 99.44), 8192: (75.88, 99.06), 16384: (69.56, 95.06)}


def load(paths):
    files = []
    for path in map(Path, paths):
        files.extend([path] if path.is_file() else sorted(path.rglob('episodes.jsonl')))
    rows = [json.loads(line) for f in files for line in f.read_text().splitlines() if line.strip()]
    if not rows:
        raise SystemExit('No episodes.jsonl found')
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('paths', nargs='+', help='Run directories or episodes.jsonl files')
    parser.add_argument('--per-task', action='store_true')
    args = parser.parse_args()
    rows = load(args.paths)
    metrics = aggregate(rows)
    lengths = sorted({r['length'] for r in rows})
    print(f'{"context":>8} {"questions":>9} {"compressed":>11} {"agent":>8} {"calls":>6}   paper (compressed / agent)')
    for length in lengths:
        base, agent = metrics.get(f'{length}/compressed/all'), metrics.get(f'{length}/agent/all')
        if not (base and agent):
            continue
        paper = '   {:.2f} / {:.2f}'.format(*PAPER[length]) if length in PAPER else ''
        print(f'{length:>8} {base["questions"]:>9} {base["score"]:>11.2f} {agent["score"]:>8.2f} '
              f'{agent["mean_tool_calls"]:>6.2f}{paper}')
    if args.per_task:
        print()
        print(f'{"subtask":<18}' + ''.join(f'{length:>16}' for length in lengths))
        for subtask in SUBTASKS:
            cells = []
            for length in lengths:
                base = metrics.get(f'{length}/compressed/{subtask}')
                agent = metrics.get(f'{length}/agent/{subtask}')
                cells.append(f'{base["score"]:6.1f} -> {agent["score"]:5.1f}' if base and agent else '-')
            print(f'{subtask:<18}' + ''.join(f'{c:>16}' for c in cells))


if __name__ == '__main__':
    main()
