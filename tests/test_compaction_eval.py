import json

import pytest

from compaction.eval_math import job_id, read_generations, score, select

TASKS = [dict(task_id=f'algebra/{i:04d}', problem=f'p{i}', answer=str(i), subject='Algebra', level='Level 1')
         for i in range(3)]


def generation(task_id, mode, text, avg, peak, compactions=()):
    return dict(task_id=task_id, mode=mode, text=text, stop_reason='eos', generated_tokens=10,
                avg_context_positions=avg, peak_context_positions=peak, compactions=list(compactions))


def test_select_is_a_fixed_subset():
    assert select(TASKS, 0) == TASKS
    assert select(TASKS, 2) == select(list(reversed(TASKS)), 2)
    assert len(select(TASKS, 2)) == 2


def test_read_generations_skips_partial_line(tmp_path):
    row = generation('algebra/0000', 'full', 'x', 1, 2)
    (tmp_path / 'generations.rank0.jsonl').write_text(json.dumps(row) + '\n{"task_id": "alg')
    assert list(read_generations(tmp_path)) == [job_id('algebra/0000', 'full')]


def test_score_averages_per_problem_statistics(tmp_path):
    pytest.importorskip('math_verify')
    rows = {}
    for i, task in enumerate(TASKS):
        tid = task['task_id']
        rows[job_id(tid, 'full')] = generation(tid, 'full', rf'\boxed{{{i}}}', 100 + i, 200 + i)
        rows[job_id(tid, 'compaction')] = generation(tid, 'compaction', r'\boxed{0}' if i == 0 else 'no answer',
                                                     50 + i, 80 + i, compactions=[{}] * i)
    metrics = score(TASKS, ('full', 'compaction'), rows, tmp_path)
    full, compact = metrics['modes']['full'], metrics['modes']['compaction']
    assert full['accuracy'] == 100 and compact['accuracy'] == pytest.approx(100 / 3)
    assert full['avg_context_positions'] == 101 and full['mean_peak_context_positions'] == 201
    assert full['max_peak_context_positions'] == 202 and compact['max_peak_context_positions'] == 82
    assert compact['compaction_activated'] == 2 and compact['unparsed'] == 2
    assert len((tmp_path / 'scored.jsonl').read_text().splitlines()) == 6
