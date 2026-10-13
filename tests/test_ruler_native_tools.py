"""CPU tests of segmented RULER NIAH with native expand calls; scripted backends, no model."""
import copy
import json
from pathlib import Path
import unittest

from agent.ruler_tools import (END, FINAL_NOTICE, MAX_CALLS, SEED, START, SUBTASKS, agent_messages,
                               aggregate, execute_expand, extract_prompt, initial_messages, prepare_task,
                               ruler_match, run_agent, run_episode, sample_rank, segment_text, select_rows)

PANEL = Path(__file__).resolve().parent.parent/'agent'/'figure6_questions.json'


class CharacterTokenizer:
    def encode(self, text, **kwargs):
        return list(text)

    def __call__(self, text, **kwargs):
        return dict(input_ids=list(text), offset_mapping=[(i, i+1) for i in range(len(text))])


def row(question_inside=False, subtask='niah_single_1', index=17, length=16384):
    question = '\nWhat is the special magic number for zebra mentioned in the provided text?'
    body = '\nThe special magic number for zebra is: 9876543.\n'
    prompt = ('<|im_start|>system\nBe precise.<|im_end|>\n<|im_start|>user\nRead the text.\n'
              + START + body + (question + END if question_inside else END + question)
              + '<|im_end|>\n<|im_start|>assistant\n')
    return dict(prompt=prompt, extra_info=dict(ruler_task=subtask,
        scoring_function='ruler_string_match_all', context_length=length,
        ground_truth=dict(answers=['9876543'], index=index)))


def task(chunk_tokens=24, **kwargs):
    return prepare_task(row(**kwargs), CharacterTokenizer(), chunk_tokens=chunk_tokens)


class Backend:
    def __init__(self, outputs):
        self.outputs = iter(outputs)
        self.received = []

    def generate(self, messages, tools):
        self.received.append((copy.deepcopy(messages), copy.deepcopy(tools)))
        output = next(self.outputs)
        if isinstance(output, tuple):
            text, reason = output
        else:
            text, reason = output, 'stop'
        return dict(text=text, finish_reason=reason, output_tokens=10, prompt_positions=100,
                    uncompressed_prompt_tokens=1000)


def expand(*segment_ids, **extra):
    return ''.join('<tool_call>' + json.dumps(dict(name='expand', arguments=dict(segment_id=s, **extra)))
                   + '</tool_call>' for s in (segment_ids or ('seg_2',)))


class RulerToolsTests(unittest.TestCase):
    def test_both_source_question_locations_preserve_text(self):
        for inside in (False, True):
            parts = extract_prompt(row(inside)['prompt'])
            self.assertEqual(parts['question_moved_out_of_memory'], inside)
            self.assertNotIn('What is', parts['body'])
            self.assertIn('What is', parts['question'])
            self.assertIn('9876543', parts['body'])
        with self.assertRaises(ValueError):
            extract_prompt(row()['prompt'] + 'extra')

    def test_unknown_wrapped_question_rejected(self):
        with self.assertRaises(ValueError):
            extract_prompt(row(True)['prompt'].replace('What is', 'Why is'))

    def test_lossless_unicode_and_newline_segments(self):
        text = 'é中🙂\nabcdefghij\n' * 7
        segments = segment_text(text, CharacterTokenizer(), 9)
        self.assertEqual(''.join(s['text'] for s in segments), text)
        self.assertTrue(all(0 < s['encoder_tokens'] <= 9 for s in segments))
        for s in segments:
            self.assertEqual(text[s['char_start']:s['char_end']], s['text'])

    def test_matched_input_and_no_private_metadata(self):
        t = task()
        for mode in ('plain', 'compressed', 'compressed_expand'):
            messages, tools = initial_messages(t, mode)
            self.assertEqual(set(messages[1]), {'role', 'content'})
            content = messages[1]['content']
            self.assertTrue(content.endswith(t['question']))
            self.assertEqual(content.count(START), 0 if mode == 'plain' else len(t['segments']))
            for s in t['segments']:
                self.assertIn(s['text'], content)
            self.assertNotIn('9876543', t['question'] + messages[0]['content'])
            self.assertEqual(bool(tools), mode == 'compressed_expand')

    def test_exact_expansion_and_schema_validation(self):
        segments = {'seg_1': '  exact\né中🙂\n'}
        good = dict(name='expand', arguments={'segment_id': 'seg_1'})
        self.assertEqual(execute_expand(good, segments, True), (segments['seg_1'], True, True))
        for args in ({'segment_id': 1}, {'segment_id': 'seg_1', 'extra': 1}, {'segment_id': 'seg_0'}):
            self.assertEqual(execute_expand(dict(name='expand', arguments=args), segments, True)[1:], (False, False))
        self.assertEqual(execute_expand(good, segments, False)[1:], (False, False))
        self.assertEqual(execute_expand(dict(name='expand', arguments={'segment_id': 'seg_2'}), segments, True)[1:], (True, False))

    def test_score_final_only_not_tool_evidence(self):
        t = task(chunk_tokens=200)
        backend = Backend([expand('seg_1'), '1111111'])
        result = run_episode(t, 'compressed_expand', backend)
        self.assertEqual(result['ruler_match'], 0)
        self.assertEqual(result['expanded_answer_coverage'], 1)
        self.assertEqual(backend.received[1][0][-1]['content'], t['segments'][0]['text'])

    def test_partial_answer_coverage_and_case(self):
        self.assertEqual(ruler_match('AbCd', ['abcd', 'xyz']), 50)
        self.assertEqual(ruler_match('abcd, XYZ', ['abcd', 'xyz']), 100)

    def test_output_cap_and_protocol_errors_fail(self):
        for output, expected in ((('9876543', 'length'), 'output_cap'),
                                 ('<tool_call>{bad}</tool_call>', 'protocol_error:'),
                                 ('<think>9876543</think>', 'protocol_error:')):
            for result in (run_episode(task(), 'compressed_expand', Backend([output])),
                           run_agent(task(), Backend([output]), 16)):
                self.assertEqual(result['ruler_match'], 0)
                self.assertTrue(result['stop_reason'].startswith(expected))

    def test_call_and_turn_budgets(self):
        result = run_episode(task(), 'compressed_expand', Backend([expand()] * 3), max_calls=2)
        self.assertEqual(result['stop_reason'], 'tool_call_limit')
        result = run_episode(task(), 'compressed_expand', Backend([expand()] * 2), max_turns=2)
        self.assertEqual(result['stop_reason'], 'turn_limit')
        result = run_episode(task(), 'compressed', Backend([expand()]))
        self.assertEqual(result['stop_reason'], 'tool_unavailable')


class AgentTests(unittest.TestCase):
    def test_compressed_baseline_has_no_tools(self):
        backend = Backend(['9876543'])
        result = run_episode(task(), 'compressed', backend, max_turns=9, max_calls=8)
        self.assertEqual(backend.received[0][1], [])
        self.assertNotIn('expand', backend.received[0][0][0]['content'])
        self.assertEqual(result['ruler_match'], 100)

    def test_agent_prompt_states_the_budget(self):
        for budget, word in ((16, 'sixteenth'), (32, 'thirty-second')):
            messages, tools = agent_messages(task(), budget)
            system = messages[0]['content']
            self.assertEqual(system.count(f'at most {budget} calls'), 2)
            self.assertIn(f'After the {word} call', system)
            self.assertTrue(system.endswith(f'Expansion budget remaining: {budget} of {budget}. '
                                            'You may answer now without expanding.'))
            self.assertEqual([t['function']['name'] for t in tools], ['expand'])

    def test_budget_exhaustion_forces_tool_free_final_answer(self):
        for budget in (16, 32):
            backend = Backend([expand('seg_1')] * budget + ['9876543'])
            result = run_agent(task(), backend, budget)
            self.assertEqual((result['tool_calls'], result['ruler_match']), (budget, 100))
            self.assertEqual(result['finalization_trigger'], 'budget_exhausted')
            messages, tools = backend.received[-1]
            self.assertEqual(tools, [])
            self.assertEqual(messages[-1], dict(role='user', content=FINAL_NOTICE))
            # Remaining-budget reminders follow every expansion except the last.
            reminders = [m for m in messages if m['role'] == 'user' and m['content'].startswith('Expansion budget remaining')]
            self.assertEqual(len(reminders), budget - 1)
            self.assertTrue(reminders[0]['content'].startswith(f'Expansion budget remaining: {budget-1} of {budget}.'))

    def test_tool_call_after_budget_scores_zero(self):
        result = run_agent(task(), Backend([expand('seg_1')] * 16 + [expand('seg_2')]), 16)
        self.assertEqual((result['stop_reason'], result['ruler_match']), ('tool_unavailable', 0))

    def test_over_budget_batch_is_not_executed(self):
        backend = Backend([expand('seg_1')] * 15 + [expand('seg_1', 'seg_2'), '9876543'])
        result = run_agent(task(), backend, 16)
        self.assertEqual(result['tool_calls'], 15)
        self.assertEqual(result['finalization_trigger'], 'over_budget_batch_rejected')
        self.assertEqual(result['ruler_match'], 100)

    def test_final_only_scoring_for_agent(self):
        t = task(chunk_tokens=200)
        result = run_agent(t, Backend([expand('seg_1'), 'not sure']), 16)
        self.assertEqual((result['ruler_match'], result['expanded_answer_coverage']), (0, 1))

    def test_paper_caps(self):
        self.assertEqual(MAX_CALLS, {4096: 16, 8192: 16, 16384: 32, 32768: 16})


class SelectionTests(unittest.TestCase):
    def rows(self, length):
        return [row(subtask=t, index=i, length=length) for t in SUBTASKS for i in range(500)]

    def test_hash_selection_ignores_answers_and_matches_across_lengths(self):
        panels = []
        for length in MAX_CALLS:
            rows = self.rows(length)
            selected = [(r['extra_info']['ruler_task'], r['extra_info']['ground_truth']['index'])
                        for r in select_rows(rows, 50)]
            for r in rows:
                r['extra_info']['ground_truth']['answers'] = ['changed']
            again = select_rows(list(reversed(rows)), 50)
            self.assertEqual(sorted(selected), sorted((r['extra_info']['ruler_task'],
                                                       r['extra_info']['ground_truth']['index']) for r in again))
            self.assertEqual(len(selected), 400)
            panels.append(sorted(selected))
        self.assertTrue(all(p == panels[0] for p in panels))
        a, b = row(), row(length=32768)
        b['extra_info']['ground_truth']['answers'] = ['changed']
        self.assertEqual(sample_rank(a), sample_rank(b))

    def test_figure6_panel(self):
        panel = json.loads(PANEL.read_text())
        self.assertEqual(sorted(panel, key=int), [str(n) for n in MAX_CALLS])
        for length, tasks in panel.items():
            self.assertEqual(set(tasks), set(SUBTASKS))
            for subtask, indices in tasks.items():
                self.assertEqual(len(set(indices)), 50)
                self.assertTrue(all(0 <= i < 500 for i in indices))
                # Five task types are exactly the seeded hash sample, in rank order.
                if subtask not in ('niah_multikey_2', 'niah_multivalue', 'niah_multiquery'):
                    expected = [(r['extra_info']['ground_truth']['index'])
                                for r in select_rows([row(subtask=subtask, index=i) for i in range(500)], 50, SEED)]
                    self.assertEqual(indices, expected)
        rows = self.rows(4096)
        chosen = select_rows(rows, 3, indices=panel['4096'])
        self.assertEqual(len(chosen), 24)
        self.assertEqual([r['extra_info']['ground_truth']['index'] for r in chosen[:3]], panel['4096'][SUBTASKS[0]][:3])


class AggregateTests(unittest.TestCase):
    def test_per_length_condition_means(self):
        rows = []
        for subtask, index in (('niah_single_1', 1), ('niah_multikey_3', 2)):
            t = task(subtask=subtask, index=index, length=4096)
            base = run_episode(t, 'compressed', Backend(['1111111' if index == 2 else '9876543']))
            agent = run_agent(t, Backend([expand('seg_1'), '9876543']), 16)
            base['condition'], agent['condition'] = 'compressed', 'agent'
            rows += [base, agent]
        summary = aggregate(rows)
        self.assertEqual(summary['4096/compressed/all']['score'], 50)
        self.assertEqual(summary['4096/agent/all']['score'], 100)
        self.assertEqual(summary['4096/agent/all']['mean_tool_calls'], 1)
        self.assertEqual(summary['4096/compressed/niah_multikey_3']['questions'], 1)


if __name__ == '__main__':
    unittest.main()
