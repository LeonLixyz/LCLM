"""CPU tests of the native tool-use diagnostic: tool execution and grading, no model."""
import json
import unittest

from agent.tool_eval import aggregate, parse_final, parse_tool_calls, run_episode
from agent.tool_tasks import ToolEnvironment, fingerprint, initial_messages, make_tasks


class ScriptedBackend:
    def __init__(self, responses):
        self.responses = iter(responses)

    def generate(self, messages, tools):
        return dict(text=next(self.responses), finish_reason='stop', output_tokens=1,
                    prompt_positions=1, uncompressed_prompt_tokens=1)


def reference_backend(task, mode):
    calls = task['reference_calls'] if task['family'] != 'memory' or mode == 'compressed_expand' else []
    responses = ['<tool_call>' + json.dumps(call) + '</tool_call>' for call in calls]
    return ScriptedBackend(responses + ['FINAL: ' + json.dumps({'answer': task['expected']})])


class NativeToolEvalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tasks = make_tasks(50)

    def test_all_reference_trajectories_execute_and_pass(self):
        rows = []
        for task in self.tasks:
            modes = ('plain', 'compressed', 'compressed_expand') if task['family'] == 'memory' else ('native',)
            for mode in modes:
                row = run_episode(task, mode, reference_backend(task, mode))
                self.assertTrue(row['task_success'], (task['task_id'], mode, row))
                rows.append(row)
        self.assertEqual(len(rows), 300)
        self.assertEqual(aggregate(rows)['paired_memory']['tasks'], 50)

    def test_generator_is_stable_when_sample_count_changes(self):
        small = {task['task_id']: task for task in make_tasks(2)}
        for task in self.tasks:
            if task['task_id'] in small:
                self.assertEqual(fingerprint(task), fingerprint(small[task['task_id']]))

    def test_final_answer_without_real_reservation_fails(self):
        task = next(t for t in self.tasks if t['subtype'] == 'stateful_reservation')
        row = run_episode(task, 'native', ScriptedBackend(['FINAL: ' + json.dumps({'answer': task['expected']})]))
        self.assertTrue(row['answer_correct'])
        self.assertFalse(row['state_correct'])
        self.assertFalse(row['task_success'])

    def test_valid_json_is_not_enough_for_schema_validation(self):
        task = self.tasks[1]
        _, tools = initial_messages(task, 'native')
        env = ToolEnvironment(task, tools)
        self.assertIn('error', env.execute('multiply', {'a': True, 'b': 2}))
        self.assertIn('error', env.execute('multiply', {'a': '3', 'b': 2}))
        self.assertIn('error', env.execute('multiply', {'a': 3, 'b': 2, 'extra': 1}))
        self.assertIn('error', env.execute('undeclared_tool', {}))
        self.assertEqual(sum(e['schema_valid'] for e in env.events), 0)

    def test_bad_protocol_is_rejected_without_execution(self):
        for text in ('<tool_call>{}</tool_call>', '<tool_call>{}',
                     '<tool_call>{"name":"a","name":"b","arguments":{}}</tool_call>',
                     '<tool_call>__import__("os").system("anything")</tool_call>',
                     '<think><tool_call>{"name":"a","arguments":{}}</tool_call></think>',
                     '<tool_call>{"name":"a","arguments":{"x":NaN}}</tool_call>'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_tool_calls(text)

    def test_multiple_calls_in_one_turn_and_tool_ids(self):
        task = next(t for t in self.tasks if t['subtype'] == 'multiple_calls')
        text = '\n'.join('<tool_call>'+json.dumps(c)+'</tool_call>' for c in task['reference_calls'])
        row = run_episode(task, 'native', ScriptedBackend([text, 'FINAL: '+json.dumps({'answer': task['expected']})]))
        self.assertTrue(row['task_success'])
        assistant = next(m for m in row['messages'] if m.get('tool_calls'))
        observations = [m for m in row['messages'] if m['role'] == 'tool']
        self.assertEqual([c['id'] for c in assistant['tool_calls']], [m['tool_call_id'] for m in observations])

    def test_transient_error_is_observed_and_recovered(self):
        task = next(t for t in self.tasks if t['family'] == 'recovery')
        row = run_episode(task, 'native', reference_backend(task, 'native'))
        self.assertTrue(row['task_success'])
        self.assertEqual(row['injected_errors'], 1)
        self.assertFalse(row['tool_events'][0]['success'])
        self.assertTrue(row['tool_events'][1]['success'])

    def test_expand_returns_exact_original_and_only_memory_body_is_wrapped(self):
        task = next(t for t in self.tasks if t['family'] == 'memory')
        messages, tools = initial_messages(task, 'compressed_expand')
        self.assertEqual(messages[1]['content'].count('<|memory_start|>'), len(task['segments']))
        self.assertTrue(messages[1]['content'].endswith(task['question']))
        self.assertNotIn('<|memory_start|>', task['question'])
        row = run_episode(task, 'compressed_expand', reference_backend(task, 'compressed_expand'))
        observation = next(m for m in row['messages'] if m['role'] == 'tool')
        self.assertEqual(observation['content'], task['segments'][task['support'][0]])
        self.assertEqual(row['expansion_recall'], 1)
        self.assertEqual(row['expansion_precision'], 1)

    def test_no_tool_control_penalizes_unnecessary_calls(self):
        task = next(t for t in self.tasks if t['no_tool'])
        backend = ScriptedBackend(['<tool_call>{"name":"multiply","arguments":{"a":1,"b":2}}</tool_call>',
                                   'FINAL: '+json.dumps({'answer': task['expected']})])
        row = run_episode(task, 'native', backend)
        self.assertTrue(row['answer_correct'])
        self.assertFalse(row['task_success'])

    def test_final_can_follow_explanatory_text(self):
        self.assertEqual(parse_final('Evidence is in segment two.\n\nFINAL: {"answer": "KEY-123"}'), ('KEY-123', True))
        self.assertEqual(parse_final('{"answer": "KEY-123"}'), ('KEY-123', False))

    def test_unwrapped_result_has_content_but_wrong_format(self):
        value = {'receipt': 'RSV-123', 'remaining_stock': 4}
        self.assertEqual(parse_final('FINAL: '+json.dumps(value)), (value, False))
        task = next(t for t in self.tasks if t['subtype'] == 'stateful_reservation')
        calls = ['<tool_call>'+json.dumps(c)+'</tool_call>' for c in task['reference_calls']]
        row = run_episode(task, 'native', ScriptedBackend(calls+['FINAL: '+json.dumps(task['expected'])]))
        self.assertTrue(row['task_success'])
        self.assertTrue(row['state_correct'])
        self.assertFalse(row['final_format_correct'])

    def test_ambiguous_or_malformed_answers_are_not_salvaged(self):
        for text in ('FINAL: {"answer":"a"}\nFINAL: {"answer":"b"}',
                     'FINAL: {"answer":"a"} trailing prose',
                     'The answer could be {"answer":"a"}',
                     'FINAL: {"answer":"a","answer":"b"}',
                     'FINAL: {"answer": NaN}'):
            with self.subTest(text=text):
                self.assertEqual(parse_final(text), (None, False))


if __name__ == '__main__':
    unittest.main()
