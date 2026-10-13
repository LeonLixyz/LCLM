"""CPU test of the full Figure 6 pipeline: selection, both conditions, resume and aggregation."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

import agent.backend
from agent import aggregate_niah, run_niah_agent
from test_ruler_native_tools import CharacterTokenizer, expand, row


class ScriptedModel:
    """Expands seg_1 when the tool is offered, then answers from what it has read."""
    instances = 0

    def __init__(self, **kwargs):
        ScriptedModel.instances += 1
        self.cache = {}

    def generate(self, messages, tools):
        read = [m['content'] for m in messages if m['role'] == 'tool']
        if tools and not read:
            text = expand('seg_1')
        else:
            text = '9876543' if any('9876543' in r for r in read) else 'unknown'
        return dict(text=text, finish_reason='stop', output_tokens=1, prompt_positions=1,
                    uncompressed_prompt_tokens=1)


def fake_rows(length, subtasks):
    return [row(subtask=t, index=i, length=length) for t in subtasks for i in range(500)]


class FullPipelineTests(unittest.TestCase):
    def run_main(self, *argv):
        tokenizers = types.SimpleNamespace(AutoTokenizer=types.SimpleNamespace(
            from_pretrained=lambda path: CharacterTokenizer()))
        with mock.patch.object(run_niah_agent, 'load_rows', fake_rows), \
             mock.patch.object(agent.backend, 'resolve_checkpoint', lambda c: '/checkpoint'), \
             mock.patch.object(agent.backend, 'Backend', ScriptedModel), \
             mock.patch.dict(sys.modules, {'transformers': tokenizers}), \
             mock.patch.object(sys, 'argv', ['run_niah_agent', *argv]), \
             contextlib.redirect_stdout(io.StringIO()):
            run_niah_agent.main()

    def test_end_to_end_resume_and_aggregate(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)/'4096'
            args = ('--context-length', '4096', '--subtasks', 'niah_single_1,niah_multikey_2',
                    '--n-questions', '3', '--segment-tokens', '200', '--output-dir', str(out))
            ScriptedModel.instances = 0
            self.run_main(*args)
            rows = [json.loads(line) for line in (out/'episodes.jsonl').read_text().splitlines()]
            self.assertEqual(len(rows), 12)
            self.assertEqual({r['condition'] for r in rows}, {'compressed', 'agent'})
            self.assertFalse(any('messages' in r for r in rows))
            self.assertFalse((out/'transcripts.jsonl').exists())
            panel = json.loads(run_niah_agent.PANEL.read_text())['4096']
            self.assertEqual(sorted({r['source_index'] for r in rows if r['subtask'] == 'niah_multikey_2'}),
                             sorted(panel['niah_multikey_2'][:3]))
            summary = json.loads((out/'summary.json').read_text())
            self.assertEqual(summary['4096/compressed/all']['score'], 0)
            self.assertEqual(summary['4096/agent/all']['score'], 100)
            self.assertEqual(json.loads((out/'config.json').read_text())['max_calls'], 16)

            # A finished run resumes without loading a model or duplicating episodes.
            self.run_main(*args)
            self.assertEqual(ScriptedModel.instances, 1)
            self.assertEqual(len((out/'episodes.jsonl').read_text().splitlines()), 12)
            with self.assertRaises(ValueError):
                self.run_main(*args[:-4], '--n-questions', '2', '--output-dir', str(out))

            buffer = io.StringIO()
            with mock.patch.object(sys, 'argv', ['aggregate_niah', tmp, '--per-task']), \
                 contextlib.redirect_stdout(buffer):
                aggregate_niah.main()
            table = buffer.getvalue()
            self.assertIn('    4096         6        0.00   100.00   1.00   80.94 / 99.44', table)
            self.assertIn('niah_multikey_2', table)

    def test_hash_selection_and_transcripts(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)/'run'
            self.run_main('--context-length', '16384', '--subtasks', 'niah_single_3', '--n-questions', '2',
                          '--selection', 'hash', '--segment-tokens', '200', '--save-transcripts',
                          '--output-dir', str(out))
            transcripts = [json.loads(line) for line in (out/'transcripts.jsonl').read_text().splitlines()]
            self.assertEqual(len(transcripts), 4)
            agent_rows = [t for t in transcripts if t['condition'] == 'agent']
            self.assertTrue(all(t['max_calls'] == 32 and t['events'] for t in agent_rows))
            self.assertEqual(agent_rows[0]['messages'][2]['tool_calls'][0]['function']['arguments'],
                             {'segment_id': 'seg_1'})

    def test_figure6_panel_rejects_unknown_length(self):
        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(ValueError):
            self.run_main('--context-length', '65536', '--max-calls', '16', '--output-dir', tmp)


if __name__ == '__main__':
    unittest.main()
