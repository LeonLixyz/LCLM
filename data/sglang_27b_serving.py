"""Pinned BF16 SGLang replicas, matching the bounded TP1 × 8 serving smoke."""
from __future__ import annotations

import json
import os
import signal
import subprocess
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from importlib.metadata import version
from pathlib import Path

from data.sglang_backlog import MODEL, REVISION, IMAGE, atomic_json
from data.sglang_throughput_smoke import graph_capture_lines


def server_command(index):
    if type(index) is not int or not 0 <= index < 8:
        raise ValueError('Expected replica index 0..7')
    return ['python3', '-m', 'sglang.launch_server', '--model-path', MODEL,
        '--revision', REVISION, '--served-model-name', MODEL, '--host', '127.0.0.1',
        '--port', str(8000 + index), '--tp', '1', '--dtype', 'bfloat16',
        '--context-length', '32768', '--mem-fraction-static', '0.80',
        '--max-running-requests', '16', '--cuda-graph-max-bs', '16',
        '--max-mamba-cache-size', '80', '--chunked-prefill-size', '32768',
        '--tool-call-parser', 'qwen3_coder', '--reasoning-parser', 'qwen3']


class Replicas:
    def __init__(self, root, commit):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.commit = commit
        self.processes, self.clients = [None] * 8, [None] * 8

    def start(self):
        from openai import OpenAI
        from transformers import AutoTokenizer
        from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest, ChatCompletionResponseChoice
        if ('return_token_ids' not in ChatCompletionRequest.model_fields
                or 'response_token_ids' not in ChatCompletionResponseChoice.model_fields):
            raise RuntimeError('Pinned runtime does not support pre-parser token capture')
        atomic_json(self.root / 'runtime.json', {'image': IMAGE,
            'packages': {p: version(p) for p in ('sglang', 'transformers', 'torch', 'openai')}})
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
        start = time.monotonic()
        def launch(index):
            command = server_command(index)
            atomic_json(self.root / f'server-{index}.command.json',
                        {'command': command, 'CUDA_VISIBLE_DEVICES': str(index)})
            with (self.root / f'server-{index}.log').open('w') as handle:
                self.processes[index] = subprocess.Popen(command, cwd='/opt/lclm',
                    env={**os.environ, 'CUDA_VISIBLE_DEVICES': str(index)}, stdout=handle,
                    stderr=subprocess.STDOUT, start_new_session=True)
            begin, last_progress = time.monotonic(), time.monotonic()
            while True:
                if self.processes[index].poll() is not None:
                    raise RuntimeError(f'Replica {index} exited; inspect {self.root}')
                try:
                    with urllib.request.urlopen(f'http://127.0.0.1:{8000 + index}/health', timeout=5) as response:
                        if response.status == 200:
                            break
                except (OSError, TimeoutError):
                    pass
                if time.monotonic() - begin > 1200:
                    raise TimeoutError(f'Replica {index} startup exceeded 20 minutes')
                if time.monotonic() - last_progress > 60:
                    print(json.dumps({'replica': index, 'startup_seconds': time.monotonic() - begin,
                        'recent_log': (self.root / f'server-{index}.log').read_text(errors='replace').splitlines()[-2:]}), flush=True)
                    last_progress = time.monotonic()
                time.sleep(2)
            capture = graph_capture_lines((self.root / f'server-{index}.log').read_text(errors='replace'))
            if not capture:
                raise RuntimeError(f'Replica {index} lacks actual CUDA graph capture evidence')
            self.clients[index] = OpenAI(api_key='not-needed', base_url=f'http://127.0.0.1:{8000 + index}/v1',
                                        timeout=300, max_retries=0)
            result = {'replica': index, 'startup_seconds': time.monotonic() - begin,
                      'graph_capture_lines': capture[-12:], 'ready': True}
            atomic_json(self.root / f'server-{index}.ready.json', result)
            print(json.dumps({'replica': index, 'ready': True, 'startup_seconds': result['startup_seconds']}), flush=True)
            return result
        try:
            with ThreadPoolExecutor(max_workers=8) as pool:
                reports = list(pool.map(launch, range(8)))
            atomic_json(self.root / 'all-ready.json', {'servers': reports, 'startup_seconds': time.monotonic() - start})
            self.commit()
        except Exception:
            self.stop()
            raise

    def healthy(self):
        return all(process is not None and process.poll() is None for process in self.processes)

    def stop(self):
        for process in self.processes:
            if process is not None and process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        for process in self.processes:
            if process is not None:
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
        self.commit()
