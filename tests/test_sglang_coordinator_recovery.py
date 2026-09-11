import json
from types import SimpleNamespace

import pytest

from data import sglang_coordinator_recovery_modal as recovery


@pytest.mark.parametrize('raw', [None, '', '{"counts":', '[]', '{"counts": null}'])
def test_optional_partial_progress_is_not_a_generation_failure(tmp_path, raw):
    path = tmp_path / 'progress.json'
    if raw is not None:
        path.write_text(raw)
    assert recovery.optional_progress(path) is None


def test_valid_progress_is_reported(tmp_path):
    path = tmp_path / 'progress.json'
    path.write_text('{"counts":{"completed":37,"automatic_accepted":12}}')
    assert recovery.optional_progress(path) == {'completed': 37, 'automatic_accepted': 12}


def test_original_gpu_module_is_unchanged():
    from pathlib import Path
    assert recovery.file_sha(Path(recovery.__file__).with_name('sglang_backlog_storage_modal.py')) == recovery.EXPECTED_GPU_MODULE_SHA


def test_empty_live_progress_then_completed_child_finishes_without_redispatch(tmp_path, monkeypatch):
    root = tmp_path
    source = root / 'outputs' / 'synthetic'
    output = source / 'chunk-00001'
    output.mkdir(parents=True)
    (root / 'manifest.json').write_text('{}')
    prior = source / 'chunk-00000' / 'report.json'
    prior.parent.mkdir()
    prior.write_text('{"status":"complete"}')
    report = {'status': 'complete', 'completed': 1, 'counts': {'completed': 1, 'automatic_accepted': 1}}
    manifest = {'sources': [{'source': 'synthetic', 'chunks': [{'index': 0}, {'index': 1}]}]}
    noop = SimpleNamespace(reload=lambda: None, commit=lambda: None)
    monkeypatch.setattr(recovery, 'ROOT', root)
    monkeypatch.setattr(recovery, 'inputs', noop)
    monkeypatch.setattr(recovery, 'volume', noop)
    monkeypatch.setattr(recovery, 'load_gate', lambda *_: (manifest, {}, {'synthetic'}))
    monkeypatch.setattr(recovery, 'reused_probe_routes', lambda: {})
    monkeypatch.setattr(recovery, 'report_path_for', lambda *_: output / 'report.json')
    monkeypatch.setattr(recovery, 'predecessor_path', lambda *_, **__: prior)
    monkeypatch.setattr(recovery.modal, 'current_function_call_id', lambda: 'fc-new')
    monkeypatch.setattr(recovery.modal, 'current_input_id', lambda: 'in-new')
    dispatches = []
    child_gets = []

    def get(timeout):
        child_gets.append(timeout)
        if len(child_gets) == 1:
            (output / 'progress.json').write_text('')
            raise TimeoutError()
        (output / 'report.json').write_text(json.dumps(report))
        return report

    def spawn(*args):
        dispatches.append(args)
        return SimpleNamespace(object_id='fc-child')

    remote = SimpleNamespace(chunk=SimpleNamespace(spawn=spawn))
    monkeypatch.setattr(recovery.modal.Cls, 'from_name', lambda app, name: lambda: remote)
    monkeypatch.setattr(recovery.modal.FunctionCall, 'from_id', lambda _: SimpleNamespace(get=get))
    monkeypatch.setattr(recovery, 'finish_report', lambda m, stage, allowed, state, path: state)
    first = recovery.repaired_coordinate('decision', 'continuation')
    assert first['entries']['synthetic:1']['status'] == 'complete'
    assert first['entries']['synthetic:1']['counts']['automatic_accepted'] == 1
    assert len(dispatches) == 1 and len(child_gets) == 2
    # A restarted coordinator reuses completed output and fixes saved counters.
    second = recovery.repaired_coordinate('decision', 'continuation')
    assert second['entries']['synthetic:1']['completed'] == 1
    assert len(dispatches) == 1 and len(child_gets) == 2
