"""Execution-only recovery: stable ownership and bounded parallel frozen reads.

The original audited selection/export functions remain byte-identical on disk.
We derive their execution bodies with exactly three explicit read replacements;
all original validation, quality filtering, ordering and output logic is retained.
Runtime reports bind both original hashes and this additional recovery module.
"""
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import inspect
import json
from pathlib import Path

from data import lexglue_selected_completion as completed
from data import export_reviewed_sglang27 as core

ORIGINAL_HASHES = {
    'lexglue_selected_completion.py': 'c55d390c0cbf9b8b58418b243c1124d9adb0d7375aa1ccabdb9fb05fa00a4a6f',
    'lexglue_selected_completion_modal.py': 'e7b0024fa784d69e8ec2caad6ab9034ee7a9e69fd69ed505c26c3685355c4a00',
    'export_reviewed_sglang27.py': '58125204fdfac615be310eb5452c130af9d2f9f35f3fd232c22de90421831b60',
}


def verify_originals():
    for name, digest in ORIGINAL_HASHES.items():
        core.require(core.sha((Path(__file__).parent/name).read_bytes()) == digest,
                     'Original audited code changed: ' + name)


def same_owner(saved, current, binding):
    """Platform retries preserve logical input IDs but change their suffix."""
    old = saved.get('identity', {})
    return (saved.get('binding') == binding
            and bool(old.get('function_call_id'))
            and old.get('function_call_id') == current.get('function_call_id')
            and bool(old.get('input_id')) and bool(current.get('input_id'))
            and old['input_id'].split(':', 1)[0] == current['input_id'].split(':', 1)[0])


def ordered_records(tasks, report, result_root, raw_root=None, *, workers=16):
    """At most32 pending results/raw captures; yield in the original sorted order."""
    core.require(0 < len(tasks) <= 512 and 1 <= workers <= 16, 'Unbounded recovery read')
    def read(key):
        core.require(Path(key).name == key and key not in ('.', '..'), 'Unsafe task filename')
        result = json.loads(core.frozen_bytes({
            'path': str(Path(result_root)/(key+'.json')), 'sha256': report['result_hashes'][key]}))
        if raw_root is None:
            return key, result
        attempt = result['attempt']
        core.require(type(attempt) is int and 1 <= attempt <= 3, 'Invalid result attempt')
        raw = json.loads(core.frozen_bytes({
            'path': str(Path(raw_root)/key/f'attempt-{attempt:02d}'/'raw.json'),
            'sha256': result['raw_responses_sha256']}))
        return key, result, raw
    keys = iter(sorted(tasks))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = deque()
        for _ in range(workers*2):
            key = next(keys, None)
            if key is not None:
                pending.append(pool.submit(read, key))
        while pending:
            yield pending.popleft().result()
            key = next(keys, None)
            if key is not None:
                pending.append(pool.submit(read, key))


def replace_once(source, old, new):
    core.require(source.count(old) == 1, 'Audited execution body no longer matches recovery patch')
    return source.replace(old, new, 1)


def selection_function():
    verify_originals()
    source = inspect.getsource(completed.freeze_selection)
    source = replace_once(source, 'for key in sorted(tasks):',
        "for key, result in ordered_records(tasks, report, directory/'results'):")
    source = replace_once(source,
        "                    result=chain.document({'path':str(directory/'results'/(key+'.json')),'sha256':report['result_hashes'][key]})\n", '')
    namespace = dict(completed.__dict__, ordered_records=ordered_records)
    exec(compile(source, '<audited-lexglue-selection-parallel-reads>', 'exec'), namespace)
    return namespace['freeze_selection']


def export_function():
    verify_originals()
    source = inspect.getsource(core.export_reviewed)
    source = replace_once(source, 'for task_id in sorted(tasks):',
        'for task_id, result, raw_records in ordered_records(tasks, report, result_root, raw_root):')
    source = replace_once(source, '                        result = json.loads(frozen_bytes(result_spec))\n', '')
    source = replace_once(source, 'check_raw(json.loads(frozen_bytes(raw_spec)), result)', 'check_raw(raw_records, result)')
    namespace = dict(core.__dict__, ordered_records=ordered_records)
    exec(compile(source, '<audited-sglang-export-parallel-reads>', 'exec'), namespace)
    return namespace['export_reviewed']


def bind_selection_recovery(selection_spec, recovery_spec):
    selected = completed.chain.document(selection_spec)
    if 'execution_recovery' in selected:
        core.require(selected['execution_recovery'] == recovery_spec, 'Different execution recovery')
        return selection_spec
    selected['execution_recovery'] = recovery_spec
    selected['approval']['artifacts'].append(recovery_spec)
    return completed.chain.write_document(Path(selection_spec['path']).parent/'execution-selection.json', selected)
