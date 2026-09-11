"""Unique CPU app: bounded read-only checksums, then one gated final writer."""
import json
import time
import traceback

import modal
from data import grouped_stage3_packing_modal as core
from data import grouped_stage3_parallel_audit as audit

app = modal.App('lclm-grouped-parallel-audit-20260909-v1')
AUDIT = core.ROOT / 'parallel-audit-v1'
OLD_COORDINATOR = 'fc-01M22DXV1TJP5Q75NQJ8J3GG6N'
OLD_FINALIZER = 'fc-01M22HM0HEAJWQP4PF16GBF4ZB'
MANIFEST_SHA = '2e9887cfa70d5419a3ab3548c38404eb251173bb48d3c80d25f69557db763250'
CAPACITY = 16


def old_terminal():
    graph = modal.FunctionCall.from_id(OLD_COORDINATOR).get_call_graph()
    evidence = audit.original_finalizer_terminal(graph, OLD_FINALIZER)
    # Coordinator cannot launch or retry the old writer after this terminal gate.
    try:
        modal.FunctionCall.from_id(OLD_COORDINATOR).get(timeout=0)
    except TimeoutError:
        raise AssertionError('Old completion coordinator remains active')
    except Exception as error:
        evidence['coordinator_terminal_exception'] = f'{type(error).__name__}: {error}'
    else:
        evidence['coordinator_status'] = 'completed'
    return evidence


@app.function(**core.options, cpu=2, memory=4096, timeout=3600, max_containers=CAPACITY)
def checksum(partition, manifest_sha=MANIFEST_SHA):
    core.volume.reload()
    try:
        result = audit.audit_partition(core.ROOT, AUDIT, partition, manifest_sha, core.volume.commit)
        return {'partition': partition, 'status': 'complete',
                'bytes': sum(f['bytes'] for output in result['outputs'].values() for f in output['files'])}
    except BaseException as error:
        path = AUDIT / 'errors' / f'part-{partition:03d}-{int(time.time())}.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({'exception': str(error), 'traceback': traceback.format_exc()}, indent=2))
        core.volume.commit()
        raise


@app.function(**core.options, cpu=4, memory=16384, timeout=3600, max_containers=1)
def publish(owner_id, manifest_sha=MANIFEST_SHA):
    core.volume.reload()
    owner = json.loads((AUDIT / 'owner.json').read_text())
    assert owner['function_call_id'] == owner_id
    def gate():
        evidence = old_terminal()
        evidence['new_coordinator'] = owner_id
        evidence['new_finalizer'] = modal.current_function_call_id()
        (AUDIT / 'single-writer-evidence.json').write_text(json.dumps(evidence, indent=2))
    gate()
    result = audit.finalize_verified(core.ROOT, AUDIT, manifest_sha, core.volume.commit, before_publish=gate)
    return {length: {'status': value['status'], 'packed_sequences': value['packed_sequences']}
            for length, value in result.items()}


@app.function(**core.options, cpu=1, memory=2048, timeout=14400, max_containers=1)
def launch(manifest_sha=MANIFEST_SHA):
    core.volume.reload()
    AUDIT.mkdir(parents=True, exist_ok=True)
    owner_id = modal.current_function_call_id()
    owner_path = AUDIT / 'owner.json'
    if owner_path.exists():
        assert json.loads(owner_path.read_text())['function_call_id'] == owner_id
    else:
        owner_path.write_text(json.dumps({'function_call_id': owner_id, 'manifest_sha256': manifest_sha}))
        core.volume.commit()
    pending = list(range(core.PARTITIONS))
    active, complete, retries = {}, set(), {}
    while pending or active:
        core.volume.reload()
        for partition in list(pending):
            if audit.certificate_path(AUDIT, partition).exists():
                audit.validate_certificate(core.ROOT, AUDIT, partition, manifest_sha)
                pending.remove(partition)
                complete.add(partition)
        while pending and len(active) < CAPACITY:
            partition = pending.pop(0)
            active[partition] = checksum.spawn(partition, manifest_sha)
        for partition, call in list(active.items()):
            try:
                value = call.get(timeout=0)
            except TimeoutError:
                continue
            except Exception as error:
                infrastructure = any(word in str(error).lower() for word in ('preempt', 'cancelled', 'terminated', 'heartbeat', 'connection'))
                if not infrastructure or retries.get(partition, 0) >= 2:
                    raise
                retries[partition] = retries.get(partition, 0) + 1
                pending.append(partition)
            else:
                assert value['status'] == 'complete'
                complete.add(partition)
            del active[partition]
        progress = {'completed': len(complete), 'active': {str(p): c.object_id for p, c in active.items()},
                    'pending': pending, 'retries': retries, 'updated_at_unix': time.time()}
        (AUDIT / 'progress.json').write_text(json.dumps(progress, indent=2))
        core.volume.commit()
        print(json.dumps(progress), flush=True)
        if pending or active:
            time.sleep(30)
    # Read-only hashing may overlap the original writer; publication cannot.
    evidence = old_terminal()
    (AUDIT / 'original-terminal-before-publish.json').write_text(json.dumps(evidence, indent=2))
    core.volume.commit()
    return publish.remote(owner_id, manifest_sha)


@app.function(**core.options, cpu=2, memory=8192, timeout=600, max_containers=1)
def tests():
    import subprocess
    result = subprocess.run(['python', '-m', 'pytest', '-q',
        '/opt/lclm/tests/test_grouped_stage3_parallel_audit.py'], capture_output=True, text=True)
    print(result.stdout, flush=True)
    print(result.stderr, flush=True)
    assert result.returncode == 0
    return {'status': 'passed', 'stdout': result.stdout}
