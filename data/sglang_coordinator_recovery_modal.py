"""CPU-only recovery of completed-input coordinator; original GPU code unchanged."""
from data.sglang_backlog_storage_modal import *
from data.sglang_backlog_storage_modal import APP_NAME as ORIGINAL_APP
from data.sglang_backlog_storage_modal import app as original_app
from data import sglang_backlog_storage_modal as original_module
import hashlib

APP_NAME = 'lclm-sglang27-coordinator-recovery-20260909-v2'
app = modal.App(APP_NAME)
OLD_COORDINATOR = 'fc-01M22H5ZKSWDDDDYRZZA1NT3Z8'
DECISION = 'e0dbab13a9cab459721d1e7eee8b703c11e492ec798353aa14b8f233c4f1336a'
EXPECTED_GPU_MODULE_SHA = '30f5fe3d29b30a5517320f299b756593b60cdac4d634b080f9141c1b3bdad3ac'
RECOVERY_ROOT = ROOT / 'coordinator-recovery-20260909-v2'


def optional_progress(path):
    """Advisory live telemetry may be momentarily incomplete after Volume reload."""
    try:
        document = json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    if not isinstance(document, dict) or not isinstance(document.get('counts'), dict):
        return None
    return document['counts']


def require_failed_original():
    try:
        result = modal.FunctionCall.from_id(OLD_COORDINATOR).get(timeout=0)
    except json.JSONDecodeError:
        return {'status': 'terminal_failure', 'error': 'JSONDecodeError', 'call_id': OLD_COORDINATOR}
    except TimeoutError:
        raise RuntimeError('Original coordinator is still active')
    raise RuntimeError('Unexpected original coordinator outcome; requires reconciliation')


def recovery_preflight():
    inputs.reload(); volume.reload()
    terminal = require_failed_original()
    if file_sha(original_module.__file__) != EXPECTED_GPU_MODULE_SHA:
        raise ValueError('Original frozen generation module changed')
    manifest, _, allowed = load_gate(DECISION, 'continuation')
    path = ROOT / 'coordinators' / ('continuation-' + DECISION) / 'state.json'
    state = json.loads(path.read_text())
    rows = accepted = reports = 0
    for entry in state['entries'].values():
        report_path = ROOT / 'outputs' / entry['source'] / f"chunk-{entry['chunk']:05d}" / 'report.json'
        report = json.loads(report_path.read_text())
        if report['status'] != 'complete' or report['completed'] != report['rows']:
            raise ValueError('Old in-flight chunk lacks completed accounting')
        if report['backlog_manifest_sha256'] != file_sha(ROOT/'manifest.json') or report['decision_sha256'] != DECISION:
            raise ValueError('Completed report review/input changed')
        rows += report['completed']; accepted += report['counts'].get('automatic_accepted', 0); reports += 1
    return {'status':'ready_to_resume_original_reviewed_plan','old_coordinator':terminal,
        'completed_rows':rows,'automatic_accepted':accepted,'completed_reports':reports,
        'original_state_sha256':file_sha(path),'decision_sha256':DECISION,
        'manifest_sha256':file_sha(ROOT/'manifest.json'),'generation_config':manifest['generation_config'],
        'gpu_module_sha256':EXPECTED_GPU_MODULE_SHA,'recovery_module_sha256':file_sha(__file__),
        'reason':'Original coordinator failed only while parsing advisory live progress; final child report completed.'}


@app.function(**cpu_options,cpu=2,memory=8192,timeout=1800)
def check_recovery():
    return recovery_preflight()


@app.function(**cpu_options,cpu=2,memory=8192,timeout=1800)
def recovery_tests():
    import subprocess
    result = subprocess.run(['python','-m','pytest','-q','tests/test_sglang_coordinator_recovery.py'],
        cwd='/opt/lclm',capture_output=True,text=True)
    print(result.stdout,result.stderr,flush=True)
    if result.returncode: raise RuntimeError('Recovery tests failed')
    return {'status':'passed','output':result.stdout,'recovery_module_sha256':file_sha(__file__),
        'gpu_module_sha256':file_sha(original_module.__file__)}


def repaired_coordinate(decision_sha256, stage):
    """Finite automatic handoff; known in-flight calls are awaited by the successor."""
    import time
    from data.sglang_backlog_storage import resolve_recorded_successor
    inputs.reload()
    volume.reload()
    manifest, _, allowed = load_gate(decision_sha256, stage)
    reused = reused_probe_routes()
    identity = {'function_call_id': modal.current_function_call_id(), 'input_id': modal.current_input_id()}
    path = ROOT / 'coordinators' / (stage + '-' + decision_sha256) / 'state.json'
    plan = [(s['source'], c['index']) for index in range(max(len(s['chunks']) for s in manifest['sources']))
        for s in manifest['sources'] if s['source'] in allowed
        for c in s['chunks'] if c['index'] == index and (stage == 'probe') == (index == 0)]
    binding = {'decision_sha256': decision_sha256, 'stage': stage,
        'manifest_sha256': file_sha(ROOT / 'manifest.json'), 'plan_sha256': sha(json.dumps(plan).encode())}
    state = json.loads(path.read_text()) if path.exists() else {'binding': binding, 'entries': {}, 'status': 'running', 'handoffs': []}
    if state['binding'] != binding:
        raise ValueError('Coordinator plan/review binding changed')
    if state.get('completion_report_path'):
        if file_sha(state['completion_report_path']) != state['completion_report_sha256']:
            raise ValueError('Terminal accounting changed')
        return {key: state[key] for key in ('status', 'completion_report_path', 'completion_report_sha256')}

    def persist():
        state['updated_at'] = time.time()
        atomic_json(path, state)
        volume.commit()

    def child_ids(call_id, suffix):
        def walk(nodes):
            for node in nodes:
                yield node
                yield from walk(node.children)
        return {node.function_call_id for node in walk(modal.FunctionCall.from_id(call_id).get_call_graph())
                if node.function_name.endswith(suffix)}

    # The previous coordinator may have been preempted immediately after spawning us.
    if state.get('handoffs'):
        last = state['handoffs'][-1]
        if last.get('status') != 'successor_started':
            successor_id = resolve_recorded_successor(last, parent_identity=last['parent_identity'],
                child_call_ids=child_ids(last['parent_identity']['function_call_id'], 'drive_plan'))
            last['successor_call_id'] = successor_id
            if identity == last['parent_identity']:
                persist()
                return {'status': 'handed_off', 'successor_call_id': successor_id, 'state_path': str(path)}
            if identity['function_call_id'] != successor_id:
                raise ValueError('Different coordinator cannot claim an approved handoff')
            last.update(status='successor_started', successor_identity=identity)
    state.update(status='running', current_identity=identity)
    persist()
    generator, started = modal.Cls.from_name(ORIGINAL_APP, 'Generator')(), time.monotonic()

    def handoff_if_due():
        if time.monotonic() - started < 17 * 3600:
            return None
        if len(state['handoffs']) >= len(plan) + 1:
            raise RuntimeError('Finite coordinator handoff bound exhausted')
        item = {'parent_identity': identity, 'status': 'dispatching', 'started_at': time.time(),
                'known_successor_call_ids': sorted(child_ids(identity['function_call_id'], 'drive_plan')),
                'plan_sha256': binding['plan_sha256']}
        state['handoffs'].append(item)
        persist()
        successor = drive_plan.spawn(decision_sha256, stage)
        item.update(successor_call_id=successor.object_id, status='spawned')
        state['status'] = 'handed_off'
        persist()
        return {'status': 'handed_off', 'successor_call_id': successor.object_id,
                'state_path': str(path), 'binding': binding}

    def hold(source, index, detail):
        item = {'source': source, 'chunk': index, 'status': 'dispatch_needs_reconciliation',
                'detail': detail, 'coordinator': str(path), 'approved_for_release': False}
        atomic_json(ROOT / 'outputs' / source / 'HOLD.json', item)
        print(json.dumps(item), flush=True)

    for source, index in plan:
        volume.reload()
        source_root = ROOT / 'outputs' / source
        report_path = report_path_for(source, index, reused)
        key = source + ':' + str(index)
        if report_path.exists():
            report = json.loads(report_path.read_text())
            if report['status'] in ('complete', 'held', 'needs_reconciliation', 'attempts_exhausted'):
                state['entries'][key] = {**state['entries'].get(key, {}), 'status': report['status'],
                                         'report_sha256': file_sha(report_path),
                                         'completed': report['completed'], 'counts': report['counts']}
                persist()
                continue
        if (source_root / 'HOLD.json').exists():
            continue
        if index:
            prior = predecessor_path(source, index, origin_root=ORIGIN, output_root=ROOT, reused_probes=reused)
            if not prior.exists() or json.loads(prior.read_text())['status'] != 'complete':
                continue
        next_coordinator = handoff_if_due()
        if next_coordinator:
            return next_coordinator
        entry = state['entries'].get(key)
        if entry and not entry.get('child_call_id'):
            owner_path = source_root / f'chunk-{index:05d}' / 'owner.json'
            child_id = None
            if owner_path.exists():
                owner = json.loads(owner_path.read_text())
                if (owner['binding']['source'] == source and owner['binding']['chunk'] == index
                        and owner['binding']['decision_sha256'] == decision_sha256):
                    child_id = owner['identity']['function_call_id']
            if child_id is None:
                known = {r.get('child_call_id') for r in state['entries'].values()}
                candidates = child_ids(entry['coordinator_identity']['function_call_id'], 'chunk') - known
                if len(candidates) == 1:
                    child_id = candidates.pop()
            if child_id is None:
                hold(source, index, 'Interrupted dispatch has no unambiguous durable child call ID')
                entry['status'] = 'dispatch_needs_reconciliation'
                persist()
                continue
            entry['child_call_id'] = child_id
            persist()
        if not entry:
            entry = {'status': 'dispatching', 'source': source, 'chunk': index,
                     'coordinator_identity': identity, 'started_at': time.time()}
            state['entries'][key] = entry
            persist()
            try:
                child = generator.chunk.spawn(source, index, decision_sha256, stage)
                entry.update(child_call_id=child.object_id, status='running')
                persist()
            except Exception as exc:
                hold(source, index, f'Dispatch outcome is unclassified: {type(exc).__name__}: {exc}')
                entry['status'] = 'dispatch_needs_reconciliation'
                persist()
                continue
        child = modal.FunctionCall.from_id(entry['child_call_id'])
        last_progress = None
        while True:
            next_coordinator = handoff_if_due()
            if next_coordinator:
                return next_coordinator
            try:
                report = child.get(timeout=60)
                entry.update(status=report['status'], completed=report['completed'], counts=report['counts'])
                volume.reload()
                entry['report_sha256'] = file_sha(report_path)
                persist()
                print(json.dumps({'source': source, 'chunk': index, 'status': report['status'],
                                  'counts': report['counts']}), flush=True)
                break
            except TimeoutError:
                volume.reload()
                progress_path = source_root / f'chunk-{index:05d}' / 'progress.json'
                progress = optional_progress(progress_path)
                if progress is not None and progress != last_progress:
                    print(json.dumps({'source': source, 'chunk': index, 'progress': progress}), flush=True)
                    last_progress = progress
            except Exception as exc:
                hold(source, index, f'Child failed without a terminal report: {type(exc).__name__}: {exc}')
                entry.update(status='dispatch_needs_reconciliation', error={'type': type(exc).__name__, 'message': str(exc)})
                persist()
                break
    return finish_report(manifest, stage, allowed, state, path)


@app.function(**cpu_options,cpu=4,memory=16384,timeout=86400,max_containers=1,retries=0)
def drive_plan(decision_sha256: str, stage: str):
    if decision_sha256 != DECISION or stage != 'continuation':
        raise ValueError('Recovery only resumes the existing approved continuation')
    path = RECOVERY_ROOT / 'recovery.json'
    if path.exists():
        saved = json.loads(path.read_text())
        if saved['decision_sha256'] != DECISION or saved['recovery_module_sha256'] != file_sha(__file__):
            raise ValueError('Recovery lineage changed')
    else:
        evidence = recovery_preflight()
        atomic_json(path, evidence); volume.commit()
    return repaired_coordinate(decision_sha256, stage)
