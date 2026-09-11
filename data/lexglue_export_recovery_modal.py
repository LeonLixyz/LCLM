"""CPU-only repair of a confirmed failed LexGLUE export; no generation changes."""
import json
import subprocess
from pathlib import Path
import modal
from data import lexglue_selected_completion_modal as original
from data import lexglue_export_recovery as recovery
from data import lexglue_selected_completion as completed
from data import expansion_completion_chain as chain
from data.sglang_backlog import atomic_json

APP_NAME = 'lclm-lexglue-export-recovery-20260910-v4'
app = modal.App(APP_NAME)
OLD_WORKER = 'fc-01M25VH9H3F0C0GJCFMR19PVA3'
OLD_CHAIN = 'fc-01M23CHSD69R8F86WE1S49CAGZ'
ROOT = completed.GENERATION_ROOT/'export-recovery-20260910-v1'


def require_failed(call_id):
    try:
        modal.FunctionCall.from_id(call_id).get(timeout=0)
    except ValueError as exc:
        chain.core.require(str(exc) == 'Unrelated incomplete export owner', 'Unexpected old failure')
        return
    raise ValueError('Old owner is not a confirmed terminal ownership failure')


def configuration():
    recovery.verify_originals()
    return {'version': 'lexglue-export-execution-recovery-v1',
        'original_configuration': completed.configuration(),
        'code_sha256': {name: chain.file_sha(Path(recovery.__file__).parent/name)
            for name in ['lexglue_export_recovery.py', 'lexglue_export_recovery_modal.py']},
        'read_workers': 16, 'max_pending_reads': 32,
        'semantics': 'Exact audited selection/export bodies; only bounded ordered frozen-byte reads and logical retry ownership change.',
        'old_worker': OLD_WORKER, 'old_chain': OLD_CHAIN}


@app.function(**original.options, cpu=4, memory=16384, timeout=1800)
def tests():
    result = subprocess.run(['python', '-m', 'pytest', '-q',
        'tests/test_lexglue_export_recovery.py', 'tests/test_lexglue_selected_completion.py',
        'tests/test_export_reviewed_sglang27.py'], cwd='/opt/lclm', capture_output=True, text=True)
    print(result.stdout, result.stderr, flush=True)
    if result.returncode:
        raise RuntimeError('Recovery tests failed')
    return {'status': 'passed', 'output': result.stdout, 'configuration': configuration()}


@app.function(**original.options, cpu=2, memory=8192, timeout=1800)
def check(recovery_spec):
    original.reload()
    decision = chain.document(recovery_spec)
    chain.core.require(decision['reviewed_by'] == 'root' and decision['approved_for_execution'] is True
        and decision['configuration'] == configuration(), 'Unreviewed execution recovery')
    require_failed(OLD_WORKER); require_failed(OLD_CHAIN)
    owner = chain.document(decision['old_owner'])
    state = chain.document(decision['old_chain_state'])
    chain.core.require(owner['identity']['function_call_id'] == OLD_WORKER
        and state['status'] == 'needs_attention'
        and state['children']['selection_export']['call_id'] == OLD_WORKER,
        'Recovery does not refer to actual failed ownership')
    args = state['children']['selection_export']['arguments']
    policy_path, policy_sha, review, manifest, terminal, base = args
    _, _, doc, _, _ = completed.governing(policy_path, policy_sha, review, manifest)
    completed.terminal_binding(terminal, manifest, review, doc)
    chain.core.require(chain.base_snapshot(chain.load_policy(policy_path,policy_sha)[0]) == base,
        'Completed base changed')
    chain.core.require(owner['binding']['review'] == review and owner['binding']['manifest'] == manifest
        and owner['binding']['terminal'] == terminal, 'Frozen input lineage changed')
    return {'status': 'ready_to_recover', 'arguments': args, 'configuration': configuration()}


@app.function(**original.options, cpu=8, memory=65536, timeout=86400, max_containers=1, retries=0)
def selection_and_export(recovery_spec):
    import uuid
    original.reload()
    decision = chain.document(recovery_spec)
    chain.core.require(decision['configuration'] == configuration()
        and decision['reviewed_by'] == 'root' and decision['approved_for_execution'] is True,
        'Recovery review/code changed')
    require_failed(OLD_WORKER); require_failed(OLD_CHAIN)
    prior_state = chain.document(decision['old_chain_state'])
    args = prior_state['children']['selection_export']['arguments']
    policy_path, policy_sha, review, manifest, terminal, base = args
    completed.governing(policy_path, policy_sha, review, manifest)
    original_binding = chain.document(decision['old_owner'])['binding']
    binding = {'recovery': recovery_spec, 'arguments': args, 'configuration': configuration()}
    ROOT.mkdir(parents=True, exist_ok=True)
    final = ROOT/'report.json'
    if final.exists():
        result = json.loads(final.read_text())
        chain.core.require(result['recovery_binding'] == binding, 'Completed recovery changed')
        chain.core.frozen_bytes(result['selection']);chain.core.frozen_bytes(result['transport'])
        return result
    owner = ROOT/'owner.json'; current = original.identity()
    if owner.exists():
        chain.core.require(recovery.same_owner(json.loads(owner.read_text()), current, binding),
            'Different logical call cannot claim unfinished recovery')
    else:
        atomic_json(owner, {'identity': current, 'binding': binding});original.outputs.commit()
    def stage_marker(stage):
        return ROOT/(stage+'-stage.json')
    def recover_marker(stage, marker):
        if stage_marker(stage).exists():
            result = json.loads(stage_marker(stage).read_text());chain.core.frozen_bytes(result['artifact']);return result['artifact']
        candidates = []
        for path in (ROOT/(stage+'-attempts')).glob('*/'+marker):
            try: document=json.loads(path.read_text())
            except (ValueError, UnicodeDecodeError):continue
            if document.get('status')=='reviewed_complete':candidates.append(path)
        chain.core.require(len(candidates)<=1, 'Multiple completed recovery stages')
        return {'path':str(candidates[0]),'sha256':chain.file_sha(candidates[0])} if candidates else None
    selection_spec = recover_marker('selection', 'selection.json')
    if selection_spec is None:
        destination = ROOT/'selection-attempts'/uuid.uuid4().hex
        result = recovery.selection_function()(*args, destination)
        selection_spec = result['selection']
    # Preserve the original completed selection artifact. Bind recovery in a new
    # immutable document, including after preemption between these two writes.
    selection_spec = recovery.bind_selection_recovery(selection_spec, recovery_spec)
    chosen = chain.document(selection_spec)
    chain.core.require(chosen.get('execution_recovery') == recovery_spec, 'Recovered selection lacks execution lineage')
    atomic_json(stage_marker('selection'), {'artifact': selection_spec});original.outputs.commit()
    print(json.dumps({'stage':'selection_complete','counts':chosen['sources'][0]['counts']}),flush=True)
    transport_spec = recover_marker('export', 'manifest.json')
    if transport_spec is None:
        destination = ROOT/'export-attempts'/uuid.uuid4().hex
        transport = recovery.export_function()(selection_spec['path'],selection_spec['sha256'],destination,
            reviewed_sources=['lex_glue'],excluded_task_ids=chosen['excluded_task_ids'])
        transport_spec = {'path':str(destination/'manifest.json'),'sha256':chain.file_sha(destination/'manifest.json')}
    transport = chain.document(transport_spec)
    chain.core.require(transport['selection_sha256']==selection_spec['sha256'], 'Recovered export selection changed')
    for spec in transport['files']:
        chain.core.require(chain.file_sha(Path(transport['transport_root'])/spec['name'])==spec['sha256'],
            'Recovered completed shard bytes changed')
    atomic_json(stage_marker('export'), {'artifact': transport_spec});original.outputs.commit()
    result = {'status':'complete','binding':original_binding,'selection':selection_spec,
        'transport':transport_spec,'rows':transport['rows'],'recovery_binding':binding}
    atomic_json(final,result);original.outputs.commit()
    return result
