"""Finite CPU audit of the frozen corrected64 LexGLUE probe only.

This never dispatches generation, changes source decisions, or approves training.
It reads frozen inputs/results and writes derived audit artifacts under the new
corrected root. Original opaque-probe artifacts are never overwritten.
"""
import json
from collections import Counter
from pathlib import Path

import modal

from data.audit_sglang_backlog_modal import audit_snapshot
from data.lexglue_task_definition_correction import (
    CONFIGS, DEFINITIONS_SHA256, VERSION, amend_lexglue_task,
    config_from_source_row_id,
)
from data.sglang_backlog import atomic_json, file_sha, sha
from data.sglang_backlog_fast import checked_rows
from data.sglang_failed_1k_modal import cpu_image

app = modal.App('lclm-sglang27-defined-lexglue-audit-20260909-final1')
ROOT = Path('/runs/stage3-build-20260906/sglang27-lexglue-defined-20260909-v4')
inputs = modal.Volume.from_name('lclm-stage3-data', create_if_missing=False)
outputs = modal.Volume.from_name('lclm-stage3-agent-outputs-v2-20260909', create_if_missing=False, version=2)
ORIGINAL_MANIFEST_SHA256 = '96584acb7b8ba93e45b79310c9793408aa7d5b18176880136db3ef511b5a3cfe'
AUDIT_FILES = (
    'audit_sglang_lexglue_defined_modal.py', 'audit_sglang_backlog_modal.py',
    'lexglue_task_definition_correction.py', 'lexglue_task_definitions.v1.json',
    'expansion_training_quality.py', 'expansion_task_normalization.py',
    'expansion_trace_audit.py', 'expansion_retry_diagnostic.py',
    'synthetic_expansion_agent.py', 'sglang_backlog_fast.py',
    'sglang_lexglue_defined.py', 'sglang_lexglue_defined_modal.py',
)
# Capture at module import so the returned identity describes this worker's
# loaded deployment, and bind it to the client's immutable snapshot before use.
LOADED_AUDIT_CODE_SHA256 = {name: file_sha(Path(__file__).with_name(name)) for name in AUDIT_FILES}


def validate_corrected_probe_tasks(tasks, original_tasks):
    """Require exact deterministic transforms of the same original64 objects."""
    old = {task['task_id']: task for task in original_tasks}
    new = {task['task_id']: task for task in tasks}
    if len(old) != len(original_tasks) or len(new) != len(tasks) or len(new) != 64 or set(new) != set(old):
        raise ValueError('Corrected probe IDs differ from the original64')
    configs = Counter()
    for key, task in new.items():
        if task != amend_lexglue_task(old[key]):
            raise ValueError('Corrected input is not the exact frozen transform: ' + key)
        configs[config_from_source_row_id(task['source_row_id'])] += 1
    if dict(configs) != {name: 10 if name == 'case_hold' else 9 for name in CONFIGS}:
        raise ValueError('Corrected probe config coverage changed')
    return dict(sorted(configs.items()))


@app.function(image=cpu_image, cpu=8, memory=32768, timeout=7200,
              max_containers=1, retries=0, volumes={'/data': inputs, '/runs': outputs})
def audit_probe(manifest_sha256: str, decision_sha256: str):
    inputs.reload()
    outputs.reload()
    manifest_path, decision_path = ROOT / 'manifest.json', ROOT / 'probe-decision.json'
    if file_sha(manifest_path) != manifest_sha256 or file_sha(decision_path) != decision_sha256:
        raise ValueError('Root-supplied manifest/decision hashes changed')
    manifest, decision = json.loads(manifest_path.read_text()), json.loads(decision_path.read_text())
    from data.sglang_lexglue_defined import generation_config
    runtime_config = generation_config()
    if runtime_config != manifest['generation_config']:
        raise ValueError('Loaded audit snapshot does not match the frozen generator configuration')
    correction = manifest['generation_config']['corrected_lexglue']
    if (manifest['task_version'] != VERSION or correction['task_version'] != VERSION
            or correction['definitions_sha256'] != DEFINITIONS_SHA256
            or manifest['ontology_artifact']['sha256'] != DEFINITIONS_SHA256
            or file_sha(manifest['ontology_artifact']['path']) != DEFINITIONS_SHA256
            or manifest['generation_config']['code_sha256']['lexglue_task_definition_correction.py']
            != file_sha(Path(__file__).with_name('lexglue_task_definition_correction.py'))
            or manifest['old_completed_lexglue_outputs_selected_for_training'] is not False):
        raise ValueError('Corrected task/ontology/code binding changed')
    if (decision['decision'] != 'generate_corrected_lexglue_probe'
            or decision['reviewed_by'] != 'root' or decision['scope'] != 'source_probes_only'
            or decision['probe_sources'] != ['lex_glue']
            or decision['approved_for_generation'] is not True
            or decision['approved_for_release'] is not False
            or decision['backlog_manifest_sha256'] != manifest_sha256
            or decision['task_version'] != VERSION
            or decision['ontology_artifact'] != manifest['ontology_artifact']
            or decision['generation_config'] != manifest['generation_config']):
        raise ValueError('Not the root-reviewed corrected probe decision')
    source, = manifest['sources']
    if source['source'] != 'lex_glue' or source['probe_rows'] != 64 or source['chunks'][0]['rows'] != 64:
        raise ValueError('Expected only the corrected LexGLUE64 probe')
    original_path = Path(manifest['origin_probe_manifest_path'])
    if file_sha(original_path) != ORIGINAL_MANIFEST_SHA256 or manifest['origin_probe_manifest_sha256'] != ORIGINAL_MANIFEST_SHA256:
        raise ValueError('Original frozen probe manifest changed')
    original = json.loads(original_path.read_text())
    old_source, = [s for s in original['sources'] if s['source'] == 'lex_glue']
    tasks, originals = list(checked_rows(source['chunks'][0])), list(checked_rows(old_source['chunks'][0]))
    configs = validate_corrected_probe_tasks(tasks, originals)
    # This also verifies every result/raw/task hash, exact saved corrected prompts,
    # native calls, pinned tool bodies, tokenization, and assistant loss masks.
    audit = audit_snapshot('lex_glue', ROOT, outputs, 'probe-decision.json')
    from data.expansion_training_quality import inspect_expansion_efficiency
    from data.expansion_task_normalization import prepare_teacher_task
    from data.sglang_backlog_fast import teacher_messages
    from data.synthetic_expansion_agent import TEACHER_SYSTEM_PROMPT
    for key in ('reasoning_fields', 'raw_think_markers'):
        if audit['counts'].get(key, 0):
            audit['errors'].append({'error': 'unexpected_nonthinking_content', 'counter': key,
                                    'count': audit['counts'][key]})
    audit['status'] = 'failed' if audit['errors'] else 'passed'
    by_config = {name: Counter() for name in CONFIGS}
    repeated = []
    task_index = {task['task_id']: task for task in tasks}
    review_path = ROOT / 'audits' / 'lex_glue.review.jsonl'
    for line in review_path.read_text().splitlines():
        row = json.loads(line)
        config = config_from_source_row_id(task_index[row['task_id']]['source_row_id'])
        if row['source_probe_stratum'] != config:
            raise ValueError('Result config differs from frozen task identity')
        normalized = prepare_teacher_task(task_index[row['task_id']], {})
        expected_start = teacher_messages([
            {'role': 'system', 'content': TEACHER_SYSTEM_PROMPT},
            {'role': 'user', 'content': normalized['rollout_user_prompt']},
        ], 'lex_glue', 'rollout')
        raw_path = (ROOT / 'outputs' / 'lex_glue' / 'chunk-00000' / 'attempts'
                    / row['task_id'] / f"attempt-{row['attempt']:02d}" / 'raw.json')
        if file_sha(raw_path) != row['raw_responses_sha256']:
            raise ValueError('Raw response changed between audit passes')
        records = json.loads(raw_path.read_text())
        if not records or records[0]['phase'] != 'rollout':
            raise ValueError('Probe lacks its initial corrected rollout request')
        for record in records:
            if record['phase'] == 'rollout' and record['request']['messages'][:2] != expected_start:
                raise ValueError('Teacher request does not contain exact corrected task definition')
        counts = by_config[config]
        counts['rows'] += 1
        counts['verification:' + row['verification']['reason']] += 1
        if row['verification']['accepted'] is True:
            counts['accepted'] += 1
            efficiency = inspect_expansion_efficiency(row['trace']['messages'])
            counts['eligible_after_repeat_policy'] += efficiency['eligible']
            if not efficiency['eligible']:
                repeated.append({'task_id': row['task_id'], 'config': config, **efficiency})
    if {name: counts['rows'] for name, counts in by_config.items()} != configs:
        raise ValueError('Audit bundle config coverage changed')
    audit.update(task_version=VERSION, ontology_artifact=manifest['ontology_artifact'],
                 probe_decision_sha256=decision_sha256,
                 original_probe_manifest_sha256=ORIGINAL_MANIFEST_SHA256,
                 exact_original_to_corrected_transforms=64,
                 exact_corrected_teacher_requests_verified=True,
                 input_sha256=source['chunks'][0]['sha256'],
                 config_counts={name: dict(counts) for name, counts in by_config.items()},
                 accepted_repeated_expansions=repeated,
                 approved_for_internal_packing=False,
                 semantic_review_complete=False)
    audit.update(actual_audit_code_sha256=LOADED_AUDIT_CODE_SHA256,
                 runtime_generation_config_sha256=sha(json.dumps(runtime_config, sort_keys=True).encode()))
    atomic_json(ROOT / 'audits' / 'lex_glue.corrected-probe.json', audit)
    outputs.commit()
    return audit
