"""Storage-only continuation lineage; model/prompt/verifier settings stay frozen."""
from __future__ import annotations

import copy
import json
from collections import Counter
from pathlib import Path

from data.sglang_backlog import sha, file_sha
from data.sglang_backlog_fast import generation_config as probe_generation_config, checked_rows

OUTPUT_VOLUME = 'lclm-stage3-agent-outputs-v2-20260909'
INPUT_VOLUME = 'lclm-stage3-data'
ORIGIN_MANIFEST_SHA256 = '96584acb7b8ba93e45b79310c9793408aa7d5b18176880136db3ef511b5a3cfe'
ORIGIN_DECISION_SHA256 = 'fbb29de5edf2572c840a97f6955f39ad939abbe7f3508e347b654d240d53bbcd'
EXPECTED_REMAINING = {'retry_rejected': 65352, 'previously_unattempted': 206935}


def generation_config():
    config = probe_generation_config()
    config['storage_successor'] = {
        'version': 'volume-v2-output-only-with-probe-resume-v1', 'input_volume': INPUT_VOLUME,
        'input_mount': '/data', 'input_volume_version': 1,
        'output_volume': OUTPUT_VOLUME, 'output_mount': '/runs', 'output_volume_version': 2,
        'origin_probe_manifest_sha256': ORIGIN_MANIFEST_SHA256,
        'origin_probe_decision_sha256': ORIGIN_DECISION_SHA256,
        'teacher_serving_prompt_validators': 'identical to origin probe configuration',
        'coordinator': 'finite automatic handoff; durable child IDs; no uncertain duplicate dispatch',
        'server_logs': 'active local files; atomic durable snapshots only',
        'probe_resume': 'reuse completed original probe reports; root-reconciled zero-reservation probes only',
    }
    for name in ('sglang_backlog_storage.py', 'sglang_backlog_storage_modal.py'):
        config['code_sha256'][name] = file_sha(Path(__file__).parent / name)
    return config


def verify_storage_only_config(config, origin_config):
    reduced = copy.deepcopy(config)
    storage = reduced.pop('storage_successor', None)
    if not storage or storage['output_volume'] != OUTPUT_VOLUME or storage['output_volume_version'] != 2:
        raise ValueError('Missing explicit Volume-v2 storage lineage')
    for name in ('sglang_backlog_storage.py', 'sglang_backlog_storage_modal.py'):
        if not reduced['code_sha256'].pop(name, None):
            raise ValueError('Storage successor code hashes are missing')
    if reduced != origin_config:
        raise ValueError('Successor changed model, serving, prompt, validator, or original code')


def continuation_coverage(origin):
    """Read compact committed ledgers, excluding exactly the original probe IDs."""
    seen, probe_ids, counts, sources = set(), set(), Counter(), []
    for source in origin['sources']:
        selected = set(source['probe_ids'])
        if len(selected) != source['probe_rows'] or probe_ids & selected:
            raise ValueError('Origin has duplicate or inconsistent probe IDs')
        probe_ids.update(selected)
        ids_by_chunk = {spec['index']: [] for spec in source['chunks']}
        source_counts = Counter()
        for row in checked_rows(source['coverage']):
            key, chunk = row['task_id'], row['chunk']
            if key in seen or chunk not in ids_by_chunk:
                raise ValueError('Duplicate/unknown origin coverage task')
            seen.add(key)
            ids_by_chunk[chunk].append(key)
            if (key in selected) != (chunk == 0):
                raise ValueError('Probe/continuation membership disagrees')
            if chunk:
                counts[row['kind']] += 1
                source_counts[row['kind']] += 1
        for spec in source['chunks']:
            ids = ids_by_chunk[spec['index']]
            if len(ids) != spec['rows'] or sha(json.dumps(sorted(ids)).encode()) != spec['task_ids_sha256']:
                raise ValueError('Origin coverage differs from effective chunk IDs')
        item = copy.deepcopy(source)
        item['origin_pending_rows'] = source['rows']
        item['rows'] = source['rows'] - source['probe_rows']
        item['counts'] = dict(source_counts)
        item['probe_output_location'] = 'original_v2_volume_v1'
        item['continuation_output_location'] = 'new_volume_v2'
        # Retain original indexes so index1 follows the external index0 probe.
        sources.append(item)
    if (len(seen) != origin['pending_rows'] or len(probe_ids) != origin['probe_rows']
            or not probe_ids <= seen or sum(counts.values()) != origin['continuation_rows']):
        raise ValueError('Origin/probe/continuation global coverage failed')
    return sources, dict(counts), sha(json.dumps(sorted(seen - probe_ids)).encode())


def predecessor_path(source, index, *, origin_root, output_root, reused_probes=None):
    if type(index) is not int or index < 1:
        raise ValueError('Continuation cannot execute or precede probe chunk0')
    root = Path(origin_root) if index == 1 and source in (reused_probes or {}) else Path(output_root)
    return root / 'outputs' / source / f'chunk-{index-1:05d}' / 'report.json'


def validate_continuation_decision(decision, manifest, origin, origin_decision, probes, *,
                                   manifest_sha, origin_manifest_sha, origin_decision_sha):
    from data.sglang_backlog_fast import validate_source_decision
    if (decision.get('decision') != 'continue_reviewed_sources_with_27b_on_volume_v2'
            or decision.get('reviewed_by') != 'root' or decision.get('approved_for_generation') is not True
            or decision.get('approved_for_release') is not False or decision.get('scope') != 'reviewed_sources'):
        raise ValueError('Missing root-reviewed storage continuation decision')
    bindings = {'backlog_manifest_sha256': manifest_sha,
                'origin_probe_manifest_sha256': origin_manifest_sha,
                'origin_probe_decision_sha256': origin_decision_sha}
    if any(decision.get(key) != value for key, value in bindings.items()):
        raise ValueError('Continuation does not match original probes/new storage manifest')
    if decision.get('generation_config') != manifest['generation_config']:
        raise ValueError('Continuation code/storage configuration changed')
    verify_storage_only_config(manifest['generation_config'], origin['generation_config'])
    if manifest['pending_rows'] != 272287 or manifest['pending_counts'] != EXPECTED_REMAINING:
        raise ValueError('Storage continuation has changed the remaining task set')
    # Reuse the strict original probe-report validator with its ORIGINAL bindings.
    review = copy.deepcopy(decision)
    # Actual mixed-storage reports are validated against their real bindings first.
    translated_probes = copy.deepcopy(probes)
    for source, item in translated_probes.items():
        report = item['report']
        if report.get('backlog_manifest_sha256') == manifest_sha:
            if report.get('decision_sha256') != decision.get('resume_probe_decision_sha256'):
                raise ValueError('New probe report is not bound to the reviewed resume decision')
            report.update(backlog_manifest_sha256=origin_manifest_sha, decision_sha256=origin_decision_sha)
        elif report.get('backlog_manifest_sha256') != origin_manifest_sha:
            raise ValueError('Probe report belongs to an unknown manifest')
    review.update(backlog_manifest_sha256=origin_manifest_sha, probe_decision_sha256=origin_decision_sha)
    allowed = validate_source_decision(review, origin, stage='continuation',
        base_decision_sha=origin_decision_sha, probe_reports=translated_probes)
    if not allowed <= set(origin_decision['probe_sources']):
        raise ValueError('Continuation includes a source never approved for probes')
    return allowed


def resolve_recorded_successor(handoff, *, parent_identity, child_call_ids):
    """Resume a known handoff; never invent a second successor for an uncertain one."""
    if handoff.get('parent_identity') != parent_identity:
        raise ValueError('Unclassified coordinator handoff identity')
    recorded = handoff.get('successor_call_id')
    if recorded:
        return recorded
    candidates = set(child_call_ids) - set(handoff.get('known_successor_call_ids', []))
    if len(candidates) != 1:
        raise ValueError('Interrupted handoff has no unambiguous successor call')
    return candidates.pop()


def validate_resume_decision(decision, manifest, origin, origin_decision, *, manifest_sha):
    if (decision.get('decision') != 'resume_original_source_probes_on_volume_v2'
            or decision.get('reviewed_by') != 'root' or decision.get('approved_for_generation') is not True
            or decision.get('approved_for_release') is not False or decision.get('scope') != 'source_probes_only'
            or decision.get('original_coordinator_stopped') is not True
            or not decision.get('original_coordinator_call_id')
            or not str(decision.get('reconciliation_review', '')).strip()):
        raise ValueError('Missing explicit root reconciliation of original stopped probes')
    for key, expected in [('backlog_manifest_sha256', manifest_sha),
            ('origin_probe_manifest_sha256', ORIGIN_MANIFEST_SHA256),
            ('origin_probe_decision_sha256', ORIGIN_DECISION_SHA256)]:
        if decision.get(key) != expected:
            raise ValueError('Probe resume lineage changed')
    if decision.get('generation_config') != manifest['generation_config']:
        raise ValueError('Probe resume code/storage configuration changed')
    verify_storage_only_config(manifest['generation_config'], origin['generation_config'])
    reused, fresh = decision.get('reused_probe_reports'), decision.get('never_attempted_probe_sources')
    if not isinstance(reused, dict) or not isinstance(fresh, list) or len(set(fresh)) != len(fresh):
        raise ValueError('Missing exact completed/never-attempted probe partition')
    if set(reused) & set(fresh) or set(reused) | set(fresh) != set(origin_decision['probe_sources']):
        raise ValueError('Probe resume does not partition original approved sources exactly once')
    if any(not isinstance(value, str) or len(value) != 64 for value in reused.values()):
        raise ValueError('Reused probe report hashes are required')
    return set(fresh), reused


def validate_reused_probe(report, *, source, origin_manifest_sha=ORIGIN_MANIFEST_SHA256,
                          origin_decision_sha=ORIGIN_DECISION_SHA256):
    spec = source['chunks'][0]
    if (report.get('status') != 'complete' or report.get('source') != source['source']
            or report.get('chunk') != 0 or report.get('rows') != source['probe_rows']
            or report.get('completed') != source['probe_rows'] or source['probe_rows'] <= 0
            or report.get('input_sha256') != spec['sha256']
            or report.get('effective_task_ids_sha256') != spec['task_ids_sha256']
            or report.get('backlog_manifest_sha256') != origin_manifest_sha
            or report.get('decision_sha256') != origin_decision_sha
            or set(report.get('result_hashes', {})) != set(source['probe_ids'])):
        raise ValueError('Reused original probe is incomplete or differently bound')


def validate_never_attempted(directory):
    """The old pre-request reload error is not a teacher attempt."""
    directory = Path(directory)
    journal = directory / 'attempts.jsonl'
    if ((journal.exists() and journal.stat().st_size) or any((directory / 'results').glob('*.json'))
            or any((directory / 'attempts').rglob('raw.json'))):
        raise ValueError('Original probe has attempts/results; explicit partial-attempt migration is required')
    if (directory / 'report.json').exists():
        report = json.loads((directory / 'report.json').read_text())
        if report.get('completed', 0) or report.get('unresolved_attempt_ids'):
            raise ValueError('Original report contains committed or unresolved attempts')


def compact_completion(manifest, *, stage, allowed, chunks, holds, binding, coordinator, handoffs):
    """Small manifest binds per-task hashes via immutable chunk report hashes."""
    sources, totals = {}, Counter()
    for source in manifest['sources']:
        name = source['source']
        expected = source['probe_rows'] if stage == 'probe' else source['rows']
        records = chunks.get(name, [])
        completed = sum(item['completed'] for item in records)
        unresolved = sum(item.get('unresolved', 0) for item in records)
        if completed + unresolved > expected:
            raise ValueError('Completion accounting exceeds finite source coverage')
        counts = Counter()
        for item in records:
            counts.update(item.get('counts', {}))
        sources[name] = {'expected_rows': expected, 'approved_for_stage': name in allowed,
            'completed': completed, 'unresolved': unresolved, 'unattempted': expected-completed-unresolved,
            'counts': dict(counts), 'held': name in holds, 'hold': holds.get(name), 'chunks': records}
        totals.update(expected_rows=expected, completed=completed, unresolved=unresolved,
                      unattempted=expected-completed-unresolved)
    eligible_complete = all(s['completed'] == s['expected_rows'] and not s['unresolved'] and not s['held']
                            for name, s in sources.items() if name in allowed)
    return {'status': 'approved_plan_complete' if eligible_complete else 'approved_plan_drained_with_holds',
        **binding, 'stage': stage, 'sources': sources, 'totals': dict(totals),
        'generation_config': manifest['generation_config'], 'coordinator_state_path': coordinator,
        'successor_terminal_status': 'terminal', 'coordinator_handoffs': handoffs,
        'approved_for_release': False, 'internal_training_export_requires_format_audit': True}
