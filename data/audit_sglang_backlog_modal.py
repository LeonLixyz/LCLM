"""Read-only audit of complete fast-backlog source probes and saved labels."""
import json
from pathlib import Path

import modal

from data.sglang_failed_1k_modal import cpu_image, volume

app = modal.App('lclm-sglang27-source-probe-audit-20260909-v1')
ROOT = Path('/data/stage3-build-20260906/sglang27-backlog-fast-20260909-v2')


@app.function(image=cpu_image, cpu=8, memory=32768, timeout=7200,
              max_containers=2, volumes={'/data': volume})
def audit_source(source_name: str):
    return audit_snapshot(source_name, ROOT, volume, 'probe-decision.json')


def audit_snapshot(source_name, root, output_volume, decision_filename):
    ROOT = Path(root)
    volume = output_volume
    import multiprocessing as mp
    import re
    from collections import Counter
    from concurrent.futures import ProcessPoolExecutor
    from data.expansion_task_normalization import prepare_teacher_task
    from data.expansion_trace_audit import init_worker, audit_worker
    from data.expansion_retry_diagnostic import response_token_ids
    from data.sglang_backlog import atomic_json, file_sha, sha
    from data.sglang_backlog_fast import checked_rows
    from data.pubmedqa_split import training_ids

    volume.reload()
    manifest = json.loads((ROOT / 'manifest.json').read_text())
    source, = [s for s in manifest['sources'] if s['source'] == source_name]
    spec = source['chunks'][0]
    output = ROOT / 'outputs' / source_name / 'chunk-00000'
    report = json.loads((output / 'report.json').read_text())
    assert report['status'] == 'complete'
    assert report['completed'] == report['rows'] == spec['rows'] == source['probe_rows'] > 0
    assert report['input_sha256'] == spec['sha256']
    assert report['backlog_manifest_sha256'] == file_sha(ROOT / 'manifest.json')
    assert report['decision_sha256'] == file_sha(ROOT / decision_filename)
    choices = json.loads(Path(manifest['maud_ontology_path']).read_text()) if source_name == 'maud' else {}
    tasks = {t['task_id']: t for t in checked_rows(spec)}
    paths = sorted((output / 'results').glob('*.json'))
    assert {p.stem for p in paths} == set(tasks) == set(report['result_hashes'])
    allowed = training_ids(json.loads(Path('/data/stage3-agent/real-expansion/sources/pubmedqa_labeled/official-splits/split-manifest.json').read_text()))
    counts, errors, warnings, payloads, review_rows = Counter(), [], [], [], []
    think = re.compile(r'</?(?:think|analysis)>|<\|(?:think|analysis)', re.I)
    for path in paths:
        assert file_sha(path) == report['result_hashes'][path.stem]
        row = json.loads(path.read_text())
        review_rows.append(row)
        key = row['task_id']
        assert row['task_sha256'] == sha(json.dumps(tasks[key], sort_keys=True).encode())
        raw_path = output / 'attempts' / key / f"attempt-{row['attempt']:02d}" / 'raw.json'
        assert file_sha(raw_path) == row['raw_responses_sha256']
        records = json.loads(raw_path.read_text())
        assert len(records) == row['requests']
        counts['tasks'] += 1
        counts['accepted'] += row['verification']['accepted'] is True
        for record in records:
            counts['requests'] += 1
            request = record['request']
            assert request['extra_body']['chat_template_kwargs']['enable_thinking'] is False
            if record.get('error'):
                counts['request_error:' + record['error']['type']] += 1
            response = record.get('response')
            if not response:
                continue
            choice = response['choices'][0]
            response_token_ids(choice)
            message, raw = choice['message'], record['raw_generated_text']
            counts['reasoning_fields'] += any(message.get(k) for k in ('reasoning', 'reasoning_content', 'analysis', 'thinking'))
            counts['raw_think_markers'] += bool(think.search(raw))
            parsed = len(message.get('tool_calls') or [])
            openings = len(re.findall(r'<tool_call>\s*<function=', raw))
            closings = raw.count('</tool_call>')
            counts['parsed_native_calls'] += parsed
            if openings != parsed or closings != parsed:
                finding = {'task_id': key, 'phase': record['phase'], 'accepted': row['verification']['accepted'],
                           'tools_enabled': bool(request.get('tools')), 'raw_openings': openings,
                           'raw_closings': closings, 'parsed_calls': parsed}
                # Tool-shaped text after the final tool budget is not parser loss.
                # Preserve every such warning for source review, even on rejected rows.
                warnings.append(finding)
                if request.get('tools') and openings != parsed:
                    errors.append({**finding, 'error': 'tool_enabled_raw_parsed_call_mismatch'})
        if row['verification']['accepted'] is True:
            trace, expected = row['trace'], prepare_teacher_task(tasks[key], choices)
            assert trace['messages'][0] == {'role': 'system', 'content': expected['training_system_prompt']}
            assert trace['messages'][1] == {'role': 'user', 'content': expected['training_user_prompt']}
            assert not any(think.search(m.get('content') or '') for m in trace['messages'] if m['role'] == 'assistant')
            payloads.append((trace, source_name, allowed))
    trace_counts, minimum = Counter(), None
    with ProcessPoolExecutor(max_workers=8, mp_context=mp.get_context('spawn'), initializer=init_worker) as pool:
        for result in pool.map(audit_worker, payloads):
            if result.get('error'):
                errors.append(result)
            else:
                metrics = result['metrics']
                size = metrics.pop('minimum_segment_tokens')
                minimum = size if minimum is None else min(size, minimum)
                trace_counts.update(metrics)
    result = {'status': 'failed' if errors else 'passed', 'source': source_name,
              'probe_report_sha256': file_sha(output / 'report.json'),
              'backlog_manifest_sha256': file_sha(ROOT / 'manifest.json'),
              'counts': dict(counts), 'trace_counts': dict(trace_counts),
              'minimum_segment_tokens': minimum, 'errors': errors, 'raw_tool_warnings': warnings,
              'approved_for_release': False,
              'scope': 'All probe result/raw hashes, nonthinking requests, raw/native calls, exact saved prompts and accepted tool bodies, pinned tokenization and independent assistant loss masks. Source semantic correctness requires separate review.'}
    audit_root = ROOT / 'audits'
    audit_root.mkdir(exist_ok=True)
    review_path = audit_root / (source_name + '.review.jsonl')
    review_path.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in review_rows))
    result['review_bundle_sha256'] = file_sha(review_path)
    atomic_json(audit_root / (source_name + '.probe.json'), result)
    volume.commit()
    print(json.dumps({k: v for k, v in result.items() if k not in ('scope', 'raw_tool_warnings')}), flush=True)
    return result
