"""Read-only token, native-tool, saved-label audit of the completed diagnostic."""
import json
from pathlib import Path

import modal

from data.sglang_failed_1k_modal import cpu_image, volume, OUTPUT

app = modal.App('lclm-sglang-failed-1k-audit-20260909-v1')


@app.function(image=cpu_image, cpu=8, memory=32768, timeout=7200,
              volumes={'/data': volume})
def audit():
    import hashlib
    import multiprocessing as mp
    import re
    from collections import Counter, defaultdict
    from concurrent.futures import ProcessPoolExecutor
    from data.expansion_trace_audit import init_worker, audit_worker
    from data.expansion_retry_diagnostic import summarize, response_token_ids
    from data.pubmedqa_split import training_ids

    def digest(raw):
        return hashlib.sha256(raw).hexdigest()

    volume.reload()
    manifest_raw = (OUTPUT / 'manifest.json').read_bytes()
    manifest = json.loads(manifest_raw)
    report_raw = (OUTPUT / 'report.json').read_bytes()
    report = json.loads(report_raw)
    results_raw = (OUTPUT / 'results.jsonl').read_bytes()
    if not results_raw.endswith(b'\n'):
        raise ValueError('Incomplete result JSONL')
    rows = [json.loads(line) for line in results_raw.splitlines()]
    if (report['status'] != 'complete_diagnostic_pending_review' or
            report['completed'] != len(rows) or len(rows) != 1000 or
            report['manifest_sha256'] != digest(manifest_raw) or
            report['results_sha256'] != digest(results_raw)):
        raise ValueError('Incomplete or changed pilot outputs')
    inputs = {}
    for source in manifest['sources']:
        raw = (OUTPUT / (source['source'] + '.inputs.json')).read_bytes()
        if digest(raw) != source['inputs_sha256']:
            raise ValueError('Frozen source input changed')
        for case in json.loads(raw):
            key = case['task']['task_id']
            if key in inputs:
                raise ValueError('Duplicate frozen task')
            inputs[key] = case
    if len({r['task_id'] for r in rows}) != 1000 or set(inputs) != {r['task_id'] for r in rows}:
        raise ValueError('Result task coverage differs from pilot inputs')
    counts, usage, errors = Counter(), Counter(), []
    source_usage = defaultdict(Counter)
    marker = re.compile(r'</?(?:think|analysis)>|<\|(?:think|analysis)', re.I)
    allowed = training_ids(json.loads(Path('/data/stage3-agent/real-expansion/sources/pubmedqa_labeled/official-splits/split-manifest.json').read_text()))
    audit_payloads = []
    for row in rows:
        key, source = row['task_id'], row['source']
        records_raw = (OUTPUT / 'raw-responses' / (key + '.json')).read_bytes()
        if digest(records_raw) != row['raw_responses_sha256']:
            raise ValueError('Raw response hash mismatch: ' + key)
        records = json.loads(records_raw)
        if len(records) != row['requests']:
            raise ValueError('Request count mismatch: ' + key)
        for record in records:
            counts['requests'] += 1
            request = record['request']
            if request['extra_body']['chat_template_kwargs']['enable_thinking'] is not False:
                errors.append({'task_id': key, 'error': 'thinking_not_disabled'})
            if 'error' in record:
                counts['api_errors:' + record['error']['type']] += 1
            response = record.get('response')
            if not response:
                continue
            choice = response['choices'][0]
            response_token_ids(choice)
            counts['finish:' + str(choice['finish_reason'])] += 1
            message = choice['message']
            counts['nonempty_reasoning_fields'] += any(message.get(k) for k in ('reasoning', 'reasoning_content', 'analysis', 'thinking'))
            counts['raw_responses_with_think_markers'] += bool(marker.search(record.get('raw_generated_text', '')))
            parsed_calls = message.get('tool_calls') or []
            counts['parsed_native_calls'] += len(parsed_calls)
            raw_calls = record.get('raw_generated_text', '').count('</tool_call>')
            if raw_calls != len(parsed_calls):
                counts['raw_parsed_tool_count_mismatches'] += 1
                errors.append({'task_id': key, 'phase': record['phase'], 'error': 'raw_parsed_tool_count_mismatch', 'raw': raw_calls, 'parsed': len(parsed_calls)})
            for name in ('prompt_tokens', 'completion_tokens', 'total_tokens'):
                n = (response.get('usage') or {}).get(name) or 0
                usage[name] += n
                source_usage[source][name] += n
        if row['verification']['accepted'] is True:
            trace = row['trace']
            case = inputs[key]
            if (trace['messages'][0] != {'role': 'system', 'content': case['task']['training_system_prompt']} or
                    trace['messages'][1]['content'] != case['task']['training_user_prompt']):
                errors.append({'task_id': key, 'error': 'saved_prompt_changed'})
            for message in trace['messages']:
                if message['role'] == 'assistant' and marker.search(message.get('content') or ''):
                    errors.append({'task_id': key, 'error': 'saved_assistant_think_marker'})
            audit_payloads.append((trace, source, allowed))
    trace_counts, trace_errors, minimum = Counter(), [], None
    with ProcessPoolExecutor(max_workers=8, mp_context=mp.get_context('spawn'), initializer=init_worker) as pool:
        for result in pool.map(audit_worker, audit_payloads):
            if result.get('error'):
                trace_errors.append(result)
            else:
                metrics = result['metrics']
                size = metrics.pop('minimum_segment_tokens')
                minimum = size if minimum is None else min(size, minimum)
                trace_counts.update(metrics)
    result = {'status': 'passed' if not errors and not trace_errors else 'failed',
        'pilot_manifest_sha256': digest(manifest_raw), 'pilot_report_sha256': digest(report_raw),
        'pilot_results_sha256': digest(results_raw), 'automatic_accepted': len(audit_payloads),
        'response_counts': dict(counts), 'token_usage': dict(usage),
        'source_token_usage': {k: dict(v) for k, v in source_usage.items()},
        'accepted_trace_counts': dict(trace_counts), 'minimum_segment_tokens': minimum,
        'response_errors': errors, 'trace_errors': trace_errors,
        'summary': summarize(rows), 'approved_for_release': False,
        'scope': 'All raw-response hashes and nonthinking requests; raw/parsed tool counts; every accepted saved prompt, exact tool result, native calls, final harvesting, source verifier, memory bodies, decoder IDs and independent assistant loss labels. Semantic correctness still requires source review.'}
    destination = OUTPUT / 'independent-format-audit.json'
    destination.write_text(json.dumps(result, indent=2))
    volume.commit()
    return result
