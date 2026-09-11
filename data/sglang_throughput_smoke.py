"""Serving-only replay checks; no rollout continuation or training acceptance."""
import re
import statistics
from collections import Counter

from data.synthetic_expansion_agent import canonicalize_assistant_response
from data.expansion_retry_diagnostic import response_token_ids, ensure_complete_response


def first_request(records):
    if not records or records[0].get('phase') != 'rollout' or 'response' not in records[0]:
        raise ValueError('Need an already captured first-rollout request and response')
    request = records[0]['request']
    if request.get('extra_body', {}).get('chat_template_kwargs', {}).get('enable_thinking') is not False:
        raise ValueError('Replay request must explicitly disable thinking')
    if len(request.get('messages', [])) != 2 or not request.get('tools'):
        raise ValueError('Only initial native-tool requests may be replayed')
    return request


def inspect_response(payload, valid_segments):
    choice = payload['choices'][0]
    ids = response_token_ids(choice)
    ensure_complete_response(choice['finish_reason'], 'first_turn_replay')
    message = choice['message']
    canonical = canonicalize_assistant_response(message)
    calls = canonical.get('tool_calls', [])
    if len({c['id'] for c in calls}) != len(calls):
        raise ValueError('Duplicate parsed native call IDs')
    if any(c['function']['arguments']['segment_id'] not in valid_segments for c in calls):
        raise ValueError('Unknown segment in parsed native call')
    return {'output_token_ids': ids, 'native_calls': len(calls),
        'valid_native_call': bool(calls), 'text_without_call': bool(canonical['content']) and not calls,
        'nonempty_reasoning_fields': any(bool(message.get(k)) for k in ('reasoning', 'reasoning_content', 'analysis', 'thinking'))}


def graph_capture_lines(log):
    return [line for line in log.splitlines()
            if re.search(r'captur\w*[^\n]{0,50}(?:cuda\s+)?graph', line, re.I)
            and not re.search(r'(?:skip|disabl)\w*[^\n]{0,30}(?:cuda\s+)?graph', line, re.I)]


def report_requests(results, elapsed):
    counts = Counter()
    latencies = []
    for row in results:
        counts['requests'] += 1
        latencies.append(row['seconds'])
        if row.get('response'):
            counts['http_responses'] += 1
            counts['completion_tokens'] += (row['response'].get('usage') or {}).get('completion_tokens', 0) or 0
            counts['prompt_tokens'] += (row['response'].get('usage') or {}).get('prompt_tokens', 0) or 0
        if 'raw_generated_text' in row:
            counts['raw_captures'] += 1
            counts['generated_reasoning_markers'] += row['generated_reasoning_markers']
        if row.get('error'):
            counts['error:' + row['error']['type']] += 1
        else:
            counts['successful_http_and_capture'] += 1
            counts['valid_native_call'] += row['inspection']['valid_native_call']
            counts['native_calls'] += row['inspection']['native_calls']
            counts['nonempty_reasoning_fields'] += row['inspection']['nonempty_reasoning_fields']
    return {'counts': dict(counts), 'elapsed_seconds': elapsed,
        'requests_per_second': len(results) / elapsed if elapsed else None,
        'completion_tokens_per_second': counts['completion_tokens'] / elapsed if elapsed else None,
        'latency_seconds': {'min': min(latencies), 'median': statistics.median(latencies),
                           'max': max(latencies), 'mean': statistics.mean(latencies)} if latencies else {}}
