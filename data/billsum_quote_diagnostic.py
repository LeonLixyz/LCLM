"""Read-only quotation diagnostics; never change saved votes or accept a row.

Reassemble only consecutively numbered, actually expanded primary chunks whose
tool bodies match prepared task bytes. Quotation presence is not entailment.
"""
import re

VERSION = 'billsum-contiguous-primary-quotes-v1'


def normalized(text):
    return ' '.join(text.split())


def evidence_runs(row, task):
    if (task.get('family') != 'billsum' or row['task_id'] != task['task_id']
            or row['task'] != task['question']):
        raise ValueError('Wrong prepared task')
    first = task['question'].split('\n', 1)[0]
    match = re.fullmatch(r'Use these source documents: (.+)\.', first)
    if not match:
        raise ValueError('Missing task source identity')
    sources = match[1].split(', ')
    if len(sources) != len(set(sources)):
        raise ValueError('Duplicate source identity')
    segments = {s['segment_id']: s for s in task['segments']}
    if len(segments) != len(task['segments']):
        raise ValueError('Duplicate segment identity')
    support = set(task['support_segment_ids'])
    if not support or not support <= segments.keys():
        raise ValueError('Unknown support segment')
    pending = {}; seen_calls = set(); expanded = set()
    for message in row['messages']:
        for call in message.get('tool_calls', []):
            fn = call['function']; args = fn['arguments']
            if (message['role'] != 'assistant' or call['id'] in seen_calls
                    or fn['name'] != 'expand' or not isinstance(args, dict)
                    or set(args) != {'segment_id'} or args['segment_id'] not in segments):
                raise ValueError('Invalid expansion call')
            seen_calls.add(call['id']); pending[call['id']] = args['segment_id']
        if message['role'] != 'tool':
            continue
        if message['tool_call_id'] not in pending:
            raise ValueError('Orphan/duplicate tool response')
        segment = pending.pop(message['tool_call_id']); item = segments[segment]
        header = 'SOURCE ' + item['record_id'] + '\n'
        expected = item['text'] if item['text'].startswith(header) else header + item['text']
        if message['content'] != expected:
            raise ValueError('Tool text differs from prepared task')
        expanded.add(segment)
    if pending:
        raise ValueError('Unanswered expansion call')
    documents = {}
    for segment in sorted(expanded & support):
        item = segments[segment]
        source, marker, part = item['record_id'].rpartition(':part')
        if not marker or source not in sources or not re.fullmatch(r'0|[1-9][0-9]*', part):
            raise ValueError('Invalid primary chunk identity')
        header = 'SOURCE ' + item['record_id'] + '\n'
        body = item['text'].removeprefix(header)
        body, padding, _ = body.partition('\nRELATED SOURCE ')
        if not body.strip():
            raise ValueError('Empty primary chunk')
        chunks = documents.setdefault(source, {})
        if int(part) in chunks:
            raise ValueError('Duplicate primary chunk number')
        chunks[int(part)] = (segment, body, bool(padding))
    runs = []
    for source, chunks in sorted(documents.items()):
        current = None; previous = None; padded = False
        for part, (segment, body, has_padding) in sorted(chunks.items()):
            # A gap or padding ends a contiguous run. Never invent adjacency.
            if current is None or part != previous + 1 or padded:
                current = {'source': source, 'parts': [], 'segments': [], 'text': ''}
                runs.append(current)
            current['parts'].append(part); current['segments'].append(segment)
            current['text'] += ('\n' if current['text'] else '') + body
            previous = part; padded = has_padding
    return runs


def diagnose_quotes(row, task):
    claims = row['verification'].get('semantic_review', {}).get('summary_claims', {}).get('claims', [])
    runs = evidence_runs(row, task)
    results = []
    for index, claim in enumerate(claims):
        quotes = claim['evidence_quotes']
        if not isinstance(quotes, list) or any(not isinstance(q, str) or not q.strip() for q in quotes):
            raise ValueError('Malformed saved quotations')
        matches = [[{'source': run['source'], 'parts': run['parts'], 'segments': run['segments']}
                    for run in runs if normalized(q) in normalized(run['text'])] for q in quotes]
        results.append({'claim_index': index, 'supported_vote': claim['supported'],
                        'saved_quotes_present': claim['quotes_present'],
                        'contiguous_primary_quotes_present': bool(quotes) and all(matches),
                        'quote_matches': matches})
    return {'task_id': row['task_id'], 'version': VERSION, 'claims': results,
            'approved_for_release': False, 'training_rows_modified': False,
            'limits': 'Mechanical saved-quote presence only. No entailment decision or row recovery.'}
