"""Training-only quality checks for immutable deterministic expansion traces.

This does not modify generation, answer verification, tool messages, or saved
attempts. Repeating an expand call returns the same source segment, so those
trajectories are excluded from training instead of being edited into new ones.
"""
import json
from collections import Counter

POLICY_VERSION = 'no-repeated-immutable-expansion-v1'


def inspect_expansion_efficiency(messages):
    calls = []
    for message in messages:
        if message.get('role') != 'assistant':
            continue
        for call in message.get('tool_calls') or []:
            function = call['function']
            arguments = function['arguments']
            if isinstance(arguments, str):
                arguments = json.loads(arguments)
            if (call.get('type') != 'function' or function.get('name') != 'expand'
                    or not isinstance(arguments, dict) or set(arguments) != {'segment_id'}
                    or not isinstance(arguments['segment_id'], str) or not arguments['segment_id']):
                raise ValueError('Quality policy requires a canonical expansion trace')
            calls.append(arguments['segment_id'])
    if not calls:
        raise ValueError('Expansion trace has no expansion calls')
    counts = Counter(calls)
    repeated = {key: count for key, count in sorted(counts.items()) if count > 1}
    return {'policy': POLICY_VERSION, 'eligible': not repeated,
            'reason': 'repeated_unchanged_segment' if repeated else 'no_repeated_expansion',
            'tool_calls': len(calls), 'distinct_expanded_segments': len(counts),
            'redundant_calls': len(calls) - len(counts), 'repeated_segments': repeated}
