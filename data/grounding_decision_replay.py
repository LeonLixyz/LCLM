"""Recheck saved V3 decisions mechanically, with no new model requests."""
import json
from data.grounding_claim_review import review_claims


def replay_decision(row, saved):
    if saved.get('task_id') != row['task_id'] or type(saved.get('keep')) is not bool:
        raise ValueError('Decision ID/type mismatch')
    if 'error' in saved:
        if (set(saved) != {'source', 'task_id', 'keep', 'error'} or saved['keep']
                or not isinstance(saved['error'], str) or not saved['error']):
            raise ValueError('Malformed saved review error')
        return 'error_quarantined'
    index = 0
    def complete(messages):
        nonlocal index
        payload = json.loads(messages[1]['content'])
        if 'sentence' in payload:
            if index >= len(saved['sentences']): raise ValueError('Missing saved sentence')
            sentence = saved['sentences'][index]; index += 1
            if sentence['sentence'] != payload['sentence']: raise ValueError('Sentence coverage/order mismatch')
            return json.dumps({k: sentence[k] for k in ('supported', 'abstention_only', 'evidence_quotes')})
        return json.dumps(saved['answer_fit'])
    replayed = review_claims(row, complete)
    if index != len(saved['sentences']) or replayed != {k: v for k, v in saved.items() if k != 'source'}:
        raise ValueError('Saved decision differs from replayed protocol')
    return 'replayed'
