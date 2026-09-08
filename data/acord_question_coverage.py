"""Extract query text without confusing prepared and normalized task wording."""
PREFIX = 'Rate how relevant the cited contract clause is to this request: '
FINAL = '\nReturn exactly FINAL: followed by your answer.'
SCALES = {'beir_0_4': '. Return the integer relevance grade from 0 (irrelevant) to 4 (highly relevant).',
          'legacy_prepared_1_5': '. Return the integer relevance grade from 1 (irrelevant) to 5 (highly relevant).'}


def parse_question(question, allow_legacy=False):
    header, sep, body = question.partition('\n')
    if (not sep or not header.startswith('Use these source documents: ') or not header.endswith('.')
            or not body.startswith(PREFIX) or not body.endswith(FINAL)):
        raise ValueError('Unexpected ACORD question envelope')
    body = body[len(PREFIX):-len(FINAL)]
    for version, suffix in SCALES.items():
        if body.endswith(suffix):
            query = body[:-len(suffix)]
            if not query.strip() or (version != 'beir_0_4' and not allow_legacy):
                raise ValueError('Empty query or unnormalized saved task')
            return query, version
    raise ValueError('Unknown ACORD relevance scale')
