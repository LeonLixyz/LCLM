"""Validate the official PubMedQA fold-0 training allowlist, not HF split names."""
REVISION = '1cbae8e92f72f20c8d3747cbb3bf5bc53554d997'

def training_ids(manifest):
    if manifest.get('revision') != REVISION or manifest.get('fold') != 0:
        raise ValueError('Unknown PubMedQA split provenance')
    groups = [manifest[key] for key in ('train_ids', 'dev_ids', 'test_ids')]
    expected = [450, 50, 500]
    for ids, count in zip(groups, expected):
        if len(ids) != count or len(set(ids)) != count or not all(isinstance(i, str) and i.isdigit() for i in ids):
            raise ValueError('Invalid official PubMedQA split IDs/counts')
    train, dev, test = map(set, groups)
    if train & dev or train & test or dev & test:
        raise ValueError('PubMedQA split overlap')
    return train
