"""Pure count checks shared by release publication and tests."""

def validate_expansion_format_audits(audits, source_reports, manifest_sha256):
    import re
    from data.stage3_tokenizers import DECODER_REVISION, ENCODER_REVISION
    expected = {r['source']:sum(v for k,v in r['reasons'].items() if k.startswith('accepted'))
                for r in source_reports}
    actual = {r['source']:r for r in audits}
    if (not expected or len(expected) != len(source_reports) or len(actual) != len(audits)
            or set(actual) != set(expected)):
        raise ValueError('Full expansion format audit source coverage mismatch')
    manifests = manifest_sha256 if isinstance(manifest_sha256, dict) else dict.fromkeys(expected, manifest_sha256)
    if set(manifests) != set(expected):
        raise ValueError('Expansion manifest source coverage mismatch')
    for source, count in expected.items():
        r = actual[source]
        if (r.get('status') != 'passed' or r.get('rows') != count
                or r.get('expected_accepted') != count or r.get('counts',{}).get('accepted') != count
                or r.get('failed_rows') != 0 or r.get('errors') != []
                or r.get('generation_manifest_file_sha256') != manifests[source]
                or r.get('decoder_tokenizer_revision') != DECODER_REVISION
                or r.get('encoder_tokenizer_revision') != ENCODER_REVISION
                or not re.fullmatch(r'[0-9a-f]{64}',r.get('accepted_file_sha256',''))
                or (count and (r.get('minimum_segment_tokens') or 0) < 512)):
            raise ValueError(f'Incomplete/stale expansion format audit: {source}')
    return actual

def validate_final_artifact_audit(audit):
    required = {'base', 'agents', 'base_recovery', 'expansion'}
    components = audit.get('components', [])
    if (audit.get('status') != 'passed' or set(components) != required
            or len(components) != len(required)):
        raise ValueError('Final artifact validation must include every component, including expansion')

def validate_pilot_review(generation, audit, review):
    import hashlib
    import json
    digest=hashlib.sha256(json.dumps(generation['manifest'],sort_keys=True).encode()).hexdigest()
    if generation.get('status')!='complete' or audit.get('status')!='passed' or review.get('approved') is not True:
        raise ValueError('Pilot generation/audit/review is not approved')
    if audit.get('generation_manifest_sha256')!=digest or review.get('generation_manifest_sha256')!=digest:
        raise ValueError('Pilot review is stale for this generation manifest')
    accepted=sum(v for k,v in generation['reasons'].items() if k.startswith('accepted'))
    if accepted!=audit['accepted_traces'] or accepted!=review.get('accepted_traces'):
        raise ValueError('Pilot review accepted-count mismatch')
    samples=review.get('reviewed_examples',{})
    if set(samples)!=set(audit['sources']) or any(len(samples[k])<min(2,v['accepted']) for k,v in audit['sources'].items()):
        raise ValueError('Pilot review is missing source-stratified examples')
    return digest

def validate_expansion_completion(generation, source_reports, provenance):
    from collections import Counter
    if generation.get('status') != 'complete' or provenance.get('status') != 'assembled':
        raise ValueError('Expansion generation/provenance is incomplete')
    expected = {r['source']:r['tasks'] for r in provenance['sources']}
    actual = {r['source']:r for r in source_reports}
    if len(expected) != len(provenance['sources']) or len(actual) != len(source_reports) or set(expected) != set(actual):
        raise ValueError('Expansion source coverage is incomplete or duplicated')
    totals = Counter(); accepted = 0
    for source, rows in expected.items():
        report = actual[source]
        reasons = report['reasons']
        if report.get('status') != 'complete' or any(type(v) is not int or v < 0 for v in reasons.values()):
            raise ValueError(f'Invalid expansion source report: {source}')
        if sum(reasons.values()) != rows:
            raise ValueError(f'Expansion attempted/task-count mismatch: {source}')
        accepted += sum(v for k,v in reasons.items() if k.startswith('accepted'))
        totals.update(reasons)
    if dict(totals) != generation['reasons'] or sum(expected.values()) != provenance['tasks']:
        raise ValueError('Expansion aggregate accounting mismatch')
    return {'tasks':sum(expected.values()), 'accepted':accepted,
            'rejected':sum(expected.values())-accepted, 'sources':len(expected)}


def validate_base_recovery(base_reports, recovery_reports, partitions=64):
    base = {r['partition']: r['counts'] for r in base_reports}
    recovery = {r['partition']: r['counts'] for r in recovery_reports}
    expected = set(range(partitions))
    if (set(base) != expected or set(recovery) != expected
            or len(base_reports) != partitions or len(recovery_reports) != partitions):
        raise ValueError('Base/recovery partition coverage is incomplete or duplicated')
    summary = dict(input_rows=0, packed_rows=0, recovered_rows=0,
                   rejected_processing=0, over_32768=0, under_18=0)
    for partition in sorted(expected):
        original, extra = base[partition], recovery[partition]
        if extra['scanned'] != original['input_rows']:
            raise ValueError(f'Recovery scan count mismatch: {partition}')
        if extra['packed_rows'] != extra['recovered_rows']:
            raise ValueError(f'Recovery packing count mismatch: {partition}')
        recovered = extra['recovered_rows']
        resolved = recovered + extra.get('over_32768', 0) + extra.get('under_18', 0)
        if resolved > original.get('rejected_processing', 0):
            raise ValueError(f'Recovery exceeds original rejected rows: {partition}')
        if extra['candidates'] != resolved + extra.get('not_recoverable', 0):
            raise ValueError(f'Recovery candidate accounting mismatch: {partition}')
        summary['input_rows'] += original['input_rows']
        summary['packed_rows'] += original['packed_rows'] + recovered
        summary['recovered_rows'] += recovered
        summary['rejected_processing'] += original.get('rejected_processing', 0) - resolved
        for key in ('over_32768', 'under_18'):
            summary[key] += original.get(key, 0) + extra.get(key, 0)
    if summary['input_rows'] != sum(summary[key] for key in
            ('packed_rows', 'rejected_processing', 'over_32768', 'under_18')):
        raise ValueError('Combined base input/output accounting mismatch')
    return summary
