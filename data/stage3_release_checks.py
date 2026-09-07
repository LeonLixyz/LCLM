"""Pure count checks shared by release publication and tests."""


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
