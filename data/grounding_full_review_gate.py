"""Fail-closed scale approval for review-only jobs, not dataset publication."""
TARGETS = {'faithdial': 7884, 'clapnq': 622}
PROTOCOL_SHA = 'd633cf782271f5d4ff68705229570e6ba8ad2186ca27f878a8baa5c7dbb63563'
INPUT_SHA = 'f1caf75d1e4716aaa8cb49c710b13cd1fa15726fdf6103405dbc4ae3dc0a2eac'


def validate_scale_approval(report, decisions, review, samples, decisions_sha, protocol_sha):
    manifest = report['manifest']; by_id = {r['task_id']: r for r in decisions}
    if (report.get('status') != 'complete' or report.get('rows') != 58 or len(decisions) != 58
            or len(by_id) != 58 or any(type(r.get('keep')) is not bool for r in decisions)
            or any(r['keep'] and 'error' in r for r in decisions)
            or report.get('kept') != sum(r['keep'] for r in decisions)
            or report.get('errors') != sum('error' in r for r in decisions)
            or report.get('calibration_passed') is not True):
        raise ValueError('Incomplete or inconsistent calibration')
    if (protocol_sha != PROTOCOL_SHA or manifest.get('protocol_sha256') != protocol_sha
            or manifest.get('input_rows_sha256') != INPUT_SHA
            or manifest.get('model') != 'Qwen/Qwen3-235B-A22B-Instruct-2507'
            or manifest.get('model_revision') != 'ac9c66cc9b46af7306746a9250f23d47083d689e'
            or manifest.get('enable_thinking') is not False
            or manifest.get('training_rows_modified') is not False
            or manifest.get('control_labels_sent_to_judge') is not False):
        raise ValueError('Unreviewed model/protocol/provenance')
    controls = report.get('controls', {})
    if len(controls) != 4 or sorted(r['expected_keep'] for r in controls.values()) != [False, False, True, True]:
        raise ValueError('Missing calibration control coverage')
    for task_id, control in controls.items():
        actual = by_id.get(task_id, {})
        if ('error' in actual or actual.get('keep') != control['expected_keep']
                or control.get('passed') is not True or control.get('actual_keep') != actual.get('keep')):
            raise ValueError('Calibration control did not pass')
    sample_ids = {r['task_id'] for r in samples['examples']}
    reviewed = review.get('reviewed_task_ids', [])
    if (samples.get('decisions_sha256') != decisions_sha or samples.get('protocol_sha256') != protocol_sha
            or len(sample_ids) < 8 or len(sample_ids) != len(samples['examples']) or not sample_ids <= set(by_id)
            or review.get('approved_for_full_source_review') is not True
            or review.get('approved_for_release') is not False
            or review.get('protocol_sha256') != protocol_sha
            or review.get('decisions_sha256') != decisions_sha
            or review.get('input_rows_sha256') != INPUT_SHA
            or review.get('target_sources') != TARGETS
            or len(reviewed) != len(set(reviewed))
            or not (sample_ids | set(controls)) <= set(reviewed) <= set(by_id)):
        raise ValueError('Missing/stale explicit manual scale approval')
    return {'candidate_rows': sum(TARGETS.values()), 'reviewed_examples': len(reviewed)}
