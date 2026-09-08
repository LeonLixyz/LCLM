"""Fail-closed gate for scaling the independently reviewed dialogue repair."""


def validate_corrective_review(generation, audit, review, manifest_digest, accepted_digest, accepted_ids, expected_tasks_sha):
    manifest = generation['manifest']
    accepted = sum(v for k, v in generation['reasons'].items() if k.startswith('accepted'))
    if (generation['status'] != 'complete' or sum(generation['reasons'].values()) != 32
            or manifest.get('pilot_limit') != 32 or accepted <= 0):
        raise ValueError('Corrective pilot must be complete and nonempty')
    if (manifest.get('model') != 'Qwen/Qwen3-235B-A22B-Instruct-2507'
            or manifest.get('model_revision') != 'ac9c66cc9b46af7306746a9250f23d47083d689e'
            or manifest.get('teacher_prompt_saved_in_training_messages') is not False
            or manifest.get('semantic_review') != 'question-first-every-claim-v1'):
        raise ValueError('Corrective teacher or verification policy changed')
    provenance = manifest.get('input_provenance', {})
    if (provenance.get('source') != 'multidoc2dial'
            or provenance.get('question_rendering_version') != 'multidoc2dial-chronological-v1'
            or provenance.get('tasks_sha256') != expected_tasks_sha):
        raise ValueError('Wrong corrective input provenance')
    if (audit.get('status') != 'passed' or audit.get('source') != 'multidoc2dial'
            or audit.get('failed_rows') != 0 or audit.get('rows') != accepted
            or audit.get('expected_accepted') != accepted or len(accepted_ids) != accepted
            or audit.get('generation_manifest_file_sha256') != manifest_digest
            or audit.get('accepted_file_sha256') != accepted_digest):
        raise ValueError('Missing or stale corrective format audit')
    reviewed = review.get('reviewed_task_ids', [])
    if (review.get('approved_for_full_generation') is not True
            or review.get('source') != 'multidoc2dial'
            or review.get('generation_manifest_file_sha256') != manifest_digest
            or review.get('accepted_file_sha256') != accepted_digest
            or review.get('accepted_traces') != accepted
            or not isinstance(reviewed, list) or len(set(reviewed)) < 2
            or not set(reviewed) <= accepted_ids):
        raise ValueError('Explicit hash-bound corrective sample review is required')
    return {'accepted': accepted, 'reviewed': len(set(reviewed))}
