"""One immutable, audited generation per source; never append a repair to V6.

Read-only selection shared by export and publication. This deliberately fails
until BOTH full runs and the corrective full-source review are complete.
"""
import hashlib
import json
from collections import Counter
from pathlib import Path

PILOTS = Path('/data/stage3-agent/real-expansion/pilots')
MAIN = PILOTS / 'full-20260906-v6'
REPAIR = PILOTS / 'multidoc2dial-chronological-v1-full'
TASK_SHA = 'ed8c72b55cd790cdc009b60e59088174ba8dfd35fb9f4d76d594379e1b005649'
PARENT_SHA = '3bcbf4e63cc096983319e958391e1745831ff7a56eb118db8fcc9084b9f82473'
SOURCE_REVISION = '1108a969d076f04c7367f0c2427d1c5d6d6bdaa0'


def selection_digest(selection):
    return hashlib.sha256(json.dumps(selection, sort_keys=True).encode()).hexdigest()


def validate_source_sample_review(source, audit, samples, review):
    """A format pass cannot override a failed or missing manual sample review."""
    expected = {x['training_row']['task_id']: x['category'] for x in samples.get('examples', [])}
    checked = review.get('reviewed_examples', [])
    actual = {x['task_id']: x['category'] for x in checked}
    if (not expected or len(expected) != len(samples['examples'])
            or len(actual) != len(checked) or expected != actual
            or samples.get('source') != source or review.get('source') != source
            or samples.get('accepted_file_sha256') != audit['accepted_file_sha256']
            or review.get('accepted_file_sha256') != audit['accepted_file_sha256']
            or review.get('status') != 'sample_review_passed'
            or any(x.get('result') != 'pass' for x in checked)):
        raise ValueError('Missing, failed or stale source manual review: ' + source)


def validate_replacement(generation, report, audit, review, manifest_sha, accepted_ids):
    from data.stage3_release_checks import validate_expansion_completion, validate_expansion_format_audits
    counts = validate_expansion_completion(generation, [report], {
        'status': 'assembled', 'tasks': 21451,
        'sources': [{'source': 'multidoc2dial', 'tasks': 21451}]})
    manifest = generation['manifest']
    if (manifest.get('pilot_limit') is not None
            or manifest.get('model') != 'Qwen/Qwen3-235B-A22B-Instruct-2507'
            or manifest.get('model_revision') != 'ac9c66cc9b46af7306746a9250f23d47083d689e'
            or manifest.get('teacher_prompt_saved_in_training_messages') is not False
            or manifest.get('semantic_review') != 'question-first-every-claim-v1'
            or manifest.get('input_provenance') != {
                'source': 'multidoc2dial', 'question_rendering_version': 'multidoc2dial-chronological-v1',
                'source_revision': SOURCE_REVISION, 'tasks_sha256': TASK_SHA,
                'parent_tasks_sha256': PARENT_SHA}):
        raise ValueError('Full corrective generation provenance mismatch')
    validate_expansion_format_audits([audit], [report], manifest_sha)
    reviewed = review.get('reviewed_task_ids', [])
    if (counts['accepted'] <= 0 or len(accepted_ids) != counts['accepted']
            or any(not task_id.startswith('rea4-md2d-') for task_id in accepted_ids)
            or review.get('source') != 'multidoc2dial'
            or review.get('approved_for_release') is not True
            or review.get('attempted_traces') != 21451
            or review.get('accepted_traces') != counts['accepted']
            or review.get('corrected_tasks_sha256') != TASK_SHA
            or review.get('generation_manifest_file_sha256') != manifest_sha
            or review.get('accepted_file_sha256') != audit['accepted_file_sha256']
            or not isinstance(reviewed, list) or len(set(reviewed)) < 2
            or not set(reviewed) <= accepted_ids):
        raise ValueError('Missing/stale full corrective release review; pilot approval is insufficient')
    return counts


def load_selection(root):
    from data.stage3_release_checks import validate_expansion_completion, validate_expansion_format_audits
    def read(path):
        return json.loads(path.read_text())
    provenance = read(root / 'expansion-source-provenance.json')
    generation = read(MAIN / 'full-generation-report.json')
    main_manifest = (MAIN / 'generation-manifest.json').read_bytes()
    if generation['manifest'] != json.loads(main_manifest):
        raise ValueError('Main report/manifest mismatch')
    if generation['manifest'].get('pubmedqa_split', {}).get('train_rows') != 450:
        raise ValueError('Expansion source split provenance is stale')
    sources = [r['source'] for r in provenance['sources']]
    if len(sources) != 15 or sources.count('multidoc2dial') != 1:
        raise ValueError('Unexpected release source set')
    original_reports = [read(MAIN / (s + '.generation.json')) for s in sources]
    validate_expansion_completion(generation, original_reports, provenance)
    repaired = read(REPAIR / 'full-generation-report.json')
    repair_manifest = (REPAIR / 'generation-manifest.json').read_bytes()
    if repaired['manifest'] != json.loads(repair_manifest):
        raise ValueError('Corrective report/manifest mismatch')
    repair_report = read(REPAIR / 'multidoc2dial.generation.json')
    repair_audit = read(REPAIR / 'format-audit/multidoc2dial.json')
    repair_review = read(REPAIR / 'release-review.json')
    digest = hashlib.sha256(); ids = set()
    with (REPAIR / 'multidoc2dial.accepted.jsonl').open('rb') as stream:
        for line in stream:
            digest.update(line); row = json.loads(line)
            if row['task_id'] in ids or row['verification']['accepted'] is not True:
                raise ValueError('Duplicate or unverified corrective row')
            ids.add(row['task_id'])
    if digest.hexdigest() != repair_audit['accepted_file_sha256']:
        raise ValueError('Corrective accepted file changed after audit')
    repair_sha = hashlib.sha256(repair_manifest).hexdigest()
    validate_replacement(repaired, repair_report, repair_audit, repair_review, repair_sha, ids)
    reports = []; audits = []; inputs = []; manifests = {}; entries = []; manual_reviews = []
    for original in original_reports:
        source = original['source']; corrected = source == 'multidoc2dial'
        directory = REPAIR if corrected else MAIN
        report = repair_report if corrected else original
        audit = repair_audit if corrected else read(root / 'full-expansion-format-audit' / (source + '.json'))
        if not corrected:
            review_root = root / 'full-expansion-manual-review-samples'
            source_review = read(review_root / (source + '.review.json'))
            validate_source_sample_review(source, audit, read(review_root / (source + '.json')), source_review)
            manual_reviews.append(source_review)
        manifests[source] = repair_sha if corrected else hashlib.sha256(main_manifest).hexdigest()
        path = directory / (source + '.accepted.jsonl')
        reports.append(report); audits.append(audit); inputs.append(path)
        entries.append({'source': source, 'path': str(path), 'replaces_old_source': corrected,
                        'generation_manifest_file_sha256': manifests[source],
                        'accepted_file_sha256': audit['accepted_file_sha256'],
                        'source_report': report})
    validate_expansion_format_audits(audits, reports, manifests)
    totals = Counter()
    for report in reports:
        totals.update(report['reasons'])
    selected_generation = {'status': 'complete', 'reasons': dict(totals)}
    counts = validate_expansion_completion(selected_generation, reports, provenance)
    selection = {'version': 'v6-with-chronological-md2d-replacement-v1', 'sources': entries,
                 'counts': counts, 'corrective_input_provenance': repaired['manifest']['input_provenance'],
                 'corrective_release_review': repair_review,
                 'source_sample_reviews': manual_reviews,
                 'original_main_counts': validate_expansion_completion(generation, original_reports, provenance)}
    return {'selection': selection, 'selection_sha256': selection_digest(selection),
            'inputs': inputs, 'format_audits': audits, 'source_reports': reports,
            'manifest_hashes': manifests, 'counts': counts, 'provenance': provenance}


def validate_transport_selection(transport, selected):
    if (transport.get('expansion_selection') != selected['selection']
            or transport.get('expansion_selection_sha256') != selected['selection_sha256']
            or transport.get('native_jsonl_inputs') != [str(p) for p in selected['inputs']]
            or transport.get('rows') != selected['counts']['accepted']
            or transport.get('format_audits') != selected['format_audits']):
        raise ValueError('Expansion export uses a stale or additive source selection')


def verify_selected_files(selected):
    """Recheck files about to be uploaded, not merely their saved audit reports."""
    for entry in selected['selection']['sources']:
        digest = hashlib.sha256()
        with Path(entry['path']).open('rb') as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                digest.update(block)
        if digest.hexdigest() != entry['accepted_file_sha256']:
            raise ValueError('Selected expansion source changed after audit: ' + entry['source'])
