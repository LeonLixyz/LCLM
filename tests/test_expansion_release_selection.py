import hashlib
import json
from copy import deepcopy

import pytest

from data import expansion_release_selection as selection
from data.stage3_release_checks import validate_expansion_format_audits
from data.stage3_tokenizers import DECODER_REVISION, ENCODER_REVISION


def fixture():
    reasons = {'accepted:qwen_semantic': 2, 'wrong_answer': 21449}
    manifest = {'pilot_limit': None, 'model': 'Qwen/Qwen3-235B-A22B-Instruct-2507',
                'model_revision': 'ac9c66cc9b46af7306746a9250f23d47083d689e',
                'teacher_prompt_saved_in_training_messages': False,
                'semantic_review': 'question-first-every-claim-v1',
                'input_provenance': {'source': 'multidoc2dial',
                    'question_rendering_version': 'multidoc2dial-chronological-v1',
                    'source_revision': selection.SOURCE_REVISION,
                    'tasks_sha256': selection.TASK_SHA, 'parent_tasks_sha256': selection.PARENT_SHA}}
    generation = {'status': 'complete', 'manifest': manifest, 'reasons': reasons.copy()}
    report = {'status': 'complete', 'source': 'multidoc2dial', 'reasons': reasons.copy()}
    audit = {'source': 'multidoc2dial', 'status': 'passed', 'rows': 2, 'expected_accepted': 2,
             'counts': {'accepted': 2}, 'failed_rows': 0, 'errors': [],
             'generation_manifest_file_sha256': 'a'*64, 'accepted_file_sha256': 'b'*64,
             'decoder_tokenizer_revision': DECODER_REVISION, 'encoder_tokenizer_revision': ENCODER_REVISION,
             'minimum_segment_tokens': 512}
    ids = {'rea4-md2d-one', 'rea4-md2d-two'}
    review = {'source': 'multidoc2dial', 'approved_for_release': True, 'attempted_traces': 21451,
              'accepted_traces': 2, 'corrected_tasks_sha256': selection.TASK_SHA,
              'generation_manifest_file_sha256': 'a'*64, 'accepted_file_sha256': 'b'*64,
              'reviewed_task_ids': sorted(ids)}
    return generation, report, audit, review, 'a'*64, ids


def test_full_review_replaces_source_with_same_task_count():
    assert selection.validate_replacement(*fixture()) == {'tasks': 21451, 'accepted': 2, 'rejected': 21449, 'sources': 1}


@pytest.mark.parametrize('failure', ['pilot', 'partial', 'wrong_count', 'wrong_parent', 'wrong_tasks',
    'wrong_model', 'old_ids', 'pilot_approval_only', 'stale_manifest', 'stale_accepted',
    'missing_review', 'unknown_review_id', 'duplicate_review_ids', 'failed_audit'])
def test_replacement_fails_closed(failure):
    g, r, a, v, h, ids = fixture()
    if failure == 'pilot': g['manifest']['pilot_limit'] = 32
    elif failure == 'partial': g['status'] = 'partial'
    elif failure == 'wrong_count': r['reasons']['wrong_answer'] -= 1
    elif failure == 'wrong_parent': g['manifest']['input_provenance']['parent_tasks_sha256'] = 'x'*64
    elif failure == 'wrong_tasks': g['manifest']['input_provenance']['tasks_sha256'] = 'x'*64
    elif failure == 'wrong_model': g['manifest']['model_revision'] = 'main'
    elif failure == 'old_ids': ids = {'rea3-one', 'rea3-two'}
    elif failure == 'pilot_approval_only': v['approved_for_release'] = False; v['approved_for_full_generation'] = True
    elif failure == 'stale_manifest': v['generation_manifest_file_sha256'] = 'c'*64
    elif failure == 'stale_accepted': v['accepted_file_sha256'] = 'c'*64
    elif failure == 'missing_review': v = {}
    elif failure == 'unknown_review_id': v['reviewed_task_ids'][0] = 'rea4-md2d-unknown'
    elif failure == 'duplicate_review_ids': v['reviewed_task_ids'] = ['rea4-md2d-one']*2
    else: a['status'] = 'failed'
    with pytest.raises(ValueError): selection.validate_replacement(g, r, a, v, h, ids)


def test_audits_bind_each_source_to_its_own_generation():
    _, report, audit, _, _, _ = fixture()
    other = deepcopy(audit); other.update(source='other', generation_manifest_file_sha256='c'*64)
    reports = [report, {**report, 'source': 'other'}]
    hashes = {'multidoc2dial': 'a'*64, 'other': 'c'*64}
    validate_expansion_format_audits([audit, other], reports, hashes)
    for bad in ({'multidoc2dial': 'a'*64}, dict.fromkeys(hashes, 'a'*64)):
        with pytest.raises(ValueError): validate_expansion_format_audits([audit, other], reports, bad)


def test_publication_rechecks_actual_source_bytes(tmp_path):
    path = tmp_path/'accepted.jsonl'; path.write_bytes(b'original\n')
    selected = {'selection': {'sources': [{'source': 'example', 'path': str(path),
                  'accepted_file_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}]}}
    selection.verify_selected_files(selected)
    path.write_bytes(b'changed\n')
    with pytest.raises(ValueError, match='changed after audit'): selection.verify_selected_files(selected)


@pytest.mark.parametrize('failure', [None, 'failed_status', 'failed_example', 'missing', 'stale', 'different_id'])
def test_manual_review_is_required_even_after_format_pass(failure):
    _, _, audit, _, _, _ = fixture()
    sample = {'source': 'multidoc2dial', 'accepted_file_sha256': audit['accepted_file_sha256'],
              'examples': [{'category': 'single', 'training_row': {'task_id': 'example'}}]}
    review = {'source': 'multidoc2dial', 'status': 'sample_review_passed',
              'accepted_file_sha256': audit['accepted_file_sha256'],
              'reviewed_examples': [{'task_id': 'example', 'category': 'single', 'result': 'pass'}]}
    if failure == 'failed_status': review['status'] = 'sample_review_failed'
    elif failure == 'failed_example': review['reviewed_examples'][0]['result'] = 'fail'
    elif failure == 'missing': review = {}
    elif failure == 'stale': review['accepted_file_sha256'] = 'c'*64
    elif failure == 'different_id': review['reviewed_examples'][0]['task_id'] = 'other'
    if failure:
        with pytest.raises(ValueError): selection.validate_source_sample_review('multidoc2dial', audit, sample, review)
    else: selection.validate_source_sample_review('multidoc2dial', audit, sample, review)


def test_changed_release_entrypoints_parse():
    import ast
    from pathlib import Path
    root = Path(__file__).parents[1]/'data'
    for name in ['export_stage3_agent_transport_modal.py', 'publish_stage3_release_modal.py',
                 'stage3_pack_release_modal.py']:
        ast.parse((root/name).read_text())


def test_selection_loader_replaces_instead_of_appending(tmp_path, monkeypatch):
    main = tmp_path/'main'; repair = tmp_path/'repair'; root = tmp_path/'release'
    monkeypatch.setattr(selection, 'MAIN', main); monkeypatch.setattr(selection, 'REPAIR', repair)
    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
    g, r, a, v, _, ids = fixture()
    write(repair/'generation-manifest.json', g['manifest'])
    digest = hashlib.sha256((repair/'generation-manifest.json').read_bytes()).hexdigest()
    a['generation_manifest_file_sha256'] = v['generation_manifest_file_sha256'] = digest
    accepted = repair/'multidoc2dial.accepted.jsonl'
    accepted.write_text(''.join(json.dumps({'task_id': i, 'verification': {'accepted': True}})+'\n' for i in sorted(ids)))
    a['accepted_file_sha256'] = v['accepted_file_sha256'] = hashlib.sha256(accepted.read_bytes()).hexdigest()
    for path, value in [('full-generation-report.json', g), ('multidoc2dial.generation.json', r),
                        ('format-audit/multidoc2dial.json', a), ('release-review.json', v)]:
        write(repair/path, value)
    main_manifest = {'pubmedqa_split': {'train_rows': 450}}
    write(main/'generation-manifest.json', main_manifest)
    main_digest = hashlib.sha256((main/'generation-manifest.json').read_bytes()).hexdigest()
    sources = [{'source': 'multidoc2dial', 'tasks': 21451}]
    old = deepcopy(r); old['reasons'] = {'accepted:qwen_semantic': 3, 'wrong_answer': 21448}
    write(main/'multidoc2dial.generation.json', old)
    for i in range(14):
        source = 'source'+str(i); sources.append({'source': source, 'tasks': 2})
        write(main/(source+'.generation.json'), {'source': source, 'status': 'complete', 'reasons': {'accepted:qwen_semantic': 2}})
        write(root/'full-expansion-format-audit'/(source+'.json'), {**a, 'source': source,
              'generation_manifest_file_sha256': main_digest})
        sample_root = root/'full-expansion-manual-review-samples'
        write(sample_root/(source+'.json'), {'source': source, 'accepted_file_sha256': a['accepted_file_sha256'],
              'examples': [{'category': 'single', 'training_row': {'task_id': source+'-example'}}]})
        write(sample_root/(source+'.review.json'), {'source': source, 'status': 'sample_review_passed',
              'accepted_file_sha256': a['accepted_file_sha256'],
              'reviewed_examples': [{'task_id': source+'-example', 'category': 'single', 'result': 'pass'}]})
    write(main/'full-generation-report.json', {'status': 'complete', 'manifest': main_manifest,
          'reasons': {'accepted:qwen_semantic': 31, 'wrong_answer': 21448}})
    write(root/'expansion-source-provenance.json', {'status': 'assembled', 'tasks': 21479, 'sources': sources})
    result = selection.load_selection(root)
    assert result['counts']['accepted'] == 30  # NOT 31 + 2!
    assert len(result['inputs']) == 15
    assert result['inputs'][0] == accepted
    assert main/'multidoc2dial.accepted.jsonl' not in result['inputs']
    assert result['selection']['original_main_counts']['accepted'] == 31
    transport = {'rows': 30, 'native_jsonl_inputs': [str(p) for p in result['inputs']],
                 'format_audits': result['format_audits'], 'expansion_selection': result['selection'],
                 'expansion_selection_sha256': result['selection_sha256']}
    selection.validate_transport_selection(transport, result)
    for key, value in [('rows', 33), ('native_jsonl_inputs', transport['native_jsonl_inputs']+[str(main/'multidoc2dial.accepted.jsonl')]),
                       ('expansion_selection_sha256', 'stale')]:
        with pytest.raises(ValueError): selection.validate_transport_selection({**transport, key: value}, result)
    accepted.write_text(accepted.read_text()+accepted.read_text())
    with pytest.raises(ValueError, match='Duplicate'): selection.load_selection(root)
