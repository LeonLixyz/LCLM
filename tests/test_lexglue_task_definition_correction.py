import copy
import hashlib
import json

import pytest

from data.lexglue_task_definition_correction import (
    CONFIGS, DEFINITIONS_PATH, DEFINITIONS_SHA256, VERSION, amend_lexglue_task,
    config_from_source_row_id, corrected_question, load_definitions,
)
from data.expansion_task_normalization import prepare_teacher_task


def fixture_task(config):
    labels = [r['label'] for r in load_definitions()['configs'][config]['labels']]
    question = (f'Classify this document for {config}. Choose labels from: ' + ', '.join(labels)
                + '. Return labels in the listed order, separated by |; use none when no label applies.')
    if config == 'case_hold':
        question = 'Which holding best completes the case? Return its zero-based index.\n' + '\n'.join(f'{i}: Choice {i}' for i in range(5))
    question = f'Use these source documents: {config}-9.\n' + question + '\nReturn exactly FINAL: followed by your answer.'
    return {'family': 'lex_glue', 'source_dataset': 'coastalcph/lex_glue', 'source_row_id': config+'-9',
            'task_id': 'original-opaque-id', 'question': question, 'raw_question': question,
            'gold_answer': 'GOLD-SENTINEL', 'expected_final': 'SECRET-EXPECTED',
            'support_segment_ids': ['seg_1'], 'tools': [{'original': True}],
            '_attempt_provenance': {'kind': 'retry_rejected'},
            'segments': [{'segment_id': 'seg_1', 'text': 'Full evidence unchanged', 'summary': 'Routing clue', 'record_id': config+'-9'}]}


def test_frozen_mapping_has_exact_feature_ontologies():
    data = load_definitions()
    assert hashlib.sha256(DEFINITIONS_PATH.read_bytes()).hexdigest() == DEFINITIONS_SHA256
    assert len(data['configs']['eurlex']['labels']) == 100
    eur = {r['label']: r['meaning'] for r in data['configs']['eurlex']['labels']}
    assert eur['100163'] == 'political framework'
    assert eur['100207'] == 'prices'
    assert eur['100245'] == 'agricultural policy'
    assert eur['100253'] == 'plant product'
    assert eur['100285'] == 'United Nations'
    scotus = {r['label']: r['meaning'] for r in data['configs']['scotus']['labels']}
    assert list(scotus) == [str(i) for i in range(1, 14)]
    assert scotus['9'] == 'Judicial Power'
    assert scotus['12'] == 'Federal Taxation'


@pytest.mark.parametrize('config', CONFIGS)
def test_gold_independence_preserved_evidence_and_identity(config):
    task = fixture_task(config)
    before = copy.deepcopy(task)
    out = amend_lexglue_task(task)
    assert task == before
    for key in ('task_id', 'source_row_id', 'gold_answer', 'expected_final', 'segments',
                'support_segment_ids', 'tools', '_attempt_provenance'):
        assert out[key] == task[key]
    other = copy.deepcopy(task)
    other['gold_answer'], other['expected_final'] = 'DIFFERENT-GOLD', 'DIFFERENT-EXPECTED'
    assert amend_lexglue_task(other)['question'] == out['question']
    assert 'GOLD-SENTINEL' not in out['question']
    if config == 'case_hold':
        assert out == task
    else:
        assert out['task_definition_correction']['version'] == VERSION
        assert out['task_definition_correction']['original_question'] == task['question']
        assert out['raw_question'] == out['question']
        assert out['training_user_prompt'] == out['user_prompt']
        assert out['training_user_prompt'].endswith(out['question'])
        assert out['rollout_user_prompt'].endswith(out['question'])
        assert 'Full evidence unchanged' not in out['rollout_user_prompt']
        assert 'Routing clue' in out['rollout_user_prompt']


@pytest.mark.parametrize('config', CONFIGS[1:])
def test_matches_normalizer_in_either_order_and_is_idempotent(config):
    task = fixture_task(config)
    a = prepare_teacher_task(amend_lexglue_task(task))
    b = amend_lexglue_task(prepare_teacher_task(task))
    assert a == b
    assert amend_lexglue_task(b) == b


def test_target_distinction_and_cardinality():
    a = amend_lexglue_task(fixture_task('ecthr_a'))['question']
    b = amend_lexglue_task(fixture_task('ecthr_b'))['question']
    assert 'Court found were violated' in a
    assert 'alleged to have been violated and considered' in b
    assert 'no violation was ultimately found' in b
    for config in ['ledgar', 'scotus']:
        q = amend_lexglue_task(fixture_task(config))['question']
        assert 'exactly one label' in q and 'use none' not in q
    assert 'main topic or theme' in amend_lexglue_task(fixture_task('ledgar'))['question']
    unfair = amend_lexglue_task(fixture_task('unfair_tos'))['question']
    assert 'actual stipulation and its exceptions' in unfair
    assert 'Fully optional arbitration can be fair' in unfair
    assert 'category mention alone' in unfair


@pytest.mark.parametrize('bad', [None, '', 'eurlex', 'eurlex-x', 'eurlex--1', 'eurlex-01', 'unknown-2', 'scotus-1\n'])
def test_malformed_source_ids_fail(bad):
    with pytest.raises(ValueError):
        config_from_source_row_id(bad)


def test_altered_ontology_and_raw_question_fail():
    task = fixture_task('scotus')
    task['question'] = task['question'].replace('12, 13.', '12, 13, 14.')
    task['raw_question'] = task['question']
    with pytest.raises(ValueError, match='ontology mismatch'):
        amend_lexglue_task(task)
    task = fixture_task('scotus')
    task['raw_question'] += 'different'
    with pytest.raises(ValueError, match='raw_question'):
        amend_lexglue_task(task)


@pytest.mark.parametrize('key', ['question', 'raw_question', 'training_user_prompt', 'user_prompt', 'rollout_user_prompt'])
def test_idempotent_reuse_rejects_changed_prompt(key):
    out = amend_lexglue_task(fixture_task('eurlex'))
    out[key] += 'drift'
    with pytest.raises(ValueError):
        amend_lexglue_task(out)


def test_changed_bundle_fails(tmp_path):
    path = tmp_path/'changed.json'
    path.write_bytes(DEFINITIONS_PATH.read_bytes()+b' ')
    with pytest.raises(ValueError, match='hash mismatch'):
        amend_lexglue_task(fixture_task('eurlex'), definitions_path=path)


def test_wrong_family_fails():
    task = fixture_task('eurlex')
    task['family'] = 'maud'
    with pytest.raises(ValueError):
        amend_lexglue_task(task)
