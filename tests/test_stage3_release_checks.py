from copy import deepcopy
import pytest
from data.stage3_release_checks import validate_base_recovery
from data.stage3_release_checks import validate_expansion_completion
from data.stage3_release_checks import validate_pilot_review
from data.stage3_release_checks import validate_final_artifact_audit
from data.stage3_release_checks import validate_expansion_format_audits


def format_audits():
    from data.stage3_tokenizers import DECODER_REVISION, ENCODER_REVISION
    return ([{'source':'source','status':'passed','rows':2,'expected_accepted':2,
        'counts':{'accepted':2},'failed_rows':0,'errors':[],
        'generation_manifest_file_sha256':'a'*64,'accepted_file_sha256':'b'*64,
        'decoder_tokenizer_revision':DECODER_REVISION,'encoder_tokenizer_revision':ENCODER_REVISION,
        'minimum_segment_tokens':512}], [{'source':'source','reasons':{'accepted:exact':2}}], 'a'*64)


def test_full_format_audit_binds_counts_manifest_and_tokenizers():
    assert set(validate_expansion_format_audits(*format_audits())) == {'source'}


@pytest.mark.parametrize('field,value', [('status','failed'),('rows',1),('expected_accepted',3),
    ('failed_rows',1),('errors',[{'error':'bad row'}]),('generation_manifest_file_sha256','stale'),
    ('decoder_tokenizer_revision','main'),('encoder_tokenizer_revision','main'),
    ('accepted_file_sha256','missing'),('minimum_segment_tokens',511)])
def test_full_format_audit_rejects_stale_or_partial_reports(field,value):
    audits,sources,digest=format_audits();audits[0][field]=value
    with pytest.raises(ValueError):validate_expansion_format_audits(audits,sources,digest)


def test_full_format_audit_requires_all_unique_sources():
    audits,sources,digest=format_audits()
    for bad in ([],audits*2):
        with pytest.raises(ValueError):validate_expansion_format_audits(bad,sources,digest)


def test_final_artifact_audit_requires_expansion():
    validate_final_artifact_audit({'status':'passed',
        'components':['base','agents','base_recovery','expansion']})


@pytest.mark.parametrize('components', [[], ['base','agents','base_recovery'],
    ['base','agents','base_recovery','expansion','expansion']])
def test_partial_or_duplicate_artifact_audit_cannot_publish(components):
    with pytest.raises(ValueError):
        validate_final_artifact_audit({'status':'passed','components':components})


def test_failed_artifact_audit_cannot_publish():
    with pytest.raises(ValueError):
        validate_final_artifact_audit({'status':'failed',
            'components':['base','agents','base_recovery','expansion']})


def reports():
    return ([{'partition': 0, 'counts': dict(input_rows=10, packed_rows=5,
        rejected_processing=3, over_32768=2)}],
        [{'partition': 0, 'counts': dict(scanned=10, candidates=5,
            recovered_rows=1, packed_rows=1, over_32768=1, not_recoverable=3)}])


def test_recovery_counts_input_only_once_and_reclassifies_skips():
    summary = validate_base_recovery(*reports(), partitions=1)
    assert summary == dict(input_rows=10, packed_rows=6, recovered_rows=1,
        rejected_processing=1, over_32768=3, under_18=0)


@pytest.mark.parametrize('key,value', [('scanned', 9), ('packed_rows', 0),
    ('recovered_rows', 4), ('candidates', 6)])
def test_recovery_rejects_inconsistent_counts(key, value):
    base, recovery = deepcopy(reports())
    recovery[0]['counts'][key] = value
    with pytest.raises(ValueError):
        validate_base_recovery(base, recovery, partitions=1)


def test_recovery_requires_complete_unique_partitions():
    base, recovery = reports()
    with pytest.raises(ValueError):
        validate_base_recovery(base, recovery, partitions=2)
    with pytest.raises(ValueError):
        validate_base_recovery(base, recovery*2, partitions=1)

def expansion_reports():
    reasons={'accepted:exact':2,'wrong_answer':1}
    return ({'status':'complete','reasons':reasons},
            [{'source':'example','status':'complete','reasons':reasons.copy()}],
            {'status':'assembled','tasks':3,'sources':[{'source':'example','tasks':3}]})

def test_expansion_requires_every_task_accounted_for():
    assert validate_expansion_completion(*expansion_reports()) == {
        'tasks':3,'accepted':2,'rejected':1,'sources':1}

@pytest.mark.parametrize('failure',['partial','missing_source','duplicate','wrong_total','wrong_attempts','negative'])
def test_incomplete_expansion_is_not_publishable(failure):
    generation,sources,provenance=expansion_reports()
    if failure=='partial':generation['status']='partial'
    elif failure=='missing_source':sources=[]
    elif failure=='duplicate':sources=sources*2
    elif failure=='wrong_total':generation['reasons']={'accepted:exact':3}
    elif failure=='wrong_attempts':provenance['sources'][0]['tasks']=4
    else:sources[0]['reasons']['wrong_answer']=-1
    with pytest.raises(ValueError):validate_expansion_completion(generation,sources,provenance)

def pilot_reports():
    import hashlib,json
    manifest={'model':'pinned'}
    digest=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()
    return ({'status':'complete','manifest':manifest,'reasons':{'accepted':2}},
            {'status':'passed','accepted_traces':2,'generation_manifest_sha256':digest,'sources':{'source':{'accepted':2}}},
            {'approved':True,'accepted_traces':2,'generation_manifest_sha256':digest,'reviewed_examples':{'source':['one','two']}})

def test_review_is_bound_to_pilot_manifest():
    reports=pilot_reports()
    assert validate_pilot_review(*reports)==reports[2]['generation_manifest_sha256']

@pytest.mark.parametrize('failure',['stale','count','samples','approval'])
def test_bad_pilot_review_fails_closed(failure):
    generation,audit,review=pilot_reports()
    if failure=='stale':review['generation_manifest_sha256']='different'
    elif failure=='count':review['accepted_traces']=3
    elif failure=='samples':review['reviewed_examples']={}
    else:review['approved']=False
    with pytest.raises(ValueError):validate_pilot_review(generation,audit,review)
