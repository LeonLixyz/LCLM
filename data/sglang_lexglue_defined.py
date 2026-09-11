"""Frozen corrected LexGLUE task chunks; original IDs/results remain unchanged."""
from __future__ import annotations
import copy
import json
from collections import Counter
from pathlib import Path
from data.sglang_backlog import atomic_json,file_sha,sha
from data.sglang_backlog_fast import checked_rows
from data.sglang_backlog_storage import generation_config as previous_config
from data.lexglue_task_definition_correction import (
    VERSION,DEFINITIONS_SHA256,DEFINITIONS_PATH,amend_lexglue_task,config_from_source_row_id)

EXPECTED_ROWS=186444
EXPECTED_PROBES=64
ONTOLOGY_PATH='/runs/stage3-build-20260906/reviewed-internal-packing-20260909-v1/lexglue_task_definitions.v1.json'
PARENT_MANIFEST_SHA256='96584acb7b8ba93e45b79310c9793408aa7d5b18176880136db3ef511b5a3cfe'


def generation_config():
    result=previous_config()
    result['storage_successor']['probe_resume']='No opaque LexGLUE probe is reused; corrected64 probe is a separate task-version attempt.'
    result['corrected_lexglue']={'task_version':VERSION,'definitions_sha256':DEFINITIONS_SHA256,
        'scope':'Exact186444 original pending LexGLUE IDs; opaque64 probes retained as diagnostics only',
        'ontology_artifact':{'path':ONTOLOGY_PATH,'sha256':DEFINITIONS_SHA256},
        'gold_fields_used_for_instruction':False,'teacher_training_question_consistent':True,
        'original_task_ids_preserved':True,'new_task_snapshot_required':True,
        'casehold':'Definitions already explicit; task content unchanged',
        'other_generation_settings':'Identical pinned27B nonthinking TP1x8 configuration and verification'}
    for name in ('sglang_lexglue_defined.py','sglang_lexglue_defined_modal.py','lexglue_task_definition_correction.py'):
        result['code_sha256'][name]=file_sha(Path(__file__).parent/name)
    return result


def verify_settings(config):
    reduced=copy.deepcopy(config)
    correction=reduced.pop('corrected_lexglue',None)
    if not correction or correction['task_version']!=VERSION or correction['definitions_sha256']!=DEFINITIONS_SHA256:
        raise ValueError('Corrected LexGLUE version/definitions changed')
    for name in ('sglang_lexglue_defined.py','sglang_lexglue_defined_modal.py','lexglue_task_definition_correction.py'):
        if not reduced['code_sha256'].pop(name,None):raise ValueError('Missing correction code hash')
    reduced['storage_successor']['probe_resume']=previous_config()['storage_successor']['probe_resume']
    if reduced!=previous_config():raise ValueError('Serving, model, teacher guidance or validators changed')


def prepare_chunk(spec,destination,*,definitions_path=DEFINITIONS_PATH):
    """Bounded chunk rewrite; publish completion only after original hash exhaustion."""
    import uuid
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    report_path=destination/'report.json'
    binding={'parent_chunk':spec,'generation_config':generation_config()}
    if report_path.exists():
        report=json.loads(report_path.read_text())
        if report['binding']!=binding or file_sha(report['chunk']['path'])!=report['chunk']['sha256']:
            raise ValueError('Completed corrected input changed')
        return report
    attempt=destination/('attempt-'+uuid.uuid4().hex);attempt.mkdir()
    path=attempt/'tasks.jsonl';ledger=attempt/'coverage.jsonl'
    counts=Counter();ids=[];modified=0
    with path.open('xb') as output,ledger.open('x') as coverage:
        for original in checked_rows(spec):
            task=amend_lexglue_task(original,definitions_path=definitions_path)
            key=task['task_id'];config=config_from_source_row_id(task['source_row_id'])
            for field in original:
                if field not in ('question','raw_question','training_user_prompt','user_prompt','rollout_user_prompt','task_definition_correction'):
                    if task[field]!=original[field]:raise ValueError('Correction changed evidence/gold/identity/provenance')
            ids.append(key);counts[config]+=1;modified+=task!=original
            output.write((json.dumps(task,ensure_ascii=False)+'\n').encode())
            coverage.write(json.dumps({'task_id':key,'config':config,'modified':task!=original,
                'original_task_sha256':sha(json.dumps(original,sort_keys=True).encode()),
                'corrected_task_sha256':sha(json.dumps(task,sort_keys=True).encode())})+'\n')
    if len(ids)!=len(set(ids)) or len(ids)!=spec['rows']:raise ValueError('Corrected chunk ID/count changed')
    chunk={key:copy.deepcopy(value) for key,value in spec.items() if key not in ('path','sha256','exclude_task_ids')}
    chunk.update(path=str(path),sha256=file_sha(path),rows=len(ids),task_ids_sha256=sha(json.dumps(sorted(ids)).encode()))
    if chunk['task_ids_sha256']!=spec['task_ids_sha256']:raise ValueError('Correction changed effective task IDs')
    report={'status':'complete','binding':binding,'chunk':chunk,'rows':len(ids),'modified':modified,'configs':dict(counts),
        'coverage':{'path':str(ledger),'rows':len(ids),'sha256':file_sha(ledger)},'task_ids':ids,'approved_for_release':False}
    atomic_json(report_path,report)
    return report


def validate_decision(decision,manifest,*,stage,manifest_sha,probe_decision_sha=None,probe_report=None,probe_report_sha=None):
    expected='generate_corrected_lexglue_probe' if stage=='probe' else 'continue_corrected_lexglue'
    if (stage not in ('probe','continuation') or decision.get('decision')!=expected or decision.get('reviewed_by')!='root'
        or decision.get('approved_for_generation') is not True or decision.get('approved_for_release') is not False
        or decision.get('backlog_manifest_sha256')!=manifest_sha or decision.get('generation_config')!=manifest['generation_config']
        or decision.get('task_version')!=VERSION or decision.get('ontology_artifact')!=manifest['ontology_artifact']
        or not str(decision.get('evidence_review','')).strip()):
        raise ValueError('Missing corrected-task root review/configuration binding')
    verify_settings(manifest['generation_config'])
    if stage=='probe':
        if decision.get('scope')!='source_probes_only' or decision.get('probe_sources')!=['lex_glue']:
            raise ValueError('Corrected probe scope must be exactly LexGLUE64')
    else:
        source=manifest['sources'][0];review=decision.get('source_reviews',{}).get('lex_glue',{})
        if (decision.get('scope')!='reviewed_sources' or set(decision.get('source_reviews',{}))!={'lex_glue'}
            or decision.get('probe_decision_sha256')!=probe_decision_sha or not probe_decision_sha
            or review.get('approved_for_generation') is not True or not str(review.get('evidence_review','')).strip()
            or not probe_report or review.get('probe_report_sha256')!=probe_report_sha
            or probe_report.get('status')!='complete' or probe_report.get('rows')!=EXPECTED_PROBES
            or probe_report.get('completed')!=EXPECTED_PROBES or probe_report.get('source')!='lex_glue'
            or probe_report.get('chunk')!=0 or probe_report.get('backlog_manifest_sha256')!=manifest_sha
            or probe_report.get('decision_sha256')!=probe_decision_sha
            or probe_report.get('input_sha256')!=source['chunks'][0]['sha256']):
            raise ValueError('Corrected LexGLUE continuation lacks completed source review')
    return {'lex_glue'}
