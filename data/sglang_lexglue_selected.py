"""Explicit config subsets of immutable corrected v4 LexGLUE tasks; no dispatch."""
from __future__ import annotations
import copy
import hashlib
import json
import sqlite3
import tempfile
from collections import Counter
from pathlib import Path
from data.sglang_backlog import atomic_json,file_sha,sha
from data.sglang_lexglue_defined import generation_config as parent_config,ONTOLOGY_PATH
from data.lexglue_task_definition_correction import CONFIGS,VERSION,DEFINITIONS_SHA256,config_from_source_row_id
from data.export_reviewed_sglang27 import frozen_bytes,require

PARENT_ROOT=Path('/runs/stage3-build-20260906/sglang27-lexglue-defined-20260909-v4')
ROOT=Path('/runs/stage3-build-20260906/sglang27-lexglue-selected-20260909-v5')
PARENT_MANIFEST_SHA256='c90e0f49da167c97ccba5d9ceac1f80d659d0661f0785e2828d5a0c7a1c5f860'
PARENT_PROBE_DECISION_SHA256='bdcaa350a220b8909b6290773581007e7689532d536fb4c7df7e9982a0b14861'
SELECTION_SCHEMA='reviewed-corrected-lexglue-configs-v1'


def document(spec):return json.loads(frozen_bytes(spec))


def optional_progress(path):
    """Live progress is advisory; an incomplete Volume snapshot is not task failure."""
    try:
        value=json.loads(Path(path).read_text())
    except (FileNotFoundError,json.JSONDecodeError,UnicodeDecodeError):
        return None
    return value['counts'] if isinstance(value,dict) and isinstance(value.get('counts'),dict) else None


def generation_config():
    config=parent_config()
    config['lexglue_config_selection']={'version':SELECTION_SCHEMA,'all_canaries_diagnostic_only':True,
        'exact_v4_task_line_bytes':True,'held_configs_explicit':True,'no_new_probe_attempts':True,
        'serving_prompts_validators':'Unchanged from the corrected v4 task snapshot'}
    for name in ('sglang_lexglue_selected.py','sglang_lexglue_selected_modal.py'):
        config['code_sha256'][name]=file_sha(Path(__file__).parent/name)
    return config


def verify_settings(config):
    reduced=copy.deepcopy(config)
    require(reduced.pop('lexglue_config_selection',None)==generation_config()['lexglue_config_selection'],
            'Config selection semantics changed')
    for name in ('sglang_lexglue_selected.py','sglang_lexglue_selected_modal.py'):
        require(reduced['code_sha256'].pop(name,None)==generation_config()['code_sha256'][name],
                'Config successor code changed')
    require(reduced==parent_config(),'Serving/prompt/validator settings changed')


def validate_selection(spec):
    decision=document(spec);parent=document(decision['parent_manifest'])
    require(decision.get('schema')==SELECTION_SCHEMA and decision.get('decision')=='select_corrected_lexglue_configs'
        and decision.get('reviewed_by')=='root' and decision.get('approved_for_preparation') is True
        and decision.get('approved_for_generation') is True and decision.get('approved_for_release') is False
        and decision.get('scope')=='exact_unattempted_config_subset','Missing explicit root config selection')
    require(decision['parent_manifest']['sha256']==PARENT_MANIFEST_SHA256
        and parent['generation_config']==parent_config() and parent['task_version']==VERSION
        and parent['ontology_artifact']=={'path':ONTOLOGY_PATH,'sha256':DEFINITIONS_SHA256},
        'Corrected parent task snapshot changed')
    frozen_bytes(parent['ontology_artifact'])
    require(decision['generation_config']==generation_config(),'Selected generation code differs from review')
    allowed=decision.get('config_allowlist',[])
    require(bool(allowed) and allowed==[c for c in CONFIGS if c in allowed] and len(set(allowed))==len(allowed),
        'Config allowlist must be explicit, known, unique, and in canonical order')
    require(set(decision.get('config_reviews',{}))==set(CONFIGS),'Every config needs an explicit approved or held review')
    for config,review in decision['config_reviews'].items():
        require(type(review.get('approved_for_generation')) is bool
            and review['approved_for_generation']==(config in allowed)
            and str(review.get('evidence_review','')).strip(),'Missing config-specific root evidence or hold')
    require(bool(decision.get('review_artifacts')),'Config choice lacks source-review artifacts')
    for artifact in decision['review_artifacts']:frozen_bytes(artifact)
    probe_decision=document(decision['corrected_probe_decision']);report=document(decision['corrected_probe_report'])
    audit=document(decision['corrected_probe_audit']);source,=parent['sources'];probe=source['chunks'][0]
    require(decision['corrected_probe_decision']['sha256']==PARENT_PROBE_DECISION_SHA256
        and probe_decision['backlog_manifest_sha256']==PARENT_MANIFEST_SHA256
        and report['status']=='complete' and report['rows']==report['completed']==64
        and report['source']=='lex_glue' and report['chunk']==0
        and report['input_sha256']==probe['sha256'] and report['backlog_manifest_sha256']==PARENT_MANIFEST_SHA256
        and report['decision_sha256']==PARENT_PROBE_DECISION_SHA256
        and set(report['result_hashes'])==set(source['probe_ids']),'All64 corrected canaries must finish unchanged')
    require(audit['status']=='passed' and not audit.get('errors') and audit['task_version']==VERSION
        and audit['ontology_artifact']==parent['ontology_artifact']
        and audit['probe_decision_sha256']==PARENT_PROBE_DECISION_SHA256
        and audit['input_sha256']==probe['sha256'] and audit['exact_original_to_corrected_transforms']==64
        and audit['exact_corrected_teacher_requests_verified'] is True,'Missing full corrected-canary format audit')
    verify_settings(decision['generation_config'])
    return decision,parent


def exact_rows(spec):
    """Yield original nonblank line bytes after validating each bounded physical file."""
    raw=frozen_bytes(spec);lines=[line for line in raw.splitlines(keepends=True) if line.strip()]
    require(len(lines)<=512,'Unbounded corrected input chunk')
    rows=[(json.loads(line),line) for line in lines];ids=[row['task_id'] for row,_ in rows]
    require(not spec.get('exclude_task_ids') and len(ids)==len(set(ids))==spec['rows']
        and sha(json.dumps(sorted(ids)).encode())==spec['task_ids_sha256'],'Corrected parent task coverage changed')
    return rows


def prepare_chunk(spec,destination,*,config_allowlist,canary_ids,selection_spec):
    import uuid
    allowed=set(config_allowlist);canaries=set(canary_ids)
    require(allowed and allowed<=set(CONFIGS) and len(canaries)==64,'Invalid approved config/canary inventory')
    root=Path(destination);root.mkdir(parents=True,exist_ok=True)
    binding={'parent_chunk':spec,'config_allowlist':config_allowlist,'canary_ids_sha256':sha(json.dumps(sorted(canaries)).encode()),
             'selection':selection_spec,'generation_config':generation_config()}
    report_path=root/'report.json'
    if report_path.exists():
        report=json.loads(report_path.read_text());require(report['binding']==binding,'Completed subset preparation changed')
        frozen_bytes(report['tasks']);frozen_bytes(report['coverage']);return report
    attempt=root/('attempt-'+uuid.uuid4().hex);attempt.mkdir();tasks=attempt/'tasks.jsonl';coverage=attempt/'coverage.jsonl'
    counts={c:Counter() for c in CONFIGS};selected=[]
    with tasks.open('xb') as output,coverage.open('x') as ledger:
        for row,line in exact_rows(spec):
            key=row['task_id'];config=config_from_source_row_id(row['source_row_id'])
            disposition='diagnostic_canary' if key in canaries else 'selected' if config in allowed else 'held_config'
            require((spec['index']==0)==(key in canaries),'Canary leaked into a continuation input or probe set changed')
            counts[config][disposition]+=1
            if disposition=='selected':output.write(line);selected.append(key)
            ledger.write(json.dumps({'task_id':key,'config':config,'disposition':disposition,
                'parent_line_sha256':sha(line),'parent_input_sha256':spec['sha256']})+'\n')
    report={'status':'complete','binding':binding,'parent_index':spec['index'],'input_rows':spec['rows'],
        'counts':{c:dict(v) for c,v in counts.items()},'selected_rows':len(selected),
        'tasks':{'path':str(tasks),'rows':len(selected),'sha256':file_sha(tasks),
            'task_ids_sha256':sha(json.dumps(sorted(selected)).encode())},
        'coverage':{'path':str(coverage),'rows':spec['rows'],'sha256':file_sha(coverage)},'approved_for_release':False}
    atomic_json(report_path,report);return report


def finalize_subset(parent,selection_spec,decision,reports,destination):
    root=Path(destination);root.mkdir(parents=True,exist_ok=True)
    source,=parent['sources'];require(len(reports)==len(source['chunks']),'Missing config partition reports')
    counts={c:Counter() for c in CONFIGS};chunks=[];artifacts=[]
    with tempfile.TemporaryDirectory(prefix='lex-config-coverage-') as temp:
        db=sqlite3.connect(str(Path(temp)/'ids.sqlite'));db.execute('PRAGMA cache_size=-1024')
        db.execute('CREATE TABLE ids (task_id TEXT PRIMARY KEY,config TEXT,disposition TEXT)')
        for original,report_spec in zip(source['chunks'],reports):
            report=document(report_spec);binding=report['binding']
            require(binding['parent_chunk']==original and binding['selection']==selection_spec
                and binding['config_allowlist']==decision['config_allowlist']
                and binding['generation_config']==generation_config(),'Subset input/approval binding changed')
            tasks_raw=frozen_bytes(report['tasks']);ledger_raw=frozen_bytes(report['coverage'])
            rows=[json.loads(line) for line in ledger_raw.splitlines() if line.strip()]
            require(len(rows)==original['rows']==report['input_rows'],'Incomplete subset disposition ledger')
            selected_ids=[];actual={c:Counter() for c in CONFIGS}
            for row in rows:
                key,config,kind=row['task_id'],row['config'],row['disposition']
                expected='diagnostic_canary' if key in source['probe_ids'] else 'selected' if config in decision['config_allowlist'] else 'held_config'
                require(kind==expected and row['parent_input_sha256']==original['sha256'],'Wrong config disposition')
                try:db.execute('INSERT INTO ids VALUES (?,?,?)',(key,config,kind))
                except sqlite3.IntegrityError as exc:raise ValueError('Duplicate selected/held/canary task ID') from exc
                actual[config][kind]+=1
                if kind=='selected':selected_ids.append(key)
            require({c:dict(v) for c,v in actual.items()}==report['counts'],'Subset config counts changed')
            for config in CONFIGS:counts[config].update(actual[config])
            physical=[(json.loads(line),line) for line in tasks_raw.splitlines(keepends=True) if line.strip()]
            require([row['task_id'] for row,_ in physical]==selected_ids and len(selected_ids)==report['selected_rows'],
                'Selected file differs from disposition ledger')
            line_hashes={row['task_id']:row['parent_line_sha256'] for row in rows if row['disposition']=='selected'}
            require(all(sha(line)==line_hashes[row['task_id']] for row,line in physical),'Selected corrected task bytes changed')
            require(report['tasks']['task_ids_sha256']==sha(json.dumps(sorted(selected_ids)).encode()),'Subset ID digest changed')
            if selected_ids:chunks.append({**report['tasks'],'index':len(chunks)+1,'kind':'continuation','parent_chunk':original})
            artifacts.append({**report_spec,'coverage':report['coverage']});db.commit()
        total=db.execute('SELECT COUNT(*) FROM ids').fetchone()[0]
        require(total==parent['original_pending_rows'] and total==sum(sum(c.values()) for c in counts.values()),'Original corrected ID coverage lost')
        require({c:sum(v.values()) for c,v in counts.items()}==parent['configs'],'Original corrected config totals changed')
        ledgers={};id_lists={}
        for kind in ('selected','held_config','diagnostic_canary'):
            keys=[r[0] for r in db.execute('SELECT task_id FROM ids WHERE disposition=? ORDER BY task_id',(kind,))]
            id_lists[kind]=keys;path=root/(kind+'-ids.jsonl')
            raw=b''.join((json.dumps({'task_id':key})+'\n').encode() for key in keys)
            if path.exists():require(path.read_bytes()==raw,'Completed config ID inventory changed')
            else:path.write_bytes(raw)
            ledgers[kind]={'path':str(path),'rows':len(keys),'sha256':file_sha(path),'task_ids_sha256':sha(json.dumps(keys).encode())}
        require(id_lists['diagnostic_canary']==sorted(source['probe_ids']) and len(id_lists['diagnostic_canary'])==64,
            'All64 original canaries must remain diagnostic only')
        require(id_lists['selected'] and sum(c['rows'] for c in chunks)==len(id_lists['selected']),'No selected continuation work')
        db.close()
    diagnostic={**source['chunks'][0],'kind':'diagnostic_only_excluded','rows':0,'exclude_task_ids':source['probe_ids'],
                'task_ids_sha256':sha(b'[]')}
    selected=len(id_lists['selected']);held=len(id_lists['held_config'])
    current={'source':'lex_glue','rows':selected,'origin_pending_rows':parent['original_pending_rows'],'probe_rows':0,'probe_ids':[],
        'chunks':[diagnostic,*chunks],'counts':{'previously_unattempted':selected},
        'probe_capacities':{c:parent['configs'][c] for c in CONFIGS},'selected_configs':decision['config_allowlist']}
    manifest={'status':'prepared_pending_exact_manifest_binding','task_version':VERSION,'ontology_artifact':parent['ontology_artifact'],
        'generation_config':generation_config(),'sources':[current],'pending_rows':selected,'original_pending_rows':parent['original_pending_rows'],
        'original_corrected_parent_rows':parent['original_pending_rows'],'probe_rows':0,
        'config_allowlist':decision['config_allowlist'],'held_configs':[c for c in CONFIGS if c not in decision['config_allowlist']],
        'config_selection':selection_spec,'parent_manifest':decision['parent_manifest'],
        'diagnostic_canary':{'rows':64,'included_in_generation':False,'included_in_training':False,
            'manifest':decision['parent_manifest'],'decision':decision['corrected_probe_decision'],
            'report':decision['corrected_probe_report'],'audit':decision['corrected_probe_audit']},
        'plan_accounting':{'selected_unattempted':selected,'held_config_unattempted':held,'diagnostic_canaries':64,
            'original_corrected_parent_rows':parent['original_pending_rows'],'config_counts':{c:dict(v) for c,v in counts.items()},
            'id_inventories':ledgers,'partition_reports':artifacts,'held_is_outside_approved_finite_plan':True},
        'pending_counts':{'previously_unattempted':selected},'remaining_task_ids_sha256':ledgers['selected']['task_ids_sha256'],
        'maud_ontology_path':parent['maud_ontology_path'],'maud_ontology_sha256':parent['maud_ontology_sha256'],
        'approved_for_generation':False,'approved_for_release':False}
    require(selected+held+64==parent['original_pending_rows'],'Subset accounting does not equal original corrected source')
    return manifest


def validate_continuation(decision,manifest,*,manifest_sha,selection_sha):
    require(decision.get('decision')=='continue_corrected_lexglue_configs' and decision.get('reviewed_by')=='root'
        and decision.get('approved_for_generation') is True and decision.get('approved_for_release') is False
        and decision.get('scope')=='reviewed_config_subset' and decision.get('backlog_manifest_sha256')==manifest_sha
        and decision.get('config_selection_sha256')==selection_sha==manifest['config_selection']['sha256']
        and decision.get('config_allowlist')==manifest['config_allowlist']
        and decision.get('generation_config')==manifest['generation_config']
        and decision.get('task_version')==VERSION and decision.get('ontology_artifact')==manifest['ontology_artifact']
        and str(decision.get('evidence_review','')).strip(),'Missing root binding to exact config subset manifest')
    verify_settings(manifest['generation_config']);return {'lex_glue'}
