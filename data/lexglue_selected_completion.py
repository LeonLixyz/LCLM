"""Corrected-Lex private completion helpers, with no source approval or dispatch."""
from __future__ import annotations
import json
import sqlite3
import tempfile
from collections import Counter
from pathlib import Path
from data import expansion_completion_chain as chain
from data import export_reviewed_sglang27 as core
from data.lexglue_task_definition_correction import VERSION,DEFINITIONS_SHA256
from data.sglang_lexglue_selected import generation_config as generator_config,ONTOLOGY_PATH,validate_selection
from data.expansion_training_quality import inspect_expansion_efficiency

GENERATION_ROOT=Path('/runs/stage3-build-20260906/sglang27-lexglue-selected-20260909-v5')


def configuration():
    names=('lexglue_selected_completion.py','lexglue_selected_completion_modal.py','expansion_trace_audit.py',
           'audit_qwen_agent_format.py','synthetic_expansion_agent.py','real_expansion_agent.py')
    return {'version':'selected-corrected-lexglue-private-completion-v1','cpu_only':True,
        'generation_config':generator_config(),'chain_configuration':chain.config(),
        'code_sha256':{name:chain.file_sha(Path(__file__).parent/name) for name in names}}


def governing(policy_path,policy_sha,review_spec,manifest_spec):
    policy,policy_spec=chain.load_policy(policy_path,policy_sha)
    manifest=chain.document(manifest_spec);review=chain.document(review_spec)
    expected={'path':ONTOLOGY_PATH,'sha256':DEFINITIONS_SHA256}
    dependencies=[d for d in policy.get('additional_readiness_dependencies',[]) if d['source_allowlist']==['lex_glue']]
    core.require(len(dependencies)==1,'Primary chain must have exactly one corrected-Lex readiness dependency')
    dependency=dependencies[0]
    core.require(dependency['task_version']==VERSION and dependency['ontology_artifact']==expected
        and manifest['task_version']==VERSION and manifest['ontology_artifact']==expected
        and manifest['generation_config']==generator_config(),'Corrected task/ontology/configuration changed')
    core.require(review.get('reviewed_by')=='root' and review.get('decision')=='continue_corrected_lexglue_configs'
        and review.get('approved_for_generation') is True and review.get('approved_for_internal_packing') is True
        and review.get('approved_for_release') is False and review.get('source_allowlist')==['lex_glue']
        and review.get('generation_manifest_sha256')==manifest_spec['sha256']
        and review.get('backlog_manifest_sha256')==manifest_spec['sha256']
        and review.get('ontology_artifact')==expected and review.get('task_version')==VERSION,
        'Actual corrected continuation review must authorize internal packing')
    core.require(review.get('completion_configuration')==configuration(),
        'Actual corrected source review must bind the completion configuration')
    config_decision,parent=validate_selection(manifest['config_selection'])
    core.require(review.get('config_selection_sha256')==manifest['config_selection']['sha256']
        and review.get('config_allowlist')==manifest['config_allowlist']==config_decision['config_allowlist'],
        'Corrected completion must bind the exact approved config subset')
    return policy,policy_spec,manifest,review,dependency


def terminal_binding(terminal_spec,manifest_spec,review_spec,manifest):
    terminal=chain.document(terminal_spec)
    core.require(terminal['successor_terminal_status']=='terminal' and terminal['stage']=='continuation'
        and terminal['manifest_sha256']==manifest_spec['sha256'] and terminal['decision_sha256']==review_spec['sha256']
        and terminal['task_version']==VERSION and terminal['ontology_artifact']==manifest['ontology_artifact'],
        'Corrected generation has not completed under this task version')
    core.require(terminal.get('config_allowlist')==manifest['config_allowlist']
        and terminal.get('config_selection')==manifest['config_selection']
        and terminal.get('plan_accounting')==manifest['plan_accounting']
        and terminal.get('diagnostic_canary')==manifest['diagnostic_canary'],
        'Terminal report lost selected/held/diagnostic config accounting')
    source=manifest['sources'][0];done=terminal['sources']['lex_glue']
    expected={s['index']:s for s in source['chunks'] if s['index']>0}
    core.require(done['approved_for_stage'] is True and not done['held'] and done['unresolved']==done['unattempted']==0
        and done['completed']==done['expected_rows']==source['rows']
        and len(done['chunks'])==len(expected) and {s['index'] for s in done['chunks']}==set(expected),
        'Corrected source has held/missing/unresolved chunks')
    for item in done['chunks']:
        spec=expected[item['index']]
        core.require(item['status']=='complete' and item['unresolved']==0 and item['completed']==spec['rows']
            and item['input_sha256']==spec['sha256'] and item['effective_task_ids_sha256']==spec['task_ids_sha256'],
            'Corrected terminal chunk coverage changed')
        report=chain.document({'path':item['report_path'],'sha256':item['report_sha256']})
        core.require(report['backlog_manifest_sha256']==manifest_spec['sha256']
            and report['decision_sha256']==review_spec['sha256'],
            'Corrected chunk was generated under a different reviewed decision')
    return terminal


def freeze_selection(policy_path,policy_sha,review_spec,manifest_spec,terminal_spec,base,destination):
    policy,policy_spec,manifest,review,dependency=governing(policy_path,policy_sha,review_spec,manifest_spec)
    terminal=terminal_binding(terminal_spec,manifest_spec,review_spec,manifest)
    output=core.absolute(destination);output.mkdir(parents=True,exist_ok=False)
    manual={r['task_id'] for r in chain.document(policy['manual_exclusions'])['entries']}
    extra=review.get('additional_manual_exclusions')
    if extra:
        extra_doc=chain.document(extra);extra_ids=[r['task_id'] for r in extra_doc['entries']]
        core.require(extra_doc['reviewed_by']=='root' and len(set(extra_ids))==extra_doc['count']==len(extra_ids),
            'Corrected-probe manual exclusions are not explicit root-reviewed IDs')
        manual.update(extra_ids)
    exclusions=set(manual);quality=Counter();configs={};counts=dict.fromkeys(('results','accepted','excluded_accepted','exported'),0)
    root=Path(manifest_spec['path']).parent;source=manifest['sources'][0];chunks=[]
    with tempfile.TemporaryDirectory(prefix='corrected-lex-selection-') as temp:
        db=sqlite3.connect(str(Path(temp)/'ids.sqlite'));db.execute('PRAGMA cache_size=-1024')
        db.execute('CREATE TABLE ids (task_id TEXT PRIMARY KEY, source TEXT, exported INTEGER)')
        with (output/'training-exclusions.jsonl').open('x') as handle:
            for spec in source['chunks']:
                if spec['index']==0:continue  # All64 canaries are retained as evidence only.
                directory=root/'outputs/lex_glue'/f"chunk-{spec['index']:05d}"
                report_spec={'path':str(directory/'report.json'),'sha256':chain.file_sha(directory/'report.json')}
                saved,=[s for s in terminal['sources']['lex_glue']['chunks'] if s['index']==spec['index']]
                core.require(report_spec=={'path':saved['report_path'],'sha256':saved['report_sha256']},
                    'Selected config result report changed after terminal accounting')
                report=chain.document(report_spec);tasks=core.frozen_tasks(spec)
                core.require(report['status']=='complete' and report['source']=='lex_glue'
                    and report['rows']==report['completed']==len(tasks) and report['input_sha256']==spec['sha256']
                    and report['backlog_manifest_sha256']==manifest_spec['sha256']
                    and report['task_version']==VERSION and report['ontology_artifact']==manifest['ontology_artifact']
                    and set(report['result_hashes'])==set(tasks),'Incorrect corrected source chunk')
                accepted_count=0
                for key in sorted(tasks):
                    from data.lexglue_task_definition_correction import config_from_source_row_id
                    core.require(config_from_source_row_id(tasks[key]['source_row_id']) in manifest['config_allowlist'],
                        'Unapproved config reached selected-source export')
                    core.require(key not in source['chunks'][0]['exclude_task_ids'],
                        'Diagnostic canary reached selected-source export')
                    core.require(Path(key).name==key and key not in ('.','..'),'Unsafe corrected task ID')
                    result=chain.document({'path':str(directory/'results'/(key+'.json')),'sha256':report['result_hashes'][key]})
                    core.require(result['task_id']==key and result['source']=='lex_glue'
                        and result['task_sha256']==core.sha(json.dumps(tasks[key],sort_keys=True).encode()),
                        'Corrected input/result identity changed')
                    accepted=result['verification']['accepted'];core.require(type(accepted) is bool,'Nonboolean acceptance')
                    excluded=False
                    if accepted:
                        excluded,info=chain.evaluate_training_exclusion(result['trace']['messages'],key,manual)
                        values=dict(accepted=1,manual=info['manual'],dynamic=info['repeated_immutable_expansion'],
                            overlap=info['overlap'],excluded_union=excluded,eligible=not excluded)
                        quality.update(values)
                        from data.lexglue_task_definition_correction import config_from_source_row_id
                        config=config_from_source_row_id(tasks[key]['source_row_id'])
                        configs.setdefault(config,Counter()).update(values)
                        if excluded:
                            exclusions.add(key);info.update(config=config,result_sha256=report['result_hashes'][key])
                            handle.write(json.dumps(info)+'\n')
                    selected=accepted and not excluded
                    try:db.execute('INSERT INTO ids VALUES (?,?,?)',(key,'lex_glue',int(selected)))
                    except sqlite3.IntegrityError as exc:raise ValueError('Duplicate corrected task ID') from exc
                    counts['results']+=1;counts['accepted']+=accepted;accepted_count+=accepted
                    counts['excluded_accepted']+=accepted and excluded;counts['exported']+=selected
                core.require(accepted_count==report['counts']['automatic_accepted'],'Corrected accepted count changed')
                chunks.append({'tasks':spec,'report':report_spec,'results_root':str(directory/'results'),
                               'attempts_root':str(directory/'attempts')});db.commit()
        ids_sha=core.ids_digest(db);db.close()
    core.require(counts['results']==manifest['pending_rows'],'Selected corrected continuation coverage is incomplete')
    excluded_spec={'path':str(output/'training-exclusions.jsonl'),'sha256':chain.file_sha(output/'training-exclusions.jsonl')}
    quality_spec=chain.write_document(output/'training-quality-accounting.json',{'counts':dict(quality),
        'by_config':{k:dict(v) for k,v in configs.items()},'exclusions':excluded_spec,
        'code_sha256':chain.file_sha(Path(__file__).parent/'expansion_training_quality.py'),
        'manual_exclusions':policy['manual_exclusions'],'additional_manual_exclusions':extra,
        'generation_verdicts_unchanged':True,'count_scope':'Accepted corrected-Lex trace candidates'})
    artifacts=[review_spec,manifest_spec,terminal_spec,base['summary'],quality_spec,excluded_spec,manifest['ontology_artifact'],
        manifest['config_selection'],manifest['parent_manifest'],*manifest['diagnostic_canary'].values()]
    artifacts=[a for a in artifacts if isinstance(a,dict) and 'path' in a and 'sha256' in a]
    if extra:artifacts.append(extra)
    selection={'schema':core.SELECTION_SCHEMA,'status':'reviewed_complete','model':core.MODEL,'model_revision':core.REVISION,
        'reviewed_sources':['lex_glue'],'excluded_task_ids':sorted(exclusions),'base_snapshot':base,
        'normalization':{'module_sha256':chain.file_sha(Path(__file__).parent/'expansion_task_normalization.py')},
        'sources':[{'source':'lex_glue','counts':counts,'task_ids_sha256':ids_sha,'chunks':chunks}],
        'generation_manifest':manifest_spec,'generation_terminal':terminal_spec,
        'config_allowlist':manifest['config_allowlist'],'plan_accounting':manifest['plan_accounting'],
        'diagnostic_canary':manifest['diagnostic_canary'],
        'training_quality_accounting':quality_spec,'approval':chain.approvals(policy,policy_spec,*artifacts),
        'task_version':VERSION,'ontology_artifact':manifest['ontology_artifact']}
    return {'status':'complete','selection':chain.write_document(output/'selection.json',selection),
            'quality_accounting':quality_spec,'counts':counts}


def transport_payload(row):
    core.require(row['source']=='lex_glue','Non-Lex row in corrected source transport')
    provenance=json.loads(row['provenance']);trace=dict(provenance['trace'])
    trace.update(messages=json.loads(row['messages']),tools=json.loads(row['tools']))
    core.require(trace['task_id']==row['task_id'] and inspect_expansion_efficiency(trace['messages'])['eligible'],
        'Transport has changed identity or repeated immutable expansion')
    return trace,'lex_glue',None


def publish_readiness(policy,policy_sha,base,dependency,ready,candidate_path):
    """Validate through the frozen consumer before publishing its watched marker."""
    import copy
    candidate=chain.write_document(candidate_path,ready)
    staged=copy.deepcopy(policy)
    for item in staged['additional_readiness_dependencies']:
        if item['stream_id']==dependency['stream_id']:item['readiness_path']=candidate['path']
    core.require(chain.ready_additional_components(staged,policy_sha,base) is not None,
        'Corrected readiness candidate did not satisfy the actual frozen consumer')
    path=Path(dependency['readiness_path'])
    if path.exists():
        core.require(path.read_bytes()==Path(candidate['path']).read_bytes(),'Published corrected readiness changed')
        return {'path':str(path),'sha256':chain.file_sha(path)}
    return chain.write_document(path,ready)
