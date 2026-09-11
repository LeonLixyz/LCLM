"""Immutable235B-first/27B-failure routing; no teacher sees the other's tasks."""
import json
from pathlib import Path
from data.prepare_teacher_transition import file_sha

KINDS = {'first235': 'previously_unattempted', 'retry27': 'retry_rejected'}


def split_source(source, prepared, output):
    prepared, output = Path(prepared), Path(output)
    parent = json.loads((prepared/(source+'.build.json')).read_text())
    source_path = prepared/(source+'.tasks.jsonl')
    expected = parent['prepared_files'][source_path.name]
    if file_sha(source_path) != expected:
        raise ValueError('Prepared input hash changed')
    marker = output/(source+'.routes.json')
    if marker.exists():
        report = json.loads(marker.read_text())
        if report['parent_sha256'] != expected:
            raise ValueError('Route parent changed')
        for route in report['routes']:
            if file_sha(output/route['path']) != route['sha256']:
                raise ValueError('Route changed')
        return report
    output.mkdir(parents=True,exist_ok=True)
    handles = {}; counts = dict.fromkeys(KINDS,0); seen=set()
    try:
        for key in KINDS:
            directory=output/key; directory.mkdir(exist_ok=True)
            handles[key]=(directory/(source+'.tasks.jsonl')).open('xb')
        with source_path.open('rb') as stream:
            for line in stream:
                if not line.endswith(b'\n'):raise ValueError('Incomplete task')
                row=json.loads(line); identifier=row['task_id']
                if identifier in seen:raise ValueError('Duplicate routed ID')
                seen.add(identifier)
                matches=[k for k,v in KINDS.items() if row['_attempt_provenance']['kind']==v]
                if len(matches)!=1:raise ValueError('Unknown routing kind')
                key=matches[0];handles[key].write(line);counts[key]+=1
    finally:
        for handle in handles.values():handle.close()
    routes=[]
    for key,kind in KINDS.items():
        if counts[key]!=parent['counts'].get(kind,0):raise ValueError('Route count mismatch')
        path=output/key/(source+'.tasks.jsonl')
        routes.append({'route':key,'tasks':counts[key],'path':str(path.relative_to(output)),
                       'sha256':file_sha(path)})
        (path.parent/(source+'.build.json')).write_text(json.dumps({'status':'complete',
            'source':source,'tasks':counts[key],'route':key}))
    if source=='maud':
        (output/'retry27/maud-choices.json').write_bytes((prepared/'maud-choices.json').read_bytes())
    report={'source':source,'parent_sha256':expected,'routes':routes}
    marker.write_text(json.dumps(report,indent=2))
    return report
