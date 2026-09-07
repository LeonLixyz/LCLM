"""Preserve upstream notices from pinned downloaded sources, without relicensing."""
import json
from pathlib import Path
import modal
from data.stage3_full_modal import ROOT, image, volume

app = modal.App('lclm-stage3-source-notices')
OUTPUT = ROOT / 'source-notices-v1'

@app.function(image=image,cpu=2,memory=8192,timeout=1800,volumes={'/data':volume})
def collect():
    from data.source_notices import notice_files, copy_notice, validate_notice_bundle
    provenance = json.loads((ROOT/'expansion-source-provenance.json').read_text())
    entries=[]
    for source in provenance['sources']:
        if source['source']=='synthetic':continue
        entries.append({'component':'expansion','key':source['source'],
            'source_id':source['source_id'],'revision':source['upstream_revision'],
            'root':Path('/data/stage3-agent/real-expansion/sources')/source['source']/'repository'})
    for source in json.loads((ROOT/'agent-source-manifest.json').read_text()):
        entries.append({'component':'agents','key':source['repo'].replace('/','--'),
            'source_id':source['repo'],'revision':source['revision'],'root':Path(source['path'])})
    results=[]
    for entry in entries:
        root=entry['root'];destination=OUTPUT/entry['component']/entry['key']
        if not root.is_dir() or not entry['revision']:
            raise ValueError(f'Missing pinned source checkout: {entry["key"]}')
        records=[copy_notice(path,root,destination) for path in notice_files(root)]
        results.append({k:v for k,v in entry.items() if k!='root'} | {
            'files':records,'notice_files':len(records),
            'has_standalone_license_or_notice':any(Path(r['upstream_path']).name.lower().startswith(
                ('license','licence','notice','copying','copyright')) for r in records)})
    result={'status':'collected','approved':False,'sources':results,
        'scope':'Pinned native-agent and non-synthetic expansion checkouts; not a legal clearance or complete base-mixture license audit.',
        'limits':'README metadata may not contain the full underlying data terms. Missing standalone notices require upstream review. Preserve all source restrictions; this collection does not relicense them.'}
    validate_notice_bundle(OUTPUT,result,{(r['component'],r['key']):(r['source_id'],r['revision']) for r in entries})
    OUTPUT.mkdir(parents=True,exist_ok=True)
    (OUTPUT/'index.json').write_text(json.dumps(result,indent=2));volume.commit()
    return {**result,'sources':[{k:v for k,v in r.items() if k!='files'} for r in results]}

@app.local_entrypoint()
def main():
    print(json.dumps(collect.remote(),indent=2))
