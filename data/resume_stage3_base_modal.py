"""Inspect/preserve incomplete base outputs, then resume only unfinished partitions.

Run only after confirming the previous base packing app has stopped. Nothing is
deleted: incomplete outputs move to a timestamped quarantine outside upload globs.
"""
import json
from pathlib import Path
import modal
from data.stage3_pack_release_modal import ROOT, image, volume, pack_partition

app=modal.App('lclm-stage3-base-resume-inspect')

@app.function(image=image,cpu=2,timeout=600,volumes={'/data':volume})
def inspect(preserve:bool=False):
    import os
    import uuid
    import pyarrow.parquet as pq
    root=ROOT/'packed-base-cs16-32768'
    report=[]
    for i in range(64):
        part=root/f'part-{i:03d}'
        if (part/'report.json').exists():continue
        item={'partition':i,'folders':[]}
        for name in ['in-progress','all_samples']:
            folder=part/name
            if not folder.exists():continue
            files=list(folder.glob('*.parquet'))
            info={'name':name,'files':len(files),'bytes':sum(p.stat().st_size for p in files)}
            info['sampled_readable_batches']=0;info['unreadable_files']=[]
            for p in sorted(set(files[:2]+files[-2:])):
                try:info['sampled_readable_batches']+=pq.ParquetFile(p).metadata.num_rows
                except Exception:info['unreadable_files'].append(p.name)
            if preserve:
                destination=ROOT/'quarantine-base-partials'/f'part-{i:03d}-{name}-{uuid.uuid4().hex}'
                destination.parent.mkdir(exist_ok=True)
                os.rename(folder,destination);info['preserved_at']=str(destination)
            item['folders'].append(info)
        report.append(item)
    if preserve:
        (ROOT/'base-resume-inspection.json').write_text(json.dumps(report,indent=2));volume.commit()
    return report

@app.local_entrypoint()
def main(preserve:bool=False):
    print(json.dumps(inspect.remote(preserve),indent=2))
