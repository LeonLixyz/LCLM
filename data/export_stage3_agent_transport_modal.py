"""Lossless Parquet transport for native JSON trajectories and tool schemas."""
import json
from pathlib import Path
import modal
from data.stage3_full_modal import image,volume,ROOT

app=modal.App('lclm-stage3-agent-transport')

@app.function(image=image,cpu=8,memory=32768,timeout=86400,volumes={'/data':volume})
def export(kind:str='agents'):
    import os
    import pyarrow as pa
    import pyarrow.parquet as pq
    from collections import Counter
    if kind=='agents':
        source=ROOT/'agents-qwen-v2'
        report=json.loads((source/'report.json').read_text())
        names=['open-thoughts--OpenThoughts-Agent-SFT-100K','nvidia--Nemotron-Agentic-v1','nvidia--Nemotron-SFT-Agentic-v2']
        if any(name not in report for name in names):raise RuntimeError('Native cleaning is not complete')
        inputs=[source/(name+'.jsonl') for name in names]
    elif kind=='expansion':
        source=Path('/data/stage3-agent/real-expansion/pilots/full-20260906-v3')
        if not (source/'full-generation-report.json').exists():raise RuntimeError('Generation is not complete')
        inputs=sorted(source.glob('*.accepted.jsonl'))
    else:raise ValueError('Unknown export kind')
    destination=ROOT/f'{kind}-transport'
    if (destination/'report.json').exists():return json.loads((destination/'report.json').read_text())
    destination.mkdir(parents=True,exist_ok=True)
    if list(destination.glob('*.parquet')):raise RuntimeError('Inspect partial export before retrying')
    schema=pa.schema([(k,pa.string()) for k in ('messages','tools','source_dataset','source_row_id','sub_dataset','compression_scope','data_type')])
    buffer=[];counts=Counter();files=[]
    def flush():
        if not buffer:return
        path=destination/f'{kind}-{len(files):05d}.parquet'
        pq.write_table(pa.Table.from_pylist(buffer,schema=schema),path,compression='zstd')
        files.append(path.name);buffer.clear()
    for path in inputs:
        with path.open() as stream:
            for line in stream:
                row=json.loads(line)
                if kind=='expansion' and not row.get('verification',{}).get('accepted'):raise ValueError('Unverified expansion trace')
                assert isinstance(row['messages'],list)
                out={'messages':json.dumps(row['messages'],ensure_ascii=False),
                    'tools':json.dumps(row.get('tools') or [],ensure_ascii=False),
                    'source_dataset':row.get('source_dataset','unknown'),
                    'source_row_id':str(row.get('source_row_id') or row.get('task_id') or ''),
                    'sub_dataset':str(row.get('sub_dataset') or row.get('family') or path.stem),
                    'compression_scope':'none' if kind=='agents' else 'input_segments',
                    'data_type':'agent_trajectory'}
                buffer.append(out);counts[out['source_dataset']+':'+out['sub_dataset']]+=1
                if len(buffer)>=2000:flush()
    flush()
    result={'kind':kind,'rows':sum(counts.values()),'source_counts':dict(counts),'parquet_files':files,
        'native_jsonl_inputs':[str(p) for p in inputs],
        'transport':'json.loads(messages), json.loads(tools) restores the native Qwen-ready objects losslessly'}
    (destination/'report.json').write_text(json.dumps(result,indent=2));volume.commit()
    return result

@app.local_entrypoint()
def main(kind:str='agents'):
    print(json.dumps(export.remote(kind),indent=2))
