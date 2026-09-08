"""Lossless Parquet transport for native JSON trajectories and tool schemas."""
import json
import modal
from data.stage3_full_modal import image,volume,ROOT

app=modal.App('lclm-stage3-agent-transport')

@app.function(image=image,cpu=8,memory=32768,timeout=86400,volumes={'/data':volume})
def export(kind:str='agents'):
    import hashlib
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
        from data.expansion_release_selection import load_selection,validate_transport_selection
        selected=load_selection(ROOT)
        generation_counts=selected['counts'];format_audits=selected['format_audits']
        audited={r['source']:r for r in format_audits};inputs=selected['inputs']
    else:raise ValueError('Unknown export kind')
    destination=ROOT/f'{kind}-transport'
    if (destination/'report.json').exists():
        existing=json.loads((destination/'report.json').read_text())
        if kind=='expansion':validate_transport_selection(existing,selected)
        return existing
    destination.mkdir(parents=True,exist_ok=True)
    if list(destination.glob('*.parquet')):raise RuntimeError('Inspect partial export before retrying')
    schema=pa.schema([(k,pa.string()) for k in ('messages','tools','source_dataset','source_row_id','sub_dataset','compression_scope','data_type')])
    buffer=[];counts=Counter();files=[];seen_ids=set()
    def flush():
        if not buffer:return
        path=destination/f'{kind}-{len(files):05d}.parquet'
        pq.write_table(pa.Table.from_pylist(buffer,schema=schema),path,compression='zstd')
        files.append(path.name);buffer.clear()
    for path in inputs:
        digest=hashlib.sha256()
        with path.open('rb') as stream:
            for line in stream:
                digest.update(line)
                row=json.loads(line)
                if kind=='expansion' and not row.get('verification',{}).get('accepted'):raise ValueError('Unverified expansion trace')
                if kind=='expansion':
                    if not row.get('task_id') or row['task_id'] in seen_ids:raise ValueError('Missing/duplicate expansion task ID')
                    seen_ids.add(row['task_id'])
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
        if kind=='expansion' and digest.hexdigest()!=audited[path.name.removesuffix('.accepted.jsonl')]['accepted_file_sha256']:
            raise RuntimeError('Expansion source file changed after full format audit')
    flush()
    if kind=='expansion' and sum(counts.values())!=generation_counts['accepted']:
        raise RuntimeError('Expansion JSONL/report accepted-count mismatch')
    result={'kind':kind,'rows':sum(counts.values()),'source_counts':dict(counts),'parquet_files':files,
        'native_jsonl_inputs':[str(p) for p in inputs],
        'transport':'json.loads(messages), json.loads(tools) restores the native Qwen-ready objects losslessly'}
    if kind=='expansion':
        result.update(format_audits=format_audits,expansion_selection=selected['selection'],
                      expansion_selection_sha256=selected['selection_sha256'])
    (destination/'report.json').write_text(json.dumps(result,indent=2));volume.commit()
    return result

@app.local_entrypoint()
def main(kind:str='agents'):
    print(json.dumps(export.remote(kind),indent=2))
