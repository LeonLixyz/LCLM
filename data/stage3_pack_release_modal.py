"""Resumable, bounded-memory packing of the Stage-3 rebuild on its data volume."""
import json
import os
from pathlib import Path
import modal

ROOT=Path('/data/stage3-build-20260906')
volume=modal.Volume.from_name('lclm-stage3-data')
app=modal.App('lclm-stage3-pack-20260906-'+os.environ.get('LCLM_PACK_JOB','base'))
image=(modal.Image.debian_slim(python_version='3.11')
    .pip_install('datasets==3.6.0','transformers==4.57.1','pyarrow>=18,<22','jinja2>=3.1')
    .env({'PYTHONPATH':'/opt/lclm','TOKENIZERS_PARALLELISM':'false'})
    .add_local_dir('.','/opt/lclm',ignore=['.git','.venv','__pycache__','_modal_run']))

@app.function(image=image,cpu=8,memory=32768,timeout=86400,max_containers=16,
              volumes={'/data':volume},secrets=[modal.Secret.from_name('huggingface')])
def pack_partition(partition:int,total:int=64,kind:str='base'):
    import multiprocessing as mp
    import os
    import pickle
    from collections import Counter
    import pyarrow.parquet as pq
    from data.preprocess_for_dynamic_packing import (
        worker_init,worker_process_example,StreamingPacker,ParquetBatchWriter)
    if kind not in ('base','agents','expansion'):raise ValueError('Unknown packing source')
    output=ROOT/f'packed-{kind}-cs16-32768'/f'part-{partition:03d}'
    output.mkdir(parents=True,exist_ok=True)
    complete=output/'report.json'
    if complete.exists():return json.loads(complete.read_text())
    source=Path('/data/stage3-final-mixture-cot50-v1') if kind=='base' else ROOT/f'{kind}-transport'
    if kind!='base' and not (source/'report.json').exists():
        raise RuntimeError(f'Input export has not completed: {kind}')
    files=sorted(source.glob('*.parquet'))
    if not files:raise RuntimeError('Missing completed input data')
    selected=files[partition::total]
    counts=Counter(input_rows=0,eligible_rows=0,packed_rows=0,packed_batches=0);source_counts=Counter()
    # Restart only this partition's explicitly named staging folder. Completed
    # partitions are immutable and skipped above; raw sources are never edited.
    stage=output/'in-progress'
    stage.mkdir(exist_ok=True)
    if list(stage.glob('*.parquet')):
        raise RuntimeError(f'Incomplete partition at {stage}; inspect before resuming')
    writer=ParquetBatchWriter(str(stage),base_name=f'part-{partition:03d}',shard_size=512)
    packer=StreamingPacker(32768,128,20260906+partition,32)
    def inputs():
        for path in selected:
            for batch in pq.ParquetFile(path).iter_batches(batch_size=32):
                for row in batch.to_pylist():
                    counts['input_rows']+=1
                    if kind=='base':row['_legacy_sft_prefix_strict']=True
                    yield row,16,'compression_prompt',32768,None
    def emit(batch):
        for row in batch:
            assert len(row['base_input_ids'])==len(row['base_labels'])
            assert len(row['memory_positions'])==len(row['memory_strings'])
            assert any(v!=-100 for v in row['base_labels'])
        writer.write(pickle.dumps(batch))
        counts['packed_batches']+=1
        counts['packed_rows']+=len(batch)
    with mp.get_context('spawn').Pool(8,initializer=worker_init,
            initargs=('Qwen/Qwen3-4B-Instruct-2507','Qwen/Qwen3-Embedding-0.6B')) as pool:
        for row in pool.imap_unordered(worker_process_example,inputs(),chunksize=16):
            if row is None:counts['rejected_processing']+=1;continue
            if row.get('_skipped_max_seq_len'):counts['over_32768']+=1;continue
            if row['estimated_seq_len']<18:counts['under_18']+=1;continue
            counts['eligible_rows']+=1
            counts['uncompressed_rows']+=not row['memory_strings']
            counts['labeled_tokens']+=row['_trainable_tokens']
            source_counts[row['_sub_dataset']]+=1
            core={k:v for k,v in row.items() if not k.startswith('_')}
            core['_sub_dataset']=row['_sub_dataset']
            for batch in packer.add_example(core):emit(batch)
            if counts['eligible_rows']%10000==0:print(partition,dict(counts),flush=True)
    for batch in packer.finalize():emit(batch)
    writer.close()
    assert counts['packed_rows']==counts['eligible_rows']
    os.replace(stage,output/'all_samples')
    report={'partition':partition,'partitions':total,'input_files':[p.name for p in selected],
        'counts':dict(counts),'sub_datasets':dict(source_counts),'reference_chunk_size':16,
        'max_packed_length':32768,'decoder_tokenizer':'Qwen/Qwen3-4B-Instruct-2507',
        'encoder_tokenizer':'Qwen/Qwen3-Embedding-0.6B',
        'base_prefix_recovery_separate':kind=='base',
        'format':'dynamic packed: decoder pretokenized; encoder memory strings tokenized at runtime'}
    complete.write_text(json.dumps(report,indent=2));volume.commit()
    return report

@app.local_entrypoint()
def main(kind:str='base'):
    for report in pack_partition.starmap((i,64,kind) for i in range(64)):
        print(json.dumps({'partition':report['partition'],'counts':report['counts']}),flush=True)
