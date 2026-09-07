"""Recover only rows rejected by the old SFT BPE-prefix check; never duplicate valid packs."""
import json
from pathlib import Path
import modal
from data.stage3_pack_release_modal import ROOT,volume,image

app=modal.App('lclm-stage3-sft-prefix-recovery')

@app.function(image=image,cpu=8,memory=32768,timeout=86400,max_containers=16,
              volumes={'/data':volume},secrets=[modal.Secret.from_name('huggingface')])
def recover(partition:int):
    import multiprocessing as mp
    import os
    import pickle
    from collections import Counter
    import pyarrow.parquet as pq
    from data.preprocess_for_dynamic_packing import worker_init,worker_process_example,StreamingPacker,ParquetBatchWriter
    output=ROOT/'packed-base-prefix-recovery'/f'part-{partition:03d}'
    report_path=output/'report.json'
    if report_path.exists():return json.loads(report_path.read_text())
    output.mkdir(parents=True,exist_ok=True)
    stage=output/'in-progress';stage.mkdir(exist_ok=True)
    if list(stage.glob('*.parquet')):raise RuntimeError('Inspect incomplete recovery partition before retrying')
    files=sorted(Path('/data/stage3-final-mixture-cot50-v1').glob('*.parquet'))[partition::64]
    counts=Counter(scanned=0,candidates=0,recovered_rows=0,packed_rows=0,packed_batches=0)
    def inputs():
        for path in files:
            index=0
            for batch in pq.ParquetFile(path).iter_batches(batch_size=64):
                for row in batch.to_pylist():
                    counts['scanned']+=1;index+=1
                    target=row.get('target_message')
                    content=target.get('content') if isinstance(target,dict) else row.get('target','')
                    # Qwen's BPE prefix newline can merge only with initial
                    # whitespace. Non-whitespace starts have a stable boundary.
                    if not isinstance(content,str) or not content or not content[0].isspace():continue
                    row['_recover_prefix_only']=True
                    counts['candidates']+=1
                    yield row,16,'compression_prompt',32768,None
    writer=ParquetBatchWriter(str(stage),base_name=f'recovered-{partition:03d}',shard_size=512)
    packer=StreamingPacker(32768,128,20260907+partition,32)
    def emit(batch):
        writer.write(pickle.dumps(batch));counts['packed_rows']+=len(batch);counts['packed_batches']+=1
    with mp.get_context('spawn').Pool(8,initializer=worker_init,
            initargs=('Qwen/Qwen3-4B-Instruct-2507','Qwen/Qwen3-Embedding-0.6B')) as pool:
        for row in pool.imap_unordered(worker_process_example,inputs(),chunksize=16):
            if row is None:counts['not_recoverable']+=1;continue
            if row.get('_skipped_max_seq_len'):counts['over_32768']+=1;continue
            if row['estimated_seq_len']<18:counts['under_18']+=1;continue
            counts['recovered_rows']+=1
            core={k:v for k,v in row.items() if not k.startswith('_')}
            core['_sub_dataset']=row['_sub_dataset']
            for batch in packer.add_example(core):emit(batch)
    for batch in packer.finalize():emit(batch)
    writer.close();assert counts['packed_rows']==counts['recovered_rows']
    os.replace(stage,output/'all_samples')
    report={'partition':partition,'counts':dict(counts),'recovery':'old_sft_token_prefix_mismatch_only'}
    report_path.write_text(json.dumps(report,indent=2));volume.commit();return report

@app.local_entrypoint()
def main():
    for report in recover.map(range(64)):print(json.dumps(report),flush=True)
