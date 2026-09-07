"""Sample actual build artifacts through the runtime expansion and release loader."""
import json
import pickle
import tempfile
from pathlib import Path
import pyarrow.parquet as pq
from transformers import AutoTokenizer
from data.dynamic_packing_dataset import DynamicPackedDataset


def main():
    root=Path('/data/stage3-build-20260906')
    decoder=AutoTokenizer.from_pretrained('Qwen/Qwen3-4B-Instruct-2507',
        revision='cdbee75f17c01a7cc42f958dc650907174af0554')
    decoder.add_special_tokens({'additional_special_tokens':['<|memory_start|>','<|memory_end|>','<|memory|>']})
    encoder=AutoTokenizer.from_pretrained('Qwen/Qwen3-Embedding-0.6B')
    sample=object.__new__(DynamicPackedDataset)
    sample.decoder_tokenizer=decoder;sample.embed_tokenizer=encoder;sample.compression_ratio=16
    sample.memory_start_id=decoder.convert_tokens_to_ids('<|memory_start|>')
    sample.memory_id=decoder.convert_tokens_to_ids('<|memory|>')
    sample.memory_end_id=decoder.convert_tokens_to_ids('<|memory_end|>')
    sample.pooling='mean'
    report=[]
    with tempfile.TemporaryDirectory(prefix='lclm-release-loader-') as directory:
        for kind,folder in [('base','packed-base-cs16-32768'),('agents','packed-agents-cs16-32768'),
                            ('base_recovery','packed-base-prefix-recovery')]:
            parts=sorted((root/folder).glob('part-*/report.json'))
            assert parts,f'No completed {kind} partitions'
            for marker in sorted(set([parts[0],parts[len(parts)//2],parts[-1]])):
                files=sorted((marker.parent/'all_samples').glob('*.parquet'))
                assert files
                link=Path(directory)/'data'/kind/marker.parent.name/files[0].name
                link.parent.mkdir(parents=True,exist_ok=True);link.symlink_to(files[0])
                for file in sorted(set([files[0],files[-1]])):
                    packed=pq.ParquetFile(file)
                    raw=next(packed.iter_batches(batch_size=1,columns=['packed_batch_bytes']))[0][0].as_py()
                    examples=pickle.loads(raw)
                    length=0;memories=0;labeled=0
                    for ex in examples:
                        expanded=sample._expand_example(ex)
                        assert expanded is not None,'Runtime expansion dropped a packed example'
                        processed=expanded['processed'];labels=processed['labels'][0]
                        length+=expanded['seq_len'];memories+=len(ex['memory_strings'])
                        labeled+=sum(v!=-100 for v in labels)
                        assert sum(v!=-100 for v in ex['base_labels'])==sum(v!=-100 for v in labels)
                        for start,end in processed['memory_positions'][0]:
                            assert all(v==-100 for v in labels[start-1:end+1])
                    assert length<=32768,f'Actual expanded pack exceeds limit: {file}: {length}'
                    assert labeled>0
                    if kind=='agents':assert memories==0
                    report.append({'file':str(file),'examples':len(examples),
                        'expanded_tokens':length,'memory_blocks':memories,'labeled_tokens':labeled})
        rank_lengths=[]
        for rank in range(2):
            dataset=DynamicPackedDataset(directory,decoder,encoder,16,num_processes=2,process_rank=rank,
                shuffle=False,shuffle_files=False,drop_last_files=False,target_length=32768)
            rank_lengths.append(len(dataset))
            batch=next(iter(dataset))
            assert batch['input_ids'].shape==batch['labels'].shape
            assert batch['input_ids'].shape[1]==32768
            assert batch['labels'].ne(-100).any()
            assert sum(batch['sample_lens'][0])<=32768
        assert rank_lengths[0]==rank_lengths[1]
    output={'status':'passed','sampled_packs':report,'two_rank_loader_lengths':rank_lengths,
            'scope':'Sampled actual base/native/recovery packs, not a full artifact scan or production model run'}
    (root/'validation/packed-artifact-audit.json').write_text(json.dumps(output,indent=2))
    print(json.dumps(output,indent=2))


if __name__=='__main__':main()
