"""Supplemental TechQA prompt tasks; never mutate the active 15-source build."""
import json
from pathlib import Path
import modal
from data.stage3_full_modal import image, volume

app = modal.App('lclm-techqa-expansion-prompt-build')
SOURCE = Path('/data/stage3-agent/real-expansion/sources/techqa/materialized_train_v1')
OUTPUT = Path('/data/stage3-agent/real-expansion/pilots/techqa-supplement-20260907-v1')

@app.function(image=image,cpu=4,memory=16384,timeout=1800,volumes={'/data':volume})
def build():
    import hashlib
    from collections import Counter
    from datasets import load_from_disk
    from transformers import AutoTokenizer
    from data.techqa_adapter import build_expansion_task
    from data.stage3_tokenizers import DECODER, DECODER_REVISION
    receipt = json.loads((SOURCE/'report.json').read_text())
    if receipt.get('status') != 'complete' or receipt['counts']['accepted'] != 396:
        raise ValueError('Verified TechQA training materialization required')
    if (OUTPUT/'techqa.build.json').exists():
        return json.loads((OUTPUT/'techqa.build.json').read_text())
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise RuntimeError('Inspect partial supplemental build before retrying')
    tasks = load_from_disk(str(SOURCE/'tasks/train'))
    documents = {r['document_id']:dict(r) for r in load_from_disk(str(SOURCE/'documents/train'))}
    heldout = set(receipt['heldout_answer_document_ids'])
    if set(documents) & heldout:
        raise ValueError('Held-out document in TechQA pool')
    tokenizer = AutoTokenizer.from_pretrained(DECODER,revision=DECODER_REVISION)
    counts = Counter(); minimum = None; samples = []; digest = hashlib.sha256()
    OUTPUT.mkdir(parents=True,exist_ok=True)
    path = OUTPUT/'techqa.tasks.jsonl'
    with path.open('wb') as stream:
        for row in tasks:
            counts['candidates'] += 1
            try:
                task = build_expansion_task(row,documents)
                lengths = [len(tokenizer.encode(s['text'],add_special_tokens=False)) for s in task['segments']]
                if min(lengths)<512:
                    raise ValueError('segment_below_512_tokens')
            except ValueError as exc:
                counts['excluded:'+str(exc)] += 1
                continue
            minimum = min(lengths) if minimum is None else min(minimum,min(lengths))
            data = (json.dumps(task,ensure_ascii=False)+'\n').encode()
            stream.write(data);digest.update(data)
            counts['tasks'] += 1
            counts['multi_segment_support'] += len(task['support_segment_ids'])>1
            counts['segments'] += len(task['segments'])
            if len(samples)<3:samples.append(task)
    report = {'status':'complete','source':'techqa','source_id':'PrimeQA/TechQA',
        'source_revision':receipt['revision'],'source_materialization_counts':receipt['counts'],
        'counts':dict(counts),'minimum_segment_tokens':minimum,
        'tasks_sha256':digest.hexdigest(),'decoder_tokenizer_revision':DECODER_REVISION,
        'included_in_running_stage3_build':False,
        'next_gate':'Separate Qwen-235B pilot and semantic review before any release inclusion.',
        'policy':'Training-only questions/documents. Exact original answer spans and complete original source chunks preserved. Related-source padding and distractors come only from the retained training pool.'}
    (OUTPUT/'techqa.build.json').write_text(json.dumps(report,indent=2))
    (OUTPUT/'preview-tasks.json').write_text(json.dumps(samples,indent=2))
    volume.commit()
    return report

@app.local_entrypoint()
def main():
    print(json.dumps(build.remote(),indent=2))
