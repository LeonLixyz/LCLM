"""Audit every accepted row of completed sources; no generation files are changed."""
import json
from pathlib import Path
import modal
from data.stage3_full_modal import image, volume, ROOT

app = modal.App('lclm-full-expansion-format-audit')
GENERATED = Path('/data/stage3-agent/real-expansion/pilots/full-20260906-v6')
AUDITS = ROOT / 'full-expansion-format-audit'

@app.function(image=image, cpu=8, memory=32768, timeout=86400, max_containers=4,
              volumes={'/data':volume})
def audit_source(source:str):
    import hashlib
    import multiprocessing as mp
    from collections import Counter
    from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
    from data.build_full_expansion_tasks_modal import SOURCES
    from data.expansion_trace_audit import init_worker, audit_worker
    from data.stage3_tokenizers import DECODER_REVISION, ENCODER_REVISION
    if source not in SOURCES:
        raise ValueError('Unknown source')
    report_path = GENERATED / f'{source}.generation.json'
    generation = json.loads(report_path.read_text())
    if generation.get('status') != 'complete':
        raise ValueError('Source generation is incomplete')
    expected = sum(v for k,v in generation['reasons'].items() if k.startswith('accepted'))
    manifest_digest = hashlib.sha256((GENERATED/'generation-manifest.json').read_bytes()).hexdigest()
    allowed = None
    if source == 'pubmedqa_labeled':
        from data.pubmedqa_split import training_ids
        allowed = training_ids(json.loads(Path('/data/stage3-agent/real-expansion/sources/pubmedqa_labeled/official-splits/split-manifest.json').read_text()))
    AUDITS.mkdir(parents=True,exist_ok=True)
    digest = hashlib.sha256(); counts = Counter(); errors = []; failed = 0; minimum = None; seen = set()
    path = GENERATED / f'{source}.accepted.jsonl'
    with path.open('rb') as stream, ProcessPoolExecutor(max_workers=8,
            mp_context=mp.get_context('spawn'), initializer=init_worker) as pool:
        pending = set()
        def drain():
            nonlocal pending, failed, minimum
            done,pending = wait(pending,return_when=FIRST_COMPLETED)
            for future in done:
                result = future.result()
                if 'error' in result:
                    failed += 1
                    if len(errors)<100:errors.append(result)
                else:
                    metrics=result['metrics']; size=metrics.pop('minimum_segment_tokens')
                    minimum=size if minimum is None else min(minimum,size)
                    counts.update(metrics)
            if (counts['accepted']+failed)%500<len(done):
                print(source,counts['accepted'],failed,flush=True)
        for line in stream:
            if not line.endswith(b'\n'):raise ValueError('Truncated accepted JSONL')
            digest.update(line); row=json.loads(line)
            if row['task_id'] in seen:raise ValueError('Duplicate accepted task ID')
            seen.add(row['task_id'])
            pending.add(pool.submit(audit_worker,(row,source,allowed)))
            if len(pending)>=32:drain()
        while pending:drain()
    passed = not failed and len(seen) == expected == counts['accepted']
    result = {'status':'passed' if passed else 'failed', 'source':source,
        'expected_accepted':expected, 'rows':len(seen), 'counts':dict(counts),
        'minimum_segment_tokens':minimum, 'failed_rows':failed, 'errors':errors,
        'accepted_file_sha256':digest.hexdigest(), 'accepted_file':str(path),
        'generation_manifest_file_sha256':manifest_digest,
        'decoder_tokenizer_revision':DECODER_REVISION, 'encoder_tokenizer_revision':ENCODER_REVISION,
        'scope':'Every accepted row: native tools, exact expansion, segment length, saved prompt, source verifier, independent assistant-only labels. Semantic votes remain heuristic.'}
    (AUDITS/f'{source}.json').write_text(json.dumps(result,indent=2)); volume.commit()
    return result

@app.local_entrypoint()
def main(sources:str):
    selected=sources.split(',')
    if len(set(selected))!=len(selected):raise ValueError('Duplicate audit sources')
    for report in audit_source.map(selected):
        print(json.dumps(report,indent=2))
