"""Publish only after all Stage-3 components and validation gates complete."""
import json
from pathlib import Path
import modal
from data.stage3_full_modal import ROOT,image,volume

app=modal.App('lclm-stage3-final-publication')
RAW_REPO='leonli66/stage3-final-mixture-cot50-native-agent-v2'
PACKED_REPO=RAW_REPO+'-packed-cs16-32k'

@app.function(image=image,cpu=8,memory=32768,timeout=86400,volumes={'/data':volume},
              secrets=[modal.Secret.from_name('huggingface')])
def publish():
    import hashlib
    from huggingface_hub import HfApi
    from data.stage3_release_checks import validate_base_recovery,validate_expansion_completion,validate_final_artifact_audit,validate_expansion_format_audits
    reports={}
    for kind in ('base','agents','expansion'):
        root=ROOT/f'packed-{kind}-cs16-32768'
        parts=[root/f'part-{i:03d}'/'report.json' for i in range(64)]
        if not all(p.exists() for p in parts):raise RuntimeError(f'Packing incomplete: {kind}')
        reports[kind]=[json.loads(p.read_text()) for p in parts]
        for report in reports[kind]:
            if report['counts'].get('packed_rows')!=report['counts'].get('eligible_rows'):
                raise RuntimeError(f'Packing count mismatch: {kind}')
    recovery_root=ROOT/'packed-base-prefix-recovery'
    recovery_paths=[recovery_root/f'part-{i:03d}'/'report.json' for i in range(64)]
    if not all(p.exists() for p in recovery_paths):raise RuntimeError('Base prefix recovery incomplete')
    reports['base_recovery']=[json.loads(p.read_text()) for p in recovery_paths]
    base_counts=validate_base_recovery(reports['base'],reports['base_recovery'])
    validation=json.loads((ROOT/'validation/report.json').read_text())
    if not {'pytest','packed_artifacts','nccl','fsdp'}<=set(r['check'] for r in validation) or any(r['exit_code'] for r in validation):
        raise RuntimeError('Validation gate is not green')
    artifact_audit=json.loads((ROOT/'validation/packed-artifact-audit.json').read_text())
    validate_final_artifact_audit(artifact_audit)
    prefix_tests=json.loads((ROOT/'validation/prefix-recovery-tests.json').read_text())
    if prefix_tests['exit_code'] or len(prefix_tests.get('real_tokenizer_boundary_cases',[]))!=8:
        raise RuntimeError('Prefix-recovery validation is not green')
    native=json.loads((ROOT/'agents-transport/report.json').read_text())
    expansion=json.loads((ROOT/'expansion-transport/report.json').read_text())
    for kind,transport in [('agents',native),('expansion',expansion)]:
        if sum(r['counts']['input_rows'] for r in reports[kind])!=transport['rows']:
            raise RuntimeError(f'Raw/packed input-count mismatch: {kind}')
    generation_root=Path('/data/stage3-agent/real-expansion/pilots/full-20260906-v6')
    generation=json.loads((generation_root/'full-generation-report.json').read_text())
    if generation['status']!='complete':raise RuntimeError('Generation incomplete')
    if generation['manifest'].get('pubmedqa_split',{}).get('train_rows')!=450:
        raise RuntimeError('Expansion source split provenance is stale')
    provenance=json.loads((ROOT/'expansion-source-provenance.json').read_text())
    source_reports=[json.loads((generation_root/(r['source']+'.generation.json')).read_text()) for r in provenance['sources']]
    generation_counts=validate_expansion_completion(generation,source_reports,provenance)
    validate_expansion_format_audits(expansion.get('format_audits',[]),source_reports,
        hashlib.sha256((generation_root/'generation-manifest.json').read_bytes()).hexdigest())
    if expansion['rows']!=generation_counts['accepted']:
        raise RuntimeError('Expansion generation/export count mismatch')
    # Source split/license and generated-data review must be recorded explicitly.
    review_path=ROOT/'release-review.json'
    if not review_path.exists() or json.loads(review_path.read_text()).get('approved') is not True:
        raise RuntimeError('Missing final source/trajectory/packed-data review')
    review=json.loads(review_path.read_text())
    notices_root=ROOT/'source-notices-v1'
    notices_bytes=(notices_root/'index.json').read_bytes()
    notices=json.loads(notices_bytes)
    if review.get('source_notices_index_sha256')!=hashlib.sha256(notices_bytes).hexdigest() or review.get('base_mixture_terms_reviewed') is not True:
        raise RuntimeError('Final review must cover this notice bundle and original base-mixture terms')
    from data.source_notices import validate_notice_bundle
    expected_notices={('expansion',r['source']):(r['source_id'],r['upstream_revision'])
        for r in provenance['sources'] if r['source']!='synthetic'}
    expected_notices.update({('agents',r['repo'].replace('/','--')):(r['repo'],r['revision'])
        for r in json.loads((ROOT/'agent-source-manifest.json').read_text())})
    validate_notice_bundle(notices_root,notices,expected_notices)
    api=HfApi()
    for repo in (RAW_REPO,PACKED_REPO):api.create_repo(repo,repo_type='dataset',exist_ok=True)
    for repo in (RAW_REPO,PACKED_REPO):
        api.upload_folder(repo_id=repo,repo_type='dataset',folder_path=notices_root,
            path_in_repo='source-notices',ignore_patterns=['state.json'])
    raw_roots={'base':Path('/data/stage3-final-mixture-cot50-v1'),
        'agents':ROOT/'agents-transport','expansion':ROOT/'expansion-transport'}
    for kind,folder in raw_roots.items():
        api.upload_folder(repo_id=RAW_REPO,repo_type='dataset',folder_path=folder,path_in_repo=f'raw/{kind}',
            allow_patterns=['*.parquet'],ignore_patterns=['state.json'])
    for kind,report in [('agents',native),('expansion',expansion)]:
        for file in report['native_jsonl_inputs']:
            api.upload_file(repo_id=RAW_REPO,repo_type='dataset',path_or_fileobj=file,
                path_in_repo=f'native-jsonl/{kind}/{Path(file).name}')
    for kind in reports:
        for i in range(64):
            packed_root=recovery_root if kind=='base_recovery' else ROOT/f'packed-{kind}-cs16-32768'
            folder=packed_root/f'part-{i:03d}'/'all_samples'
            if not list(folder.glob('*.parquet')):continue
            api.upload_folder(repo_id=PACKED_REPO,repo_type='dataset',folder_path=folder,
                path_in_repo=f'data/{kind}/part-{i:03d}',allow_patterns=['*.parquet'],ignore_patterns=['state.json'])
    summary={'raw_repo':RAW_REPO,'packed_repo':PACKED_REPO,
        'native_rows':native['rows'],'expansion_rows':expansion['rows'],'packing':reports,
        'base_combined_counts':base_counts,
        'expansion_generation_counts':generation_counts,
        'expansion_format_audits':expansion['format_audits'],
        'source_notices_index_sha256':hashlib.sha256(notices_bytes).hexdigest(),
        'expansion_source_provenance':provenance,
        'source_revisions':json.loads((ROOT/'agent-source-manifest.json').read_text()),
        'validation':validation,'packed_artifact_audit':artifact_audit,
        'tokenizer_provenance_note':'Expansion packing pins decoder and encoder revisions in partition reports. Earlier base/native/recovery packs loaded main without explicit revisions; the final pinned-tokenizer artifact audit is sampled validation, not retroactive build provenance.',
        'review':json.loads(review_path.read_text())}
    for repo in (RAW_REPO,PACKED_REPO):
        api.upload_file(repo_id=repo,repo_type='dataset',path_or_fileobj=json.dumps(summary,indent=2).encode(),path_in_repo='build-manifest.json')
    raw_card='''---
language: en
configs:
- config_name: base
  data_files:
  - split: train
    path: raw/base/*.parquet
- config_name: agents
  data_files:
  - split: train
    path: raw/agents/*.parquet
- config_name: expansion
  data_files:
  - split: train
    path: raw/expansion/*.parquet
---
# Stage-3 CoT50 + native agents + expansion agents

This collection has three raw configs. No single config is the whole mixture.
The companion packed dataset includes all three components. See build-manifest.json
for counts, exclusions, revisions, and validation evidence.

Base: leonli66/stage3-final-mixture with the 50/50 reasoning rewrite: half ordinary
reasoning, half CoT memory wrapping; reasoning prompts are not compressed.

Agents: OpenThoughts-Agent-SFT-100K (open-thoughts), Nemotron-Agentic-v1 and
Nemotron-SFT-Agentic-v2 (NVIDIA). Follow each upstream source's licenses and terms;
this collection does not relicense them. Native imported agents contain no
compression and no annotated assistant CoT. All assistant calls/answers receive
loss; user/system/tool observations do not. Final malformed/ambiguous rows are excluded.

Expansion: Qwen3-235B-A22B-Instruct-2507 traces over real-source and synthetic
seg_i tasks. Memory-wrapped input segments can be expanded with a native tool.
Teacher and judge instructions are excluded from training messages. Free-form
answer judging is a heuristic, not a guarantee of correctness.

Agent/expansion Parquet fields messages and tools are lossless JSON strings to
avoid heterogeneous Arrow tool-schema casts. Decode with json.loads before
applying Qwen's chat template. The LCLM packer handles this automatically.
Fully nested, native training JSONL is also available under native-jsonl/.
Preserved upstream dataset cards and notices are under source-notices/; the
index records their original paths, pinned source revisions and file hashes.
'''
    raw_card += '\n## Expansion source provenance\n\n'
    raw_card += '| Source | Upstream | Declared license | Prompt tasks |\n| --- | --- | --- | ---: |\n'
    for record in provenance['sources']:
        source_id=record['source_id']
        upstream=source_id if source_id.startswith('https://') else (
            'https://huggingface.co/datasets/'+source_id if record['source']!='synthetic' else 'https://github.com/LeonLixyz/LCLM')
        raw_card += f"| {record['source']} | [{source_id}]({upstream}) | {record['declared_license']} | {record['tasks']:,} |\n"
    raw_card += '\nThese are input task counts, not accepted trace counts. Exact revisions, conversion receipts, exclusions and official PubMedQA holdouts are in build-manifest.json. Upstream terms and notices continue to apply; this collection does not relicense source content.\n'
    packed_card='''---
configs:
- config_name: default
  data_files:
  - split: train
    path: data/*/*/*.parquet
---
# Stage-3 dynamic packed mixture — cs16, 32768

All three components are included: base, plain native agents, and expansion agents.
Load with LCLM DynamicPackedDataset. Each parquet row contains packed_batch_bytes
(pickle of the packed examples); load only artifacts you trust. Decoder tokens and
labels are precomputed; memory strings are encoder-tokenized at runtime. Reference
compression ratio is 16. Packed length limit is 32768. Use all_samples, not only
exactly full packs. Overlength and invalid exclusions are reported in the manifest.
The base_recovery packed shard group contains only legacy SFT rows rejected by
the original token-prefix boundary check. It supplements base without duplicating
originally valid rows; base_combined_counts counts each input row once.
Pass the downloaded snapshot root to DynamicPackedDataset; it discovers
data/<component>/part-*/ shards. Existing flat parquet folders remain supported.

See the companion raw collection and build-manifest.json for source licenses,
provenance, counts, validation limits and per-partition statistics.
'''
    for repo,card in [(RAW_REPO,raw_card),(PACKED_REPO,packed_card)]:
        api.upload_file(repo_id=repo,repo_type='dataset',path_or_fileobj=card.encode(),path_in_repo='README.md')
    (ROOT/'publication-report.json').write_text(json.dumps(summary,indent=2));volume.commit()
    return {'raw':RAW_REPO,'packed':PACKED_REPO}

@app.local_entrypoint()
def main():
    print(json.dumps(publish.remote(),indent=2))
