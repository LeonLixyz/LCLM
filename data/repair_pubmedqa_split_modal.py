"""Preserve old 1,000-example diagnostics; rebuild only official fold-0 training tasks."""
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path
import modal
from data.stage3_full_modal import volume

app = modal.App('lclm-pubmedqa-official-split-repair')
SOURCE = Path('/data/stage3-agent/real-expansion/sources/pubmedqa_labeled')
ROOT = Path('/data/stage3-agent/real-expansion/pilots/full-20260906-v3')
image = (modal.Image.debian_slim(python_version='3.11').apt_install('git')
         .pip_install('datasets==3.6.0', 'pyarrow>=18,<22')
         .env({'PYTHONPATH':'/opt/lclm'})
         .add_local_dir('.', '/opt/lclm', ignore=['.git','.venv','__pycache__','_modal_run']))

@app.function(image=image, cpu=4, memory=16384, timeout=1800, volumes={'/data':volume})
def repair():
    from data.pubmedqa_split import REVISION, training_ids
    from data.full_expansion_tasks import candidates, build_task
    destination = SOURCE/'official-splits'
    destination.mkdir(exist_ok=True)
    marker = destination/'split-manifest.json'
    if not marker.exists():
        # Fresh temporary checkout: upstream split script cannot overwrite any
        # existing user data when it creates its fold directories.
        with tempfile.TemporaryDirectory(prefix='pubmedqa-official-') as temporary:
            repo = Path(temporary)/'repository'
            subprocess.run(['git','clone','https://github.com/pubmedqa/pubmedqa.git',str(repo)],check=True)
            subprocess.run(['git','checkout','--detach',REVISION],cwd=repo,check=True)
            subprocess.run(['python','split_dataset.py','pqal'],cwd=repo/'preprocess',check=True)
            partitions = {name: json.loads((repo/'data'/file).read_text()) for name,file in {
                'train':'pqal_fold0/train_set.json','dev':'pqal_fold0/dev_set.json','test':'test_set.json'}.items()}
            manifest = {'repository':'https://github.com/pubmedqa/pubmedqa','revision':REVISION,'fold':0,
                'split_script_sha256':hashlib.sha256((repo/'preprocess/split_dataset.py').read_bytes()).hexdigest(),
                **{name+'_ids':sorted(rows) for name,rows in partitions.items()}}
            training_ids(manifest)
            marker.write_text(json.dumps(manifest,indent=2)); volume.commit()
    manifest = json.loads(marker.read_text()); allowed = training_ids(manifest)
    final_report = destination/'task-repair-report.json'
    if final_report.exists():
        return json.loads(final_report.read_text())
    rows = list(candidates('pubmedqa_labeled'))
    assert {r[0] for r in rows} == allowed
    pool = [r[3][0] for r in rows]
    tasks = [build_task('pubmedqa_labeled','qiaojin/PubMedQA:pqa_labeled',identifier,
                        question,answer,documents,pool) for identifier,question,answer,documents in rows]
    quarantine = ROOT/'quarantine-pubmed-original1000'
    quarantine.mkdir(exist_ok=True)
    for filename in ('pubmedqa_labeled.tasks.jsonl','pubmedqa_labeled.build.json','manifest.json'):
        old = ROOT/filename
        saved = quarantine/filename
        if old.exists() and not saved.exists():
            old.rename(saved)
    task_path = ROOT/'pubmedqa_labeled.tasks.jsonl'
    with task_path.with_suffix('.tmp').open('w') as stream:
        for task in tasks:
            stream.write(json.dumps(task,ensure_ascii=False)+'\n')
    task_path.with_suffix('.tmp').replace(task_path)
    build = {'source':'pubmedqa_labeled','builder':'v3-official-fold0',
             'counts':{'candidates':len(rows),'tasks':len(tasks),
                       'multi_segment_support':sum(len(t['support_segment_ids'])>1 for t in tasks)},
             'official_split_manifest':str(marker), 'excluded_heldout':550}
    (ROOT/'pubmedqa_labeled.build.json').write_text(json.dumps(build,indent=2))
    overall = json.loads((quarantine/'manifest.json').read_text())
    overall['sources'] = [build if r['source']=='pubmedqa_labeled' else r for r in overall['sources']]
    overall['pubmedqa_split_repair'] = str(marker)
    (ROOT/'manifest.json').write_text(json.dumps(overall,indent=2))
    result = {'status':'complete','training_tasks':len(tasks),'excluded_heldout':550,
              'originals_preserved_at':str(quarantine),'new_task_sha256':hashlib.sha256(task_path.read_bytes()).hexdigest(),
              'old_pubmed_pilots_are_diagnostics':True}
    final_report.write_text(json.dumps(result,indent=2)); volume.commit()
    return result

@app.local_entrypoint()
def main():
    print(json.dumps(repair.remote(),indent=2))
