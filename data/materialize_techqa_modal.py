"""Recover TechQA from its ordinary tar archive; never infer HF train splits."""
import hashlib
import json
import tarfile
from pathlib import Path
import modal
from data.stage3_full_modal import image, volume

app = modal.App('lclm-techqa-training-materialization')
SOURCE = Path('/data/stage3-agent/real-expansion/sources/techqa')

@app.function(image=image, cpu=4, memory=16384, timeout=1800, volumes={'/data':volume})
def materialize():
    from datasets import Dataset
    from data.techqa_adapter import convert_training
    output = SOURCE/'materialized_train_v1'
    report_path = output/'report.json'
    if report_path.exists():
        return json.loads(report_path.read_text())
    if output.exists() and any(output.iterdir()):
        raise RuntimeError('Inspect partial TechQA materialization before retrying')
    paths = {'training':'TechQA/training_and_dev/training_Q_A.json',
             'dev':'TechQA/training_and_dev/dev_Q_A.json',
             'validation':'TechQA/validation/validation_reference.json',
             'documents':'TechQA/training_and_dev/training_dev_technotes.json',
             'license':'TechQA/CDLA-Permissive-v1.0.pdf','readme':'TechQA/README.txt'}
    reverse = {path:key for key,path in paths.items()}; payloads = {}; hashes = {}; notices = {}
    with tarfile.open(SOURCE/'repository/TechQA.tar.gz', mode='r|gz') as stream:
        for member in stream:
            key = reverse.get(member.name)
            if key is None:
                continue
            if not member.isfile() or key in hashes:
                raise ValueError('Unexpected TechQA archive member')
            payload = stream.extractfile(member).read()
            hashes[key] = hashlib.sha256(payload).hexdigest()
            if key in ('readme','license'):
                notices[key] = payload
            else:
                payloads[key] = json.loads(payload)
    if set(hashes) != set(paths):
        raise ValueError('TechQA source archive is incomplete')
    if len(payloads['training']) != 600 or len(payloads['dev']) != 310:
        raise ValueError('Unexpected official TechQA split sizes')
    tasks, documents, counts, heldout = convert_training(**payloads)
    output.mkdir(parents=True,exist_ok=True)
    if not tasks or not documents:
        raise ValueError('No verified TechQA training questions')
    Dataset.from_list(tasks).save_to_disk(str(output/'tasks/train'))
    Dataset.from_list(documents).save_to_disk(str(output/'documents/train'))
    (output/'SOURCE_README.txt').write_bytes(notices['readme'])
    (output/'CDLA-Permissive-v1.0.pdf').write_bytes(notices['license'])
    snapshot = json.loads((SOURCE/'snapshot-manifest.json').read_text())
    report = {'status':'complete','source':'PrimeQA/TechQA','revision':snapshot['revision'],
              'data_license':'CDLA-Permissive-1.0','archive_member_sha256':hashes,
              'counts':counts,'heldout_answer_document_ids':heldout,
              'policy':'Only TRAIN_ questions with exact source spans; exclude any document that answers a dev/validation question. Unanswerable training tasks excluded. Pool contains only retained training answer documents.',
              'included_in_running_stage3_build':False}
    report_path.write_text(json.dumps(report,indent=2)); volume.commit()
    return {k:v for k,v in report.items() if k not in ('heldout_answer_document_ids','archive_member_sha256')}

@app.local_entrypoint()
def main():
    print(json.dumps(materialize.remote(),indent=2))
