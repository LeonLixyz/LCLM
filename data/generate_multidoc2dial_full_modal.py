"""Generate the corrected source only after a separately approved pilot.

Deploy and spawn full() once. Never append these records to the old source:
the reviewed full replacement must be explicitly integrated at release time.
"""
import hashlib
import json
import modal
from data.generate_multidoc2dial_repair_modal import (
    image, data_volume, hf_cache_volume, CACHE_ROOT, OUTPUTS, EXPECTED_SHA, run_corrective,
)

APP_NAME = 'lclm-md2d-corrective-full-v1'
app = modal.App(APP_NAME)
FULL_OUTPUTS = OUTPUTS.parent / 'multidoc2dial-chronological-v1-full'


@app.function(image=image, gpu='H200:8', cpu=16, memory=65536, timeout=86400,
              max_containers=1, scaledown_window=60,
              volumes={'/data': data_volume, CACHE_ROOT: hf_cache_volume},
              secrets=[modal.Secret.from_name('huggingface')])
def full():
    from data.corrective_expansion_review import validate_corrective_review
    data_volume.reload()
    generation = json.loads((OUTPUTS / 'pilot-generation-report.json').read_text())
    manifest_bytes = (OUTPUTS / 'generation-manifest.json').read_bytes()
    if generation['manifest'] != json.loads(manifest_bytes):
        raise ValueError('Pilot report/manifest mismatch')
    audit = json.loads((OUTPUTS / 'format-audit/multidoc2dial.json').read_text())
    review = json.loads((OUTPUTS / 'review-passed.json').read_text())
    digest = hashlib.sha256(); accepted_ids = set()
    with (OUTPUTS / 'multidoc2dial.accepted.jsonl').open('rb') as stream:
        for line in stream:
            digest.update(line); row = json.loads(line)
            if row['task_id'] in accepted_ids or row['verification']['accepted'] is not True:
                raise ValueError('Invalid corrective accepted rows')
            accepted_ids.add(row['task_id'])
    validate_corrective_review(generation, audit, review, hashlib.sha256(manifest_bytes).hexdigest(),
                               digest.hexdigest(), accepted_ids, EXPECTED_SHA)
    return run_corrective(None, FULL_OUTPUTS)
