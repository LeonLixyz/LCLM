"""Validate persisted rollout checkpoints without modifying generation files."""
import json
from pathlib import Path
import modal
from data.stage3_full_modal import image, volume, ROOT

app = modal.App('lclm-expansion-resume-audit')

@app.function(image=image, cpu=8, memory=16384, timeout=1800, volumes={'/data':volume})
def audit():
    from datetime import datetime, timezone
    from data.expansion_checkpoint_audit import audit_source
    from data.build_full_expansion_tasks_modal import SOURCES
    inputs = Path('/data/stage3-agent/real-expansion/pilots/full-20260906-v3')
    outputs = inputs.parent / 'full-20260906-v6'
    reports = []
    for source in SOURCES:
        if not any((outputs / f'{source}.{kind}.jsonl').exists() for kind in ('accepted','rejected')):
            continue
        report = audit_source(inputs / f'{source}.tasks.jsonl', outputs, source)
        reports.append(report)
        print(source, report['persisted'], report['accepted'], flush=True)
    result = {'status':'passed', 'at':datetime.now(timezone.utc).isoformat(),
        'scope':'Read-only audit of sources with existing rollout files; not final release approval',
        'sources':reports, 'persisted':sum(r['persisted'] for r in reports),
        'accepted':sum(r['accepted'] for r in reports)}
    (ROOT / 'expansion-resume-audit.json').write_text(json.dumps(result, indent=2))
    volume.commit()
    return {**result, 'sources':[{k:v for k,v in r.items() if k not in ('files','samples','reasons')} for r in reports]}

@app.local_entrypoint()
def main():
    print(json.dumps(audit.remote(), indent=2))
