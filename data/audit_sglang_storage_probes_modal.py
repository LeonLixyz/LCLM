"""Read-only token/label audit for source probes resumed on the output volume."""
import modal
from data.sglang_failed_1k_modal import cpu_image
from data.audit_sglang_backlog_modal import audit_snapshot

app = modal.App('lclm-sglang27-storage-probe-audit-20260909-v3')
inputs = modal.Volume.from_name('lclm-stage3-data', create_if_missing=False)
outputs = modal.Volume.from_name('lclm-stage3-agent-outputs-v2-20260909', create_if_missing=False)
ROOT = '/runs/stage3-build-20260906/sglang27-continuation-storage-20260909-v3'


@app.function(image=cpu_image, cpu=8, memory=32768, timeout=7200,
              max_containers=2, volumes={'/data': inputs, '/runs': outputs})
def audit_source(source_name: str):
    inputs.reload()
    return audit_snapshot(source_name, ROOT, outputs, 'resume-probe-decision.json')


@app.function(image=cpu_image, cpu=1, memory=2048, timeout=600)
def test_quality_policy():
    import subprocess
    result = subprocess.run(['python', '-m', 'pytest', '-q', 'tests/test_expansion_training_quality.py'],
                            cwd='/opt/lclm', capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(result.stdout + result.stderr)
    return {'status': 'passed', 'output': result.stdout}
