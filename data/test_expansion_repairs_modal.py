"""CPU-only regression checks on Modal; never starts a generator or publishes."""
import json
import subprocess
import modal
from data.stage3_full_modal import image, volume, ROOT

app = modal.App('lclm-expansion-repair-regressions')


@app.function(image=image, cpu=2, memory=8192, timeout=600, volumes={'/data': volume})
def test():
    files = ['test_multidoc2dial_dialogue.py', 'test_expansion_corrective_paths.py', 'test_corrective_expansion_review.py',
             'test_expansion_judge_json.py', 'test_expansion_task_normalization.py',
             'test_harvest_expansion_trace.py']
    result = subprocess.run(['python', '-m', 'pytest', '-q'] + ['/opt/lclm/tests/' + f for f in files],
                            text=True, capture_output=True)
    report = {'exit_code': result.returncode, 'tests': files,
              'stdout': result.stdout, 'stderr': result.stderr,
              'scope': 'Corrective provenance/path isolation and existing harvest/prompt regressions; not GPU validation'}
    (ROOT / 'expansion-repair-regressions.json').write_text(json.dumps(report, indent=2))
    volume.commit()
    return report


@app.local_entrypoint()
def main():
    print(json.dumps(test.remote(), indent=2))
