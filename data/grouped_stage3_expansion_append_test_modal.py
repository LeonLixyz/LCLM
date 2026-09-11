"""CPU-only fixture validation; never reads or appends production expansion."""
import subprocess
import modal
from data.grouped_stage3_packing_modal import options

app = modal.App('lclm-grouped-expansion-append-tests-20260909-v2')


@app.function(**options, cpu=8, memory=32768, timeout=900)
def test():
    result = subprocess.run(['python', '-m', 'pytest',
        '/opt/lclm/tests/test_grouped_stage3_expansion_append.py', '-q'],
        capture_output=True, text=True)
    print(result.stdout, flush=True)
    if result.returncode:
        print(result.stderr, flush=True)
        raise RuntimeError(result.stdout[-16000:] + result.stderr[-4000:])
    return {'status': 'passed', 'stdout': result.stdout,
            'scope': 'generated local fixtures in CPU container; no production selection/export/append'}
