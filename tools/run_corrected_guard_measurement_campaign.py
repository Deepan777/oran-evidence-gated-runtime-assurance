"""Bounded approved rerun: preflight, real fault, matched evaluation, archive."""
from pathlib import Path
import json
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
STUDY='20261027_measurement_amendment1'

def main():
    result=0
    for phase in ('smoke','fault','evaluation'):
        result=subprocess.run([sys.executable,'tools/run_corrected_guard_measurement_amendment.py','--phase',phase],cwd=ROOT).returncode
        if result: break
        if phase=='fault':
            progress=json.loads((ROOT/'results/corrected_guard_integration'/STUDY/'fault_progress.json').read_text())
            if not all(row.get('fault_recovery_sequence',{}).get('passed') is True for row in progress['completed']):
                print('INCONCLUSIVE_FUNCTIONAL_RECOVERY: preserved, no scenario/threshold change or retry',flush=True)
                result=3
                break
    packaged=subprocess.run([sys.executable,'tools/finalize_corrected_guard_evaluation.py',
        '--study-name',STUDY,'--run-prefix','cg-20261027m1',
        '--release-name','corrected_guard_measurement_amendment1_data_20261027',
        '--shutdown-on-finish'],cwd=ROOT).returncode
    return result or packaged

if __name__=='__main__': raise SystemExit(main())
