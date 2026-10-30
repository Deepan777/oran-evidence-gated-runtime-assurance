"""Approved prospective measurement amendment; never resume or rewrite old rows."""
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools import run_corrected_guard_integration as campaign

campaign.OUT=ROOT/'results/corrected_guard_integration/20261027_measurement_amendment1'
campaign.STUDY_ID='corrected_guard_20261027_measurement_amendment1'
campaign.RUN_ID_PREFIX='cg-20261027m1'
campaign.ACCEPTANCE_PROFILE='observed_outage_and_unbounded_sinr_v2'
campaign.AMENDMENT={
    'approval':'User explicitly approved finishing the described prospective measurement-rule amendment',
    'scope':'Only native positive-infinite SNR from an attached receiver is represented explicitly as unbounded; independently finite receiver fields remain measured',
    'not_a_finite_measurement':'sinr_db remains absent/NaN, never capped or replaced in raw measurement records',
    'policy_input':'Existing frozen causal imputer may supply SNR with explicit previous_measured/training_mean provenance; never measured guard evidence',
    'guard_inputs':'Measured throughput, BLER and RLC delay must still be finite; all thresholds and action logic unchanged',
    'other_nonfinite_fields':'NaN, negative infinity, malformed and other missing fields remain subject to the prior rejection/censoring rules',
    'preservation':'Old complete preflights and partial comparison remain unchanged; all 48 comparison rows are freshly executed under this amendment',
    'factors_unchanged':['scenario-seed design','offered traffic','jammer factors','policy checkpoint','numerical preprocessing','guard configuration','action definitions','radio images','radio-window phases','analysis estimands'],
}
# Include the actual acquisition/parser/analysis dependency closure, not only
# top-level controller files, in the prospective source identity.
extra=[path.relative_to(ROOT).as_posix() for directory in ('src','experiments/collection','experiments/live')
       for path in (ROOT/directory).rglob('*.py')]
extra+=['tools/run_corrected_guard_measurement_amendment.py',
        'tools/run_corrected_guard_measurement_campaign.py','tools/finalize_corrected_guard_evaluation.py',
        'tests/unit/test_unbounded_receiver_measurement.py']
campaign.SOURCE_FILES=tuple(sorted(set(campaign.SOURCE_FILES).union(extra)))

if __name__=='__main__': campaign.main()
