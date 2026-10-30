"""Freeze new execution rows before any real v1.4 acquisition."""
from pathlib import Path
from collections import OrderedDict
import hashlib
import json
import random
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT/'results/prospective_comparison_v14/20261030'
NEW_FILES = ['docs/protocol_amendment_v1.4.md','experiments/live/prospective_runtime_v14.py',
             'tools/run_live_pilot_scenario_v14.sh','tools/run_prospective_v14.py',
             'analysis/prospective_comparison_v14.py','tools/prepare_prospective_v14.py']


def main():
    source = ROOT/'revision_2026-10-23/corrected_guard_measurement_amendment1_data_20261027.tar.gz'
    if hashlib.sha256(source.read_bytes()).hexdigest() != '1edd4e7a02b1789bee1bc3ccb756ced413eaad5b52e634b80315ff1838670e47':
        raise ValueError('Prior evidence archive identity mismatch')
    with tarfile.open(source) as archive:
        def get(suffix):
            names = [n for n in archive.getnames() if n.endswith(suffix)]
            if len(names) != 1: raise ValueError(names)
            return json.load(archive.extractfile(names[0]))
        prior = get('evaluation_freeze.json'); progress = get('evaluation_progress.json')
    accepted = {r['run_id'] for r in progress['completed'] if r['measurements_complete']}
    rejected = [r['run_id'] for r in progress['completed'] if not r['measurements_complete']]
    assert len(accepted) == 14 and rejected == ['cg-20261027m1-cg-b02-r2-policy-s9504']
    historical = []
    for row in progress['completed']:
        historical.append({**{k:row[k] for k in ('run_id','block','arm','seed','jammer','jsr_db','load','attenuation_db','exit_code')},
            'cohort':'primary','historical':True,'previously_accepted':row['measurements_complete'],
            'acceptance_epoch':'original_strict_45_of_45'})
    grouped = OrderedDict()
    for block,jammer,jsr,load,attenuation,seed,arm in prior['rows']:
        old_id = f'cg-20261027m1-{block}-{arm}-s{seed}'
        if old_id in accepted: continue
        row = dict(run_id=f'v14-20261030-{block}-{arm}-s{seed}',block=block,arm=arm,seed=seed,
                   jammer=jammer,jsr_db=jsr,load=load,attenuation_db=attenuation,cohort='primary',
                   historical=False,fault='none',randomization_seed=seed,
                   acceptance_epoch='prospective_v14_44_of_45_native_nan_only')
        if old_id in rejected: row['repeats_rejected_run'] = old_id
        grouped.setdefault(block, []).append(row)
    planned = []
    for block,rows in grouped.items():
        if rows[0]['jammer'] != 'none':
            insertion_seed = 20261030 + rows[0]['seed']
            position = random.Random(insertion_seed).randrange(5)
            gpre = {**rows[0],'arm':'gpre','cohort':'exploratory_gpre','exploratory':True,
                    'run_id':f'v14-20261030-{block}-gpre-s{rows[0]["seed"]}',
                    'exploratory_insertion_seed':insertion_seed,'exploratory_insertion_position':position}
            rows.insert(position,gpre)
        planned.extend(rows)
    replications = [('lost_ack',9601+i,window) for i,window in enumerate((17,20,23,26,29))]
    replications += [('command_timeout',9611,18),('command_timeout',9612,25),
                     ('applied_parameter_mismatch',9621,17),('applied_parameter_mismatch',9622,24)]
    replications += [('interception',9631+i,-1) for i in range(3)]
    for kind,seed,window in replications:
        planned.append(dict(run_id=f'v14-20261030-rep-{kind}-s{seed}',block=f'rep-{kind}-s{seed}',
            arm='guarded_policy',seed=seed,jammer='none' if kind=='interception' else 'partial_band',
            jsr_db=0 if kind=='interception' else 5,load='low',attenuation_db=0.,
            cohort='replication',historical=False,replication_kind=kind,
            fault='none' if kind=='interception' else kind,fault_window=window,
            acceptance_epoch='prospective_v14_44_of_45_native_nan_only'))
    assert len(planned)==54 and sum(r['cohort']=='primary' for r in planned)==34
    # Generate a separate pilot entry point by a single transparent path
    # substitution. The original shell script remains byte-identical.
    original = (ROOT/'tools/run_live_pilot_scenario.sh').read_bytes()
    assert original.count(b'experiments/live/run_xapp_loop.py') == 1
    pilot = original.replace(b'experiments/live/run_xapp_loop.py',b'experiments/live/prospective_runtime_v14.py')
    # The original registration helper moves the pre-created evidence
    # directory before recreating it. Do not let tee's open stdout inode
    # disappear into that pre-run directory archive.
    old = b'bash "$PROJECT/tools/run_live_registration.sh" | tee "$E/registration.log"'
    assert pilot.count(old) == 1
    new = (b'bash "$PROJECT/tools/run_live_registration.sh" | tee "$PROJECT/results/prospective_comparison_v14/20261030/logs/$RUN_ID.registration.log"\n'
           b'cp "$PROJECT/results/prospective_comparison_v14/20261030/logs/$RUN_ID.registration.log" "$E/registration.log"')
    pilot = pilot.replace(old,new)
    (ROOT/'tools/run_live_pilot_scenario_v14.sh').write_bytes(pilot)
    commit = subprocess.check_output(['git','log','-1','--format=%H','--','docs/protocol_amendment_v1.4.md'],cwd=ROOT,text=True).strip()
    subprocess.run(['git','diff','--exit-code','--','docs/protocol_amendment_v1.4.md'],cwd=ROOT,check=True)
    payload = dict(schema_version=1,study='Study V prospective continuation v1.4',protocol_commit=commit,
        historical_freeze=prior,historical_rows=historical,planned_rows=planned,
        recovery_endpoint_seconds=10,recovery_by_20s='unavailable_not_imputed',
        complete_windows_required=44,expected_windows=45,phases_seconds=prior['phases_seconds'],
        primary_holm_family_size=36,exploratory_gpre_holm_family_size=16,bootstrap_iterations=10000,
        missingness_rule='Native-NaN windows are missing, excluded from available-window metric means and break recovery persistence. No imputed observations.',
        fault_rule='Real loopback command proxy: lost ack withheld after observed upstream ack; timeout request withheld after real handshake; mismatch changes actual wire parameter. Never alter stored acknowledgements/readback.',
        exploratory_gpre_rule='Mean of the three completed, consecutive measured windows immediately before first authorized action; retain within active episode. Calibration/admission and all numeric thresholds unchanged.',
        pilot_derivation={'original_sha256':hashlib.sha256(original).hexdigest(),
                          'changes':['xApp wrapper entry-point path','preserve actual registration stdout outside moved pre-run directory, then copy it verbatim into final evidence directory']},
        files={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in NEW_FILES})
    STUDY.mkdir(parents=True,exist_ok=True)
    target = STUDY/'freeze.json'
    if target.exists(): raise FileExistsError('Freeze already exists; no overwrite')
    target.write_text(json.dumps(payload,indent=2)+'\n')
    print(json.dumps({'freeze':str(target),'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),
                      'protocol_commit':commit,'new_primary':34,'exploratory_gpre':8,'replications':12},indent=2))


if __name__=='__main__': main()
