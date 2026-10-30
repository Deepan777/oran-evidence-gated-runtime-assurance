"""Bounded real-execution continuation. No automatic replacement runs."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import os
import signal
import subprocess
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from analysis.prospective_comparison_v14 import audit_run, analyse, sha
STUDY = ROOT/'results/prospective_comparison_v14/20261030'


def write(path, payload):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(payload,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)


def identity(freeze):
    for relative,digest in freeze['historical_freeze']['source_sha256'].items():
        if sha(ROOT/relative) != digest: raise ValueError('Original frozen source changed: '+relative)
    for relative,digest in freeze['files'].items():
        if sha(ROOT/relative) != digest: raise ValueError('Prospective source changed: '+relative)
    prior = freeze['historical_freeze']
    for key,path in [('controller_sha256','results/training/live_controller_recalibrated/md3qn_per_live_candidate.pt'),
                     ('schema_sha256','results/training/live_controller_recalibrated_v2/state_schema.json'),
                     ('detector_sha256','results/training/live_detector/detector_live_calibrated.joblib')]:
        if sha(ROOT/path) != prior[key]: raise ValueError('Frozen artifact changed: '+path)
    for name,key in [('image','image_id'),('ue_image','ue_image_id')]:
        observed = subprocess.check_output(['sudo','docker','image','inspect',prior[name],'--format','{{.Id}}'],text=True).strip()
        if observed != prior[key]: raise ValueError('Frozen image changed: '+prior[name])


def package(freeze, state):
    result = analyse(STUDY,ROOT/'results/raw')
    files = [p for p in STUDY.rglob('*') if p.is_file() and p.name not in ('packaging_manifest.json','service.log')]
    # Include every newly attempted raw directory; original cohorts remain
    # preserved at their original locations and are separately referenced.
    for row in state['completed']:
        files += [p for p in (ROOT/'results/raw'/('iq-broker-live-'+row['run_id'])).rglob('*') if p.is_file()]
    ledger = [{'path':p.relative_to(ROOT).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(set(files))]
    manifest = dict(status='COMPLETE' if result['complete_primary_comparison'] else 'INCOMPLETE_ALL_ATTEMPTS_PRESERVED',
                    primary_population=result['population'],protocol_commit=freeze['protocol_commit'],files=ledger,
                    historical_source_archive_sha256='1edd4e7a02b1789bee1bc3ccb756ced413eaad5b52e634b80315ff1838670e47',
                    measured_values_modified=False,automatic_retries=0)
    write(STUDY/'packaging_manifest.json',manifest)
    release = ROOT/'results/prospective_comparison_v14_release'; release.mkdir(exist_ok=True)
    archive = release/'prospective_comparison_v14_20261030_full_private.tar.gz'
    if archive.exists(): raise FileExistsError('Prior archive preserved; refuse overwrite')
    with tarfile.open(archive,'w:gz') as tar:
        for row in ledger: tar.add(ROOT/row['path'],arcname=row['path'],recursive=False)
        tar.add(STUDY/'packaging_manifest.json',arcname=(STUDY/'packaging_manifest.json').relative_to(ROOT).as_posix())
    with tarfile.open(archive) as tar:
        observed = {m.name:hashlib.sha256(tar.extractfile(m).read()).hexdigest() for m in tar if m.isfile()}
    expected = {r['path']:r['sha256'] for r in ledger}
    expected[(STUDY/'packaging_manifest.json').relative_to(ROOT).as_posix()] = sha(STUDY/'packaging_manifest.json')
    if observed != expected: raise ValueError('Archive contents failed integrity verification')
    digest = sha(archive)
    archive.with_suffix('.tar.gz.sha256').write_text(digest+'  '+archive.name+'\n')
    write(release/'verification.json',dict(archive_sha256=digest,entries=len(observed),all_members_verified=True,
          status=manifest['status'],primary_population=result['population']))


def main():
    freeze = json.loads((STUDY/'freeze.json').read_text())
    identity(freeze)
    state_path = STUDY/'progress.json'
    state = json.loads(state_path.read_text()) if state_path.exists() else {'completed':[],'expected_new_runs':54,'status':'RUNNING'}
    done = {r['run_id'] for r in state['completed']}
    (STUDY/'logs').mkdir(exist_ok=True)
    for row in freeze['planned_rows']:
        if row['run_id'] in done: continue
        identity(freeze)
        if __import__('shutil').disk_usage(ROOT).free < 8*1024**3:
            raise RuntimeError('Disk-space guard: preserve existing evidence and stop before next run')
        evidence = ROOT/'results/raw'/('iq-broker-live-'+row['run_id'])
        if evidence.exists():
            audit = audit_run(evidence,{**row,'exit_code':-2})
            audit['acceptance_reasons'].append('prior_unclassified_attempt_no_retry')
            audit['accepted'] = False; audit['status']='rejected'
        else:
            protected,background = (.3,2.1) if row['load']=='low' else (.9,6.3)
            prior = freeze['historical_freeze']
            arm = row['arm']
            env = dict(RUN_ID=row['run_id'],SCENARIO_ID=row['block'],JAMMER=row['jammer'],JSR_DB=str(row['jsr_db']),
                SEED=str(row['seed']),TRAFFIC_LOAD=row['load'],SERVICE_MIX='protected_plus_background',
                ATTENUATION_DB=str(row['attenuation_db']),RUN_SECONDS='45',PRE_JAM_SECONDS='15',
                JAMMER_ON_SECONDS='20',RECOVERY_SECONDS='10',PROTECTED_MBPS=str(protected),BACKGROUND_MBPS=str(background),
                EVIDENCE_CLASS='prospective_study_v_v14_exploratory_gpre' if arm=='gpre' else 'prospective_study_v_v14',
                GNB_IMAGE=prior['image'],UE_IMAGE=prior['ue_image'],GNB_CONFIG='ocudu_gnb_zmq_e2_udp_um.yml',
                MAX_WALL_SECONDS='240',REQUIRE_COMPLETE_TELEMETRY='0',REQUIRE_EVIDENCE_COMPLETE='0',
                REQUIRE_PROTECTED_TRAFFIC='1',ACCEPTANCE_PROFILE='observed_outage_and_unbounded_sinr_v2',
                OOD_GUARD_PROFILE='physical_plausibility_with_shift_audit',UE_TIMEOUT_MS='5000',
                CAPTURE_FULL_CONTAINER_LOGS='1',ALLOW_UPLINK_SILENCE='0',ENABLE_LIVE_XAPP='0' if arm=='no_action' else '1',
                LIVE_XAPP_XAI='0',CONTROLLER_KIND='md3qn_per',
                DETECTOR_MODEL=str(ROOT/'results/training/live_detector/detector_live_calibrated.joblib'),
                CONTROLLER_MODEL=str(ROOT/'results/training/live_controller_recalibrated/md3qn_per_live_candidate.pt'),
                STATE_SCHEMA=str(ROOT/'results/training/live_controller_recalibrated_v2/state_schema.json'),
                LIVE_XAPP_SELECTION_MODE={'no_action':'policy','policy':'policy','guarded_policy':'safeguarded_policy',
                    'gpre':'safeguarded_policy','fixed_a2':'bounded_action_probe'}[arm],
                ORAN_RECEIVER_MEASUREMENT_PROFILE='native_positive_infinity_v1',
                ORAN_V14_EVIDENCE=str(evidence),ORAN_V14_GPRE='1' if arm=='gpre' else '0',
                ORAN_V14_FAULT=row.get('fault','none'),ORAN_V14_FAULT_WINDOW=str(row.get('fault_window',-1)),
                ORAN_TEST_DROP_FIRST_NONTRIVIAL_ACK='0',ORAN_TEST_DROP_ACK_COMMAND_INDEX='0')
            if arm=='fixed_a2': env['LIVE_XAPP_FORCED_ACTION_ID']='2'
            started = time.time()
            with (STUDY/'logs'/(row['run_id']+'.log')).open('w') as stream:
                process = subprocess.Popen(['sudo','env',*[f'{k}={v}' for k,v in env.items()],
                    'bash','tools/run_live_pilot_scenario_v14.sh'],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                try: code = process.wait(timeout=360)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid,signal.SIGTERM)
                    try: process.wait(timeout=15)
                    except subprocess.TimeoutExpired: os.killpg(process.pid,signal.SIGKILL); process.wait()
                    code = -1
            audit = audit_run(evidence,{**row,'exit_code':code,'wall_seconds':time.time()-started})
            if evidence.exists():
                evidence_files = [{'path':p.name,'bytes':p.stat().st_size,'sha256':sha(p)}
                                  for p in sorted(evidence.iterdir()) if p.is_file()]
                write(evidence/'v14_attestation.json',dict(run_id=row['run_id'],protocol_commit=freeze['protocol_commit'],
                    freeze_sha256=sha(STUDY/'freeze.json'),source_sha256=freeze['files'],
                    preserved_native_evidence=evidence_files,acceptance=audit))
        state['completed'].append(audit)
        state['updated_utc'] = datetime.now(timezone.utc).isoformat()
        write(state_path,state)
        print(json.dumps({k:audit.get(k) for k in ('run_id','status','complete_windows','acceptance_reasons','fault_delivery')}) ,flush=True)
    state['status']='ALL_PLANNED_ATTEMPTS_RECORDED'; write(state_path,state)
    package(freeze,state)
    print('PROSPECTIVE_V14_ALL_ATTEMPTS_ANALYSED_AND_ARCHIVED',flush=True)
    # Safe bounded resource lifecycle; release synchronisation occurs locally
    # after this private archive has been verified and the service has exited.
    subprocess.run(['sudo','shutdown','-h','+3'],check=True)


if __name__=='__main__': main()
