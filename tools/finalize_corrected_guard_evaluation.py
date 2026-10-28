"""Verify, analyze and package real corrected-guard evidence; never retry runs."""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import sys
import tarfile
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from analysis.corrected_guard_integration_analysis import build_report

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wait',action='store_true')
    parser.add_argument('--shutdown-on-finish',action='store_true')
    parser.add_argument('--study-name',default='20261027_final')
    parser.add_argument('--run-prefix',default='cg-20261027')
    parser.add_argument('--service',default='oran-corrected-guard-evaluation-final.service')
    parser.add_argument('--release-name',default='corrected_guard_radio_data_20261027')
    args=parser.parse_args()
    service=args.service
    deadline=time.monotonic()+10800
    while args.wait and subprocess.run(['systemctl','is-active','--quiet',service]).returncode==0:
        if time.monotonic()>deadline: raise TimeoutError('Bounded finalizer deadline; no run retried')
        time.sleep(15)
    study=ROOT/'results/corrected_guard_integration'/args.study_name
    reports={}
    for phase in ('smoke','fault','evaluation'):
        progress=study/f'{phase}_progress.json'
        if progress.is_file():
            report=build_report(study,ROOT/'results/raw',phase=phase,iterations=10000)
            (study/f'{phase}_analysis.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
            reports[phase]={'population':report['population'],'completeness':report['completeness']}
    expected={'smoke':2,'fault':1,'evaluation':48}
    complete=all(phase in reports and reports[phase]['completeness']=='COMPLETE'
                 and reports[phase]['population']['recorded_runs']==count for phase,count in expected.items())
    sources=[]
    for path in study.rglob('*'):
        if path.is_file() and path.name!='packaging_manifest.json': sources.append(path)
    for directory in sorted((ROOT/'results/raw').glob(f'iq-broker-live-{args.run_prefix}-*')):
        sources.extend(path for path in directory.rglob('*') if path.is_file())
    ledger=[]
    for path in sorted(set(sources)):
        ledger.append({'path':path.relative_to(ROOT).as_posix(),'bytes':path.stat().st_size,'sha256':sha(path)})
    manifest={'schema_version':1,'status':'COMPLETE' if complete else 'INCOMPLETE_PRESERVED',
        'cohorts':reports,'expected_radio_windows':{'smoke':90,'fault':45,'evaluation':2160},
        'retries_performed_by_finalizer':0,'files':ledger,'measured_values_changed':False}
    manifest_path=study/'packaging_manifest.json'
    manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
    release=ROOT/'results/corrected_guard_release'
    release.mkdir(exist_ok=True)
    archive=release/(args.release_name+'.tar.gz')
    if archive.exists(): raise FileExistsError('Existing evidence archive retained; no overwrite')
    with tarfile.open(archive,'w:gz') as tar:
        for item in ledger: tar.add(ROOT/item['path'],arcname=item['path'],recursive=False)
        tar.add(manifest_path,arcname=manifest_path.relative_to(ROOT).as_posix())
    with tarfile.open(archive,'r:gz') as check:
        observed={}
        for member in check:
            if member.isfile(): observed[member.name]=hashlib.sha256(check.extractfile(member).read()).hexdigest()
    expected_hashes={item['path']:item['sha256'] for item in ledger}
    expected_hashes[manifest_path.relative_to(ROOT).as_posix()]=sha(manifest_path)
    if observed!=expected_hashes: raise RuntimeError('Archive member integrity or cardinality mismatch')
    checksum=sha(archive)
    archive.with_suffix(archive.suffix+'.sha256').write_text(checksum+'  '+archive.name+'\n')
    verification={'archive_sha256':checksum,'every_archived_file_verified':True,
        'entry_count':len(observed),'cohort_complete':complete,'status':manifest['status']}
    (release/(args.release_name+'.verification.json')).write_text(json.dumps(verification,indent=2)+'\n')
    print(json.dumps(verification),flush=True)
    if args.shutdown_on_finish: subprocess.run(['sudo','shutdown','-h','+2'],check=True)
    return 0 if complete else 2

if __name__=='__main__': raise SystemExit(main())
