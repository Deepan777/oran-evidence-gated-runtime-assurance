#!/usr/bin/env python3
"""Validate existing code, measurements and analysis cardinalities offline."""
from pathlib import Path, PurePosixPath
import argparse
import collections
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reproduction_outputs'
sys.path.insert(0, str(ROOT))


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as s:
        for b in iter(lambda:s.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def call(name, args):
    env=dict(os.environ, PYTHONPATH=str(ROOT), PYTHONDONTWRITEBYTECODE='1')
    env.pop('PYTHONNOUSERSITE',None)
    command=[sys.executable]+args
    p=subprocess.run(command,cwd=ROOT,env=env,capture_output=True,text=True)
    (OUT/(name+'.log')).write_text(p.stdout+p.stderr,encoding='utf-8')
    if p.returncode:
        raise RuntimeError(f'{name} failed; inspect reproduction_outputs/{name}.log')
    return dict(command=command,returncode=p.returncode)


def find(pattern):
    matches=list(ROOT.glob(pattern))
    if len(matches)!=1:raise ValueError(f'Expected one input for {pattern}: {len(matches)}')
    return matches[0]


def run(name):
    if name=='integrity':
        rows=json.loads((ROOT/'RELEASE_MANIFEST.json').read_text())['files']
        for row in rows:
            p=ROOT/row['path']
            assert p.stat().st_size==row['bytes'] and sha(p)==row['sha256'],row['path']
        for line in (ROOT/'SHA256SUMS.txt').read_text().splitlines():
            digest,relative=line.split('  ',1);assert sha(ROOT/relative)==digest
        return dict(files=len(rows),all_hashes_verified=True)
    if name=='tests':
        return call(name,['-m','pytest','-q','-p','no:cacheprovider','tests'])
    if name=='inference':
        return call(name,['scripts/reproduce_release.py','--components-only'])
    if name in ('expanded','actuator','receiver'):
        module={'expanded':'recovery_censoring_correction','actuator':'actuator_recovery_censoring_correction','receiver':'receiver_metadata_recovery_sensitivity'}[name]
        result=call(name,['-m','analysis.'+module,'--output-dir',str(OUT/name)])
        csv_name={'expanded':'recovery_censoring_corrected_per_run.csv','actuator':'actuator_recovery_corrected_per_run.csv','receiver':'receiver_metadata_per_run.csv'}[name]
        if name in ('expanded','actuator'):
            candidates=list(ROOT.glob('*/analysis_outputs/'+csv_name))
            assert candidates and all(sha(p)==sha(OUT/name/csv_name) for p in candidates)
            result['per_run_csv_matches_preserved_bytes']=True
        return result
    if name=='records':
        result=call(name,['-m','analysis.audit_followup_records_20261028','--raw-manifest','provenance/live_validation_v10_raw_evidence_manifest.json','--frozen-loop','frozen_controller_360_run/run_xapp_loop.py','--output',str(OUT/name)])
        for filename in ('a1_per_run.csv','a1_recovery_scheduler_presence.csv','a2_fault_per_window_timing.csv','a3_rollback_event_windows.csv'):
            assert sha(ROOT/'analysis_outputs'/filename)==sha(OUT/name/filename),filename
        result['four_csv_files_match_preserved_bytes']=True
        return result
    if name=='transport':
        source=find('results/*_r1_release/extracted/results/*/command_impairment.json')
        data=json.loads(source.read_text())
        counts=collections.Counter(a['condition'] for a in data['attempts'])
        assert len(data['attempts'])==320 and len(counts)==8 and set(counts.values())=={40}
        assert len(data['conditions'])==8 and len(data['summaries'])==8
        return dict(attempts=320,conditions=8,attempts_each=40,source_sha256=sha(source))
    if name=='benchmark':
        source=find('*/analysis_outputs/corrected_guard_benchmark_measurement_amendment1.json')
        data=json.loads(source.read_text())
        assert data['repetitions']==30 and len(data['raw_repetition_means'])==30 and data['decisions_per_repetition']==5000
        assert {r['process_repetition'] for r in data['raw_repetition_means']}==set(range(30))
        for filename,digest in data['implementation_sha256'].items():assert sha(ROOT/filename)==digest,filename
        return dict(repetitions=30,decisions_per_repetition=5000,original_source_hashes_match=True,new_timing_observations_collected=False)
    if name=='corrected-loop':
        from analysis.corrected_guard_integration_analysis import build_report
        archive=ROOT/'data/evidence/corrected_loop_data_public.tar.gz'
        inputs=OUT/'corrected_loop_inputs'
        inputs.mkdir(exist_ok=True)
        with tarfile.open(archive) as h:
            for m in h:
                path=PurePosixPath(m.name)
                if not m.isfile() or path.is_absolute() or '..' in path.parts:continue
                if path.suffix not in ('.json','.jsonl'):continue
                destination=inputs/m.name
                destination.parent.mkdir(parents=True,exist_ok=True)
                destination.write_bytes(h.extractfile(m).read())
        study=next((inputs/'results/corrected_guard_integration').iterdir())
        reports={}
        for phase in ('smoke','fault','evaluation'):
            data=build_report(study,inputs/'results/raw',phase=phase)
            (OUT/('corrected_loop_'+phase+'.json')).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
            reports[phase]=dict(population=data['population'],completeness=data['completeness'])
        return dict(phases=reports,original_archive_sha256=sha(archive),new_observations_created=False)
    raise ValueError(name)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    names=('integrity','tests','inference','expanded','actuator','receiver','transport','benchmark','records','corrected-loop')
    parser.add_argument('--check',choices=('all',)+names,default='all')
    args=parser.parse_args()
    OUT.mkdir(exist_ok=True)
    results=[]
    for name in names if args.check=='all' else (args.check,):
        print('Checking '+name,flush=True)
        try:result=dict(name=name,passed=True,details=run(name))
        except Exception as exc:result=dict(name=name,passed=False,error=str(exc))
        results.append(result)
    packages={p:importlib.metadata.version(p) for p in ('numpy','pandas','scipy','torch','pytest','pyarrow')}
    report=dict(passed=all(r['passed'] for r in results),checks=results,python=sys.version,platform=platform.platform(),packages=packages,new_experiments=False)
    (OUT/('validation_'+args.check+'.json')).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    return 0 if report['passed'] else 1


if __name__=='__main__':
    raise SystemExit(main())
