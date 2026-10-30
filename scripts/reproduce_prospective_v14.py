"""Reproduce Study V offline from packaged public evidence, one run at a time."""
from pathlib import Path
import argparse
import hashlib
import json
import re
import shutil
import sys
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from analysis import prospective_comparison_v14 as analysis


def reproduce(output):
    projection = json.loads((ROOT/'provenance/prospective_v14_public_projection.json').read_text())
    archive = ROOT/projection['public_archive']
    assert analysis.sha(archive)==projection['public_archive_sha256']
    original = json.loads((ROOT/'analysis_outputs/study_v_v14/prospective_comparison_v14.json').read_text())
    expected_rows = {r['run_id']:r for r in original['all_attempts']}
    expected_members = {r['path']:r for r in projection['retained_files']}
    source_members = {r['path']:r for r in projection['original_member_inventory']}
    omitted = {r['path'] for r in projection['omitted_files']}
    seen_members = set()
    rows = []
    attested = 0
    pattern = re.compile(r'^results/raw/iq-broker-live-([^/]+)/(.+)$')
    output.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='run-record-',dir=output) as temp:
        scratch = Path(temp)
        current = None
        def flush(run_id):
            nonlocal attested
            if run_id is None:
                return
            evidence = scratch/('iq-broker-live-'+run_id)
            attestation = evidence/'v14_attestation.json'
            if attestation.exists():
                native = json.loads(attestation.read_text())
                assert native['run_id']==run_id
                for item in native['preserved_native_evidence']:
                    name = 'results/raw/iq-broker-live-'+run_id+'/'+item['path']
                    assert source_members[name]['sha256']==item['sha256']
                    assert source_members[name]['bytes']==item['bytes']
                    assert name in seen_members or name in omitted
                attested += 1
            row = expected_rows[run_id]
            actual = analysis.audit_run(evidence,row,historical=row.get('historical',False),verify_attestation=False)
            for key in ('accepted','metrics','counts','revocations','fault_delivery','acceptance_reasons','complete_windows'):
                assert actual.get(key)==row.get(key),(run_id,key)
            rows.append(actual)
            # Remove only this disposable extraction, never original records.
            shutil.rmtree(evidence)
        with tarfile.open(archive,'r|gz') as source:
            for member in source:
                assert member.isfile()
                assert not Path(member.name).is_absolute() and '..' not in Path(member.name).parts
                payload = source.extractfile(member).read()
                expected = expected_members[member.name]
                assert len(payload)==expected['bytes'] and hashlib.sha256(payload).hexdigest()==expected['sha256']
                assert member.name not in seen_members
                seen_members.add(member.name)
                match = pattern.match(member.name)
                if not match:
                    continue
                run_id,relative = match.groups()
                assert run_id in expected_rows
                if run_id!=current:
                    flush(current)
                    assert run_id not in {r['run_id'] for r in rows}
                    current = run_id
                destination = scratch/('iq-broker-live-'+run_id)/relative
                destination.parent.mkdir(parents=True,exist_ok=True)
                destination.write_bytes(payload)
            flush(current)
    assert seen_members==set(expected_members)
    assert {r['run_id'] for r in rows}==set(expected_rows) and len(rows)==69
    original_wilcoxon = analysis.wilcoxon
    def archived_wilcoxon(a,b,**kwargs):
        # Pin the archived cloud zero-difference rule, not SciPy's newer auto.
        if np.any(np.asarray(a)-np.asarray(b)==0):
            kwargs['method']='approx'
        return original_wilcoxon(a,b,**kwargs)
    analysis.wilcoxon = archived_wilcoxon
    primary = [r for r in rows if r['cohort']=='primary']
    stats = []
    for scope in ('all','interference','no_interference'):
        for right in ('policy','no_action','fixed_a2'):
            for outcome in analysis.OUTCOMES:
                stats.append(analysis.comparison(primary,'guarded_policy',right,outcome,
                    scope=scope,seed=20261030+len(stats)))
    analysis.holm_family(stats,36)
    exploratory = []
    for right in ('guarded_policy','policy','no_action','fixed_a2'):
        for outcome in analysis.OUTCOMES:
            exploratory.append(analysis.comparison(rows,'gpre',right,outcome,scope='interference',
                seed=20262030+len(exploratory),exploratory=True))
    analysis.holm_family(exploratory,16)
    assert stats==original['primary_statistics']
    assert exploratory==original['exploratory_gpre_statistics']
    observed = {r['run_id']:r for r in rows}
    rows = [observed[r['run_id']] for r in original['all_attempts']]
    analysis.csv_write(output/'per_run_acceptance.csv',rows)
    analysis.csv_write(output/'primary_matched_statistics.csv',stats)
    analysis.csv_write(output/'exploratory_gpre_matched_statistics.csv',exploratory)
    return dict(all_attempts=rows,primary_statistics=stats,exploratory_gpre_statistics=exploratory,
        archive_members_verified=len(seen_members),native_execution_attestations_verified=attested,
        historical_native_execution_attestations_unavailable=15,original_outputs_reproduced_exactly=True,
        new_experiments=False,archived_wilcoxon_rule='normal approximation when paired differences contain zeros; otherwise archived auto rule',
        recovery_endpoint_seconds=10,recovery_by_20s='unavailable_not_imputed')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'reproduction_outputs/study_v_v14')
    args=parser.parse_args()
    assert not args.output.exists(),'Use a fresh output directory'
    result=reproduce(args.output)
    small={k:v for k,v in result.items() if k not in ('all_attempts','primary_statistics','exploratory_gpre_statistics')}
    (args.output/'reproduction_verification.json').write_text(json.dumps(small,indent=2)+'\n')
    print(json.dumps(dict(passed=True,attempts=len(result['all_attempts']),primary_tests=36,exploratory_tests=16)))


if __name__=='__main__':main()
