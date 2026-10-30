"""Validate prespecified v1.4.0 release criteria from package files only."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import sys
import tempfile
sys.dont_write_bytecode=True

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'scripts'))
from reproduce_prospective_v14 import reproduce


def validate(output):
    manifest=json.loads((ROOT/'RELEASE_MANIFEST.json').read_text())
    for item in manifest['files']:
        path=ROOT/item['path']
        assert path.is_file() and path.stat().st_size==item['bytes']
        digest=hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda:stream.read(1024*1024),b''):digest.update(chunk)
        assert digest.hexdigest()==item['sha256'],item['path']
    with tempfile.TemporaryDirectory(prefix='v140-validation-',dir=output.parent) as temp:
        result=reproduce(Path(temp))
    rows=result['all_attempts']
    checks={}
    def check(name,condition,actual):checks[name]=dict(passed=bool(condition),actual=actual)
    check('packaged_file_manifest',True,'all declared packaged files verified')
    accepted=[r for r in rows if r['accepted']]
    rejected=[r for r in rows if not r['accepted']]
    primary=[r for r in accepted if r['cohort']=='primary']
    unique={(r['block'],r['arm']) for r in primary}
    complete=[b for b in {r['block'] for r in primary} if {a for k,a in unique if k==b}=={'no_action','policy','guarded_policy','fixed_a2'}]
    kinds={r['block']:r['jammer'] for r in primary}
    population=dict(attempts=len(rows),accepted=len(accepted),rejected=len(rejected),accepted_unique_primary=len(unique),
        complete_blocks=len(complete),complete_without_interference=sum(kinds[b]=='none' for b in complete),
        complete_with_interference=sum(kinds[b]!='none' for b in complete))
    check('population',population==dict(attempts=69,accepted=64,rejected=5,accepted_unique_primary=45,
        complete_blocks=10,complete_without_interference=4,complete_with_interference=6),population)
    observed=sorted((r['seed'],r['arm'],r['complete_windows'],r['cohort']) for r in rejected)
    expected=sorted([(9504,'policy',44,'primary'),(9505,'no_action',42,'primary'),(9505,'fixed_a2',37,'primary'),
                     (9506,'guarded_policy',36,'primary'),(9621,'guarded_policy',45,'replication')])
    check('all_five_rejections',observed==expected and all(r['acceptance_reasons'] for r in rejected),observed)
    with (ROOT/'analysis_outputs/study_v_v14/per_run_acceptance.csv').open(newline='') as stream:csv_rows=list(csv.DictReader(stream))
    status={r['run_id']:r['accepted'].lower()=='true' for r in csv_rows}
    check('acceptance_csv',len(csv_rows)==69 and status=={r['run_id']:r['accepted'] for r in rows},dict(rows=len(csv_rows)))
    stats=result['primary_statistics'];exploratory=result['exploratory_gpre_statistics']
    pairs={arm:[next(r['n_pairs'] for r in stats if r['right']==arm and r['scope']==scope)
                for scope in ('all','interference','no_interference')] for arm in ('policy','no_action','fixed_a2')}
    check('matched_pair_counts',pairs==dict(policy=[11,7,4],no_action=[10,6,4],fixed_a2=[10,6,4]),pairs)
    check('primary_36_holm',len(stats)==36 and all(r['holm_p']==1. for r in stats),dict(tests=len(stats),adjusted_values=sorted({r['holm_p'] for r in stats})))
    check('exploratory_16_holm',len(exploratory)==16 and all(r['holm_p']==1. for r in exploratory),dict(tests=len(exploratory),adjusted_values=sorted({r['holm_p'] for r in exploratory})))
    requested=[('G_P_retention',stats,'policy','throughput_retention','all',['-0.0203','-0.0678','0.0106']),
        ('G_B0_drop',stats,'no_action','rlc_drop_rate','all',['-0.0088','-0.0156','-0.0026']),
        ('G_A2_recovery_all',stats,'fixed_a2','recovery_by_10s','all',['0.3000','0.0000','0.6000']),
        ('G_A2_recovery_interference',stats,'fixed_a2','recovery_by_10s','interference',['0.5000','0.1667','0.8333']),
        ('Gpre_G_recovery',exploratory,'guarded_policy','recovery_by_10s','interference',['0.2857','0.0000','0.5714'])]
    for name,table,arm,outcome,scope,target in requested:
        row=next(r for r in table if r['right']==arm and r['outcome']==outcome and r['scope']==scope)
        values=[row['mean_difference'],*row['ci95']];rounded=[f'{v:.4f}' for v in values]
        check(name,rounded==target,dict(rounded=rounded,unrounded=values))
    reps=[r for r in rows if r['cohort']=='replication']
    lost=[r for r in reps if r['replication_kind']=='lost_ack'];timeouts=[r for r in reps if r['replication_kind']=='command_timeout']
    mismatch=[r for r in reps if r['replication_kind']=='applied_parameter_mismatch']
    interception=sorted([r for r in reps if r['replication_kind']=='interception'],key=lambda r:r['seed'])
    for name,selected,count in [('lost_ack_replications',lost,5),('timeout_replications',timeouts,2)]:
        check(name,len(selected)==count and all(r['fault_delivery']=='observed' and r['counts']['unknown_episodes']==1 for r in selected),
            [dict(seed=r['seed'],delivery=r['fault_delivery'],UNKNOWN=r['counts']['unknown_episodes']) for r in selected])
    check('mismatch_inconclusive',len(mismatch)==2 and all(r['fault_delivery']=='inconclusive_not_delivered' for r in mismatch),
        [dict(seed=r['seed'],accepted=r['accepted'],delivery=r['fault_delivery']) for r in mismatch])
    counts=[r['counts']['interceptions'] for r in interception]
    check('interception_replications',counts==[45,44,45] and all(r['counts'].get('nontrivial_commands_sent',0)==0 for r in interception),counts)
    gpre=[r for r in accepted if r['cohort']=='exploratory_gpre']
    boundary=dict(harm=sum(r['guard_reasons'].get('post_action_harm_boundary_crossed',0) for r in gpre),
        maximum_duration=sum(r['guard_reasons'].get('maximum_action_duration_reached',0) for r in gpre))
    check('Gpre_boundary_counts',len(gpre)==8 and boundary==dict(harm=3,maximum_duration=13),boundary)
    g=[r for r in primary if r['arm']=='guarded_policy' and r['jammer']!='none'];counts=[r['harm_boundary_revocations'] for r in g]
    check('G_boundary_counts',len(g)==7 and counts==[3]*7,counts)
    integrity=len(list((ROOT/'provenance/run_integrity_v140').glob('*.json')))
    for row in rows:
        receipt=json.loads((ROOT/'provenance/run_integrity_v140'/(row['run_id']+'.json')).read_text())
        assert receipt['run_id']==row['run_id'] and receipt['original_members']
        assert receipt['native_execution_attestation_available']==(not row.get('historical',False))
        assert receipt['historical_execution_proof_inferred'] is False
    check('native_attestations_and_preservation',result['native_execution_attestations_verified']==54 and integrity==69,
        dict(native_execution_attestations=result['native_execution_attestations_verified'],historical_native_attestations_unavailable=15,
             post_collection_integrity_records=integrity))
    check('frozen_outputs_reproduced',result['original_outputs_reproduced_exactly'],True)
    assembly=json.loads((ROOT/'provenance/v140_assembly.json').read_text())
    license_ok=hashlib.sha256((ROOT/'LICENSE.txt').read_bytes()).hexdigest()==assembly['license_sha256']
    metadata=json.loads((ROOT/'ZENODO_METADATA.json').read_text())
    creators=[c['name'] for c in metadata['creators']]
    check('license_and_creators',license_ok and metadata['license']=='cc-by-4.0' and creators==['Deepanramkumar, P.','Sharmila, A. Helen'],
        dict(license_identical=license_ok,creator_count=len(creators)))
    report=dict(version='1.4.0',passed=all(r['passed'] for r in checks.values()),checks=checks,
        settings=dict(recovery_endpoint_seconds=10,bootstrap_resamples=10000,primary_holm_family=36,exploratory_holm_family=16,
                      wilcoxon=result['archived_wilcoxon_rule']),real_records_only=True,new_experiments=False,measured_values_modified=False,
        historical_attestation_boundary='15 earlier runs have no native execution attestations; post-collection integrity records do not supply historical execution proof')
    output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(passed=report['passed'],checks={k:v['passed'] for k,v in checks.items()}),indent=2))
    if not report['passed']:raise SystemExit('STOP: required release value differs; do not publish')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'validation/v1.4.0_validation_report.json')
    args=parser.parse_args();args.output.parent.mkdir(parents=True,exist_ok=True);validate(args.output)
