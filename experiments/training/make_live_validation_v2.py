from __future__ import annotations
import csv, pathlib, yaml, shutil

def main() -> None:
    root=pathlib.Path('/home/deepanramkumar_p/oran-cloud-project')
    old=root/'datasets/metadata/manifests/live_validation'
    new=root/'datasets/metadata/manifests/live_validation_v2'
    new.mkdir(parents=True,exist_ok=True)
    for method in ('B0','B2','B5','P'):
        rows=list(csv.DictReader((old/f'{method}_live_validation_runs.csv').open(newline='')))
        for row in rows:
            if method=='P':
                row['STATE_SCHEMA']='/home/deepanramkumar_p/oran-cloud-project/results/training/live_controller_recalibrated_v2/state_schema.json'
                row['CONTROLLER_MODEL']='/home/deepanramkumar_p/oran-cloud-project/results/training/live_controller_recalibrated/md3qn_per_live_candidate.pt'
            elif method=='B5':
                row['STATE_SCHEMA']='/home/deepanramkumar_p/oran-cloud-project/results/training/live_controller_recalibrated_v2/state_schema.json'
                row['CONTROLLER_MODEL']='/home/deepanramkumar_p/oran-cloud-project/results/training/live_controller_recalibrated/prb_only_masked_dqn_live_candidate.pt'
        with (new/f'{method}_live_validation_runs.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    doc=yaml.safe_load((root/'LIVE_VALIDATION_FREEZE_MANIFEST.yaml').read_text())
    doc['schema_version']=2; doc['status']='LIVE_VALIDATION_FROZEN_V2'; doc['parent_freeze_manifest']='LIVE_VALIDATION_FREEZE_MANIFEST.yaml'
    doc['notes']='Versioned live-validation freeze using recalibrated action-transition P/B5 artifacts; original freeze unchanged.'
    for entry in doc['execution_plan']:
        method=entry['method']; entry['scenario_manifest']=f'datasets/metadata/manifests/live_validation_v2/{method}_live_validation_runs.csv'; entry['result_manifest']=f'results/raw/LIVE_VALIDATION_V2_{method}.json'
        entry['command']=[x.replace('datasets/metadata/manifests/live_validation/', 'datasets/metadata/manifests/live_validation_v2/').replace('results/raw/LIVE_VALIDATION_', 'results/raw/LIVE_VALIDATION_V2_') for x in entry['command']]
    doc['artifacts']['p_recalibrated_schema']={'path':'results/training/live_controller_recalibrated_v2/state_schema.json','sha256':'127719f66f1b3bc3622473089c8b94206dfe65fdd9cd212a7b51a2d773946fc5'}
    doc['artifacts']['p_recalibrated_controller']={'path':'results/training/live_controller_recalibrated/md3qn_per_live_candidate.pt','sha256':'713b48f1a16fdae31c37cf5b6ed2a8757fa80facbd4e99de3a5e64d74621e498'}
    doc['artifacts']['b5_recalibrated_controller']={'path':'results/training/live_controller_recalibrated/prb_only_masked_dqn_live_candidate.pt','sha256':'958ba0fc13d6fe505fb45e65bcbe5a59676161e4725ee5c3b1c42a8893ccdd68'}
    (root/'LIVE_VALIDATION_FREEZE_MANIFEST_V2.yaml').write_text(yaml.safe_dump(doc,sort_keys=False))
    print({'manifest':str(root/'LIVE_VALIDATION_FREEZE_MANIFEST_V2.yaml'),'rows_per_method':90})
if __name__=='__main__': main()
