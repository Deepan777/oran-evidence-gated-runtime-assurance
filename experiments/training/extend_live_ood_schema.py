from __future__ import annotations
import json, pathlib

def main() -> None:
    root=pathlib.Path('/home/OPERATOR_USER/oran-cloud-project')
    src=root/'results/training/live_controller_recalibrated/state_schema.json'
    out=root/'results/training/live_controller_recalibrated_v2'
    out.mkdir(parents=True,exist_ok=True)
    data=json.loads(src.read_text())
    idx=data['feature_names'].index('dl_throughput_mbps')
    data['raw_p99'][idx]=6.0
    data['status']='DEVELOPMENT_LIVE_SCHEMA_RECALIBRATED_ACTION_TRANSITIONS_CONTINUOUS_RANGE_EXTENSION'
    data['not_confirmatory']=True
    data['continuous_range_extension']={'feature':'dl_throughput_mbps','raw_p99':6.0,'basis':'observed complete development pilot maximum 5.55016 Mbps'}
    (out/'state_schema.json').write_text(json.dumps(data,indent=2))
    import shutil
    shutil.copy2(root/'results/training/live_controller_recalibrated/scaler.pkl',out/'scaler.pkl')
    print(json.dumps({'status':data['status'],'schema':str(out/'state_schema.json'),'raw_p99_dl_throughput':data['raw_p99'][idx]}))
if __name__=='__main__': main()
