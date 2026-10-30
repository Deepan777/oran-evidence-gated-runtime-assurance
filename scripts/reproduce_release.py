#!/usr/bin/env python3
"""Offline reproduction from this package only; never creates cloud resources."""
from __future__ import annotations
import argparse, hashlib, importlib.metadata, json, os, pathlib, platform, subprocess, sys, time
ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
OUT=ROOT/"reproduction_outputs"
OUT.mkdir(exist_ok=True)
def run(name, command):
    started=time.time()
    env=dict(os.environ); env["PYTHONPATH"]=str(ROOT); env.pop("PYTHONNOUSERSITE",None)
    proc=subprocess.run(command,cwd=ROOT,env=env,text=True,capture_output=True)
    (OUT/(name+".log")).write_text(proc.stdout+proc.stderr,encoding="utf-8")
    return {"name":name,"passed":proc.returncode==0,"returncode":proc.returncode,"seconds":time.time()-started,"log":name+".log"}
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--components-only",action="store_true",help="run inference and component tests without the separate experimental data archive")
    args=parser.parse_args()
    checks=[]
    manifest=ROOT/"RELEASE_MANIFEST.json"
    if manifest.exists():
        try:
            entries=json.loads(manifest.read_text())["files"]
            for entry in entries:
                actual=hashlib.sha256((ROOT/entry["path"]).read_bytes()).hexdigest()
                assert actual==entry["sha256"],entry["path"]
            checks.append({"name":"all_manifest_file_hashes","passed":True,"files":len(entries)})
        except Exception as exc:
            checks.append({"name":"all_manifest_file_hashes","passed":False,"error":str(exc)})
    try:
        import numpy as np
        import torch
        from src.controller.md3qn import DuelingQNetwork
        from src.controller.state import StateBuilder,StateSchema
        frozen=ROOT/"frozen_controller_360_run"
        checkpoint=frozen/"md3qn_per_live_candidate.pt"
        audit=json.loads((ROOT/"provenance/controller_provenance_audit.json").read_text())
        assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==audit["checkpoint"]["frozen_sha256"]
        schema_path=frozen/"state_schema.json"
        assert hashlib.sha256(schema_path.read_bytes()).hexdigest()==audit["preprocessing_schema"]["frozen_sha256"]
        data=json.loads(schema_path.read_text())
        schema=StateSchema(tuple(data["feature_names"]),np.array(data["means"]),np.array(data["scales"]))
        builder=StateBuilder(schema)
        test_values=dict(zip(schema.feature_names,schema.means,strict=True))
        state=builder.build(test_values)
        model=DuelingQNetwork(len(schema.feature_names)).eval()
        model.load_state_dict(torch.load(checkpoint,map_location="cpu",weights_only=True))
        with torch.no_grad():
            q=model(torch.tensor(state)[None,:]); again=model(torch.tensor(state)[None,:])
        assert q.shape==(1,5) and torch.isfinite(q).all() and torch.equal(q,again)
        (OUT/"frozen_inference.json").write_text(json.dumps({"input_role":"numerical preprocessing smoke, schema mean vector, not measured data","checkpoint_sha256":hashlib.sha256(checkpoint.read_bytes()).hexdigest(),"q_values":q.tolist(),"feature_count":len(state)},indent=2)+"\n")
        checks.append({"name":"frozen_cpu_inference","passed":True})
    except Exception as exc:
        checks.append({"name":"frozen_cpu_inference","passed":False,"error":str(exc)})
    tests=['test_safety_guard.py', 'test_scheduler_readback.py', 'test_xapp_actions.py', 'test_xapp_http_backend.py', 'test_ocudu_websocket_backend.py', 'test_xapp_runtime.py', 'test_recovery_censoring_correction.py', 'test_paired_recovery_statistics.py', 'test_action_parameter_planner.py', 'test_causal_state_imputer.py', 'test_guard_reconciliation_sequence.py', 'test_corrected_guard_integration_analysis.py', 'test_srsue_csv.py', 'test_unbounded_receiver_measurement.py', 'test_receiver_metadata_recovery_sensitivity.py', 'test_release_packaging.py']
    checks.append(run("corrected_components",[sys.executable,"-m","pytest","-q",*["tests/unit/"+name for name in tests]]))
    if not args.components_only:
        checks.append(run("expanded_recovery",[sys.executable,"-m","analysis.recovery_censoring_correction","--output-dir",str(OUT/"expanded_recovery")]))
        checks.append(run("actuator_recovery",[sys.executable,"-m","analysis.actuator_recovery_censoring_correction","--output-dir",str(OUT/"actuator_recovery")]))
        checks.append(run("receiver_metadata_sensitivity",[sys.executable,"-m","analysis.receiver_metadata_recovery_sensitivity","--output-dir",str(OUT/"receiver_metadata_sensitivity")]))
    packages={}
    for name in ("numpy","pandas","scipy","torch","scikit-learn","pyarrow","pytest","PyYAML","joblib"):
        try: packages[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: packages[name]="not installed"
    report={"passed":all(x["passed"] for x in checks),"checks":checks,"python":sys.version,"platform":platform.platform(),"packages":packages,"scope":"offline inference and component behavior" if args.components_only else "offline inference, component behavior and aggregate analysis; no full radio rerun"}
    (OUT/"reproduction_report.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))
    return 0 if report["passed"] else 1
if __name__=="__main__": raise SystemExit(main())
