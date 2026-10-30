"""Bounded, version-attested matched radio evaluation of the corrected guard."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
from datetime import datetime, timezone

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"results/corrected_guard_integration/20261027_final"
STUDY_ID="corrected_guard_20261027_final"
RUN_ID_PREFIX="cg-20261027"
ACCEPTANCE_PROFILE="observed_outage_censoring_v1"
AMENDMENT=None
SOURCE_FILES=("src/controller/safety_guard.py","src/controller/action_mask.py","src/controller/state.py","src/controller/md3qn.py","src/xapp/adapters/actions.py","src/xapp/adapters/ocudu_websocket.py","src/xapp/adapters/parameter_planner.py","src/xapp/runtime/loop.py","experiments/live/run_xapp_loop.py","tools/run_live_pilot_scenario.sh","tools/run_live_registration.sh","tools/run_corrected_guard_integration.py","analysis/corrected_guard_integration_analysis.py","configs/experiments/protocol.yaml","config/baseline/ocudu_gnb_zmq_e2_udp_um.yml")
UE_IMAGE="srsue-zmq:telemetry-nr-attachment-20261027"
ARMS=("no_action","policy","guarded_policy","fixed_a2")
STRATA=(
    ("none",0,"low",0.), ("none",0,"high",6.),
    ("partial_band",0,"low",0.), ("partial_band",5,"high",0.),
    ("barrage",5,"low",6.), ("intermittent",5,"high",0.),
)


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+".tmp"); tmp.write_text(json.dumps(data,indent=2)+"\n"); tmp.replace(path)


def fault_sequence(decisions):
    lost = [i for i,d in enumerate(decisions) if "predeclared_integration_fault" in str(d)]
    if not lost:
        return {"status":"INCONCLUSIVE_NO_NONTRIVIAL_ACK_FAULT", "passed":False}
    i=lost[0]
    first=decisions[i].get("decision",{}).get("execution",{})
    unknown=first.get("configuration_state_status")=="UNKNOWN"
    restored=[]
    for j in range(i+1,len(decisions)):
        restoration=decisions[j].get("baseline_reconciliation") or {}
        if restoration.get("restoration_verified"):
            restored.append(j)
    resumed=[]
    if restored:
        for j in range(restored[0]+1,len(decisions)):
            e=decisions[j].get("decision",{}).get("execution",{})
            if e.get("executed_action") not in (None,"NO_OP") and e.get("configuration_state_status")=="VERIFIED": resumed.append(j)
    passed=unknown and bool(restored) and bool(resumed)
    return {"status":"PASS" if passed else "INCOMPLETE_FUNCTIONAL_RECOVERY", "passed":passed,"lost_ack_window":i,"verified_baseline_windows":restored,"resumed_nontrivial_windows":resumed}


def design(phase):
    if phase=="smoke": return [("smoke-none","none",0,"low",0.,9301,"guarded_policy"),("smoke-partial","partial_band",5,"low",0.,9302,"guarded_policy")]
    if phase=="fault": return [("fault-partial","partial_band",5,"low",0.,9401,"guarded_policy")]
    rows=[]
    for j,(jammer,jsr,load,attenuation) in enumerate(STRATA):
        for replicate in range(2):
            block=f"cg-b{j+1:02d}-r{replicate+1}"
            seed=9501+j*2+replicate
            arms=list(ARMS); random.Random(seed).shuffle(arms)
            rows.extend((block,jammer,jsr,load,attenuation,seed,arm) for arm in arms)
    return rows


def main():
    from dataclasses import asdict
    if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
    from src.controller.safety_guard import SafetyGuardConfig
    p=argparse.ArgumentParser(); p.add_argument("--phase",choices=("smoke","fault","evaluation"),required=True); p.add_argument("--image",default="ocudu-gnb-zmq:consumer-readback-20261023"); p.add_argument("--ue-image",default=UE_IMAGE); p.add_argument("--limit",type=int,default=0); a=p.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    rows=design(a.phase)
    freeze=OUT/f"{a.phase}_freeze.json"
    identity={f:sha(ROOT/f) for f in SOURCE_FILES}
    artifacts={"controller_sha256":sha(ROOT/"results/training/live_controller_recalibrated/md3qn_per_live_candidate.pt"),
               "schema_sha256":sha(ROOT/"results/training/live_controller_recalibrated_v2/state_schema.json"),
               "detector_sha256":sha(ROOT/"results/training/live_detector/detector_live_calibrated.joblib"),
               "image":a.image,"image_id":subprocess.check_output(["sudo","docker","image","inspect",a.image,"--format","{{.Id}}"],text=True).strip(),
               "ue_image":a.ue_image,"ue_image_id":subprocess.check_output(["sudo","docker","image","inspect",a.ue_image,"--format","{{.Id}}"],text=True).strip(),
               "guard_configuration":asdict(SafetyGuardConfig())}
    if freeze.exists():
        old=json.loads(freeze.read_text())
        if old["source_sha256"]!=identity: raise RuntimeError("frozen implementation changed; preserve evidence and stop")
        if any(old.get(k)!=v for k,v in artifacts.items()): raise RuntimeError("frozen model, preprocessing, detector or image identity changed")
    else:
        write(freeze,{"created_utc":datetime.now(timezone.utc).isoformat(),"phase":a.phase,"study_identity":STUDY_ID,"run_id_prefix":RUN_ID_PREFIX,"source_sha256":identity,
                      "measurement_amendment":AMENDMENT,"acceptance_profile":ACCEPTANCE_PROFILE,
                      **artifacts,
                      "rows":rows,"phases_seconds":{"pre":15,"interference":20,"recovery":10},
                      "analysis_role":"prospective bounded exploratory integrated evaluation, independent from all historical cohorts",
                      "replication_rationale":"12 independent matched scenario-seed blocks across six strata, two seeds per stratum; four arms give 48 runs and 2160 radio windows. Resource-bounded estimation rather than a superiority-powered study. At n=12, an all-success run-level rate has two-sided exact 95% lower limit 0.735. Report interval width instead of claiming broad robustness.",
                      "precision_target":"Estimate paired run-level goodput and latency effects with bootstrap 95% intervals; no threshold tuning or sample-size stopping based on effects. Prior 24-block actuator variability is supplied for context, not treated as guard variance.",
                      "analysis_plan":{
                          "role":"exploratory estimation; no confirmatory superiority claim",
                          "independent_unit":"matched scenario-seed block; 12 blocks, four independently executed arms; radio windows are repeated observations",
                          "primary_service_estimates":["mean protected goodput across 20 exposure radio windows","goodput retention relative to the same run's 15 pre-exposure radio windows","95th percentile finite observed exposure RLC SDU delay with missingness and outage counts separately"],
                          "service_pairs":["guarded_policy minus policy","guarded_policy minus no_action","guarded_policy minus fixed_a2"],
                          "latency_pair":"guarded_policy minus policy, comparing each run's mean, p95 and p99 full local processing including logging",
                          "timing_boundary":"radio-window waiting excluded; command communication and scheduler readback combined unless separately instrumented; complete stage coverage reported",
                          "uncertainty":"10,000 matched-block bootstrap resamples; paired mean differences; percentile 2.5/97.5 limits; deterministic seeds 20261026 plus comparison index; fewer than two blocks yields no confidence interval",
                          "multiplicity":"all intervals are exploratory pointwise 95% estimates; no simultaneous-coverage or familywise-significance claim; all twelve planned estimates reported",
                          "controller_outcomes":["policy proposals","operational-predicate support and interception","authorization/command/acknowledgement counts","software readback and parameter agreement","UNKNOWN windows and monotonic command-event intervals","baseline restoration and later verified nontrivial resumption"],
                          "harm":"report guard harm-boundary activations and direction of matched service differences; no post hoc favorable-threshold selection",
                          "missingness":"unavailable proposals never become A0; nonfinite measured guard inputs are not imputed for admission; missing control records, incomplete measurements and failed controller outcomes retained",
                          "analysis_source":"analysis/corrected_guard_integration_analysis.py"},
                      "fault":"none" if a.phase!="fault" else "discard the first nontrivial command acknowledgement after real application; all radio evidence retained; require UNKNOWN, verified A0 restoration and later verified nontrivial execution",
                      "acceptance":"measurement completeness only; controller failures/lockout/skips are outcomes and never removed to improve performance",
                      "retry_policy":"no automatic retries; incomplete attempts are archived and require an infrastructure diagnosis"})
    progress=OUT/f"{a.phase}_progress.json"
    state=json.loads(progress.read_text()) if progress.exists() else {"phase":a.phase,"completed":[],"expected_runs":len(rows)}
    done={r["run_id"] for r in state["completed"]}
    for block,jammer,jsr,load,attenuation,seed,arm in rows[:a.limit or None]:
        run_id=f"{RUN_ID_PREFIX}-{block}-{arm}-s{seed}"
        if run_id in done: continue
        if any(sha(ROOT/f)!=digest for f,digest in identity.items()): raise RuntimeError("implementation changed during frozen evaluation")
        evidence=ROOT/"results/raw"/f"iq-broker-live-{run_id}"
        if evidence.exists(): raise RuntimeError(f"unclassified prior attempt retained at {evidence}")
        protected,background=(.3,2.1) if load=="low" else (.9,6.3)
        env={"RUN_ID":run_id,"SCENARIO_ID":block,"JAMMER":jammer,"JSR_DB":str(jsr),"SEED":str(seed),"TRAFFIC_LOAD":load,"SERVICE_MIX":"protected_plus_background","ATTENUATION_DB":str(attenuation),
             "RUN_SECONDS":"45","PRE_JAM_SECONDS":"15","JAMMER_ON_SECONDS":"20","RECOVERY_SECONDS":"10","PROTECTED_MBPS":str(protected),"BACKGROUND_MBPS":str(background),
             "EVIDENCE_CLASS":"prospective_corrected_guard_integration","GNB_IMAGE":a.image,"UE_IMAGE":a.ue_image,"GNB_CONFIG":"ocudu_gnb_zmq_e2_udp_um.yml","MAX_WALL_SECONDS":"240",
             "REQUIRE_COMPLETE_TELEMETRY":"0","REQUIRE_EVIDENCE_COMPLETE":"1","REQUIRE_PROTECTED_TRAFFIC":"1","ACCEPTANCE_PROFILE":ACCEPTANCE_PROFILE,"OOD_GUARD_PROFILE":"physical_plausibility_with_shift_audit","UE_TIMEOUT_MS":"5000","CAPTURE_FULL_CONTAINER_LOGS":"1","ALLOW_UPLINK_SILENCE":"0",
             "ENABLE_LIVE_XAPP":"0" if arm=="no_action" else "1","LIVE_XAPP_XAI":"0","CONTROLLER_KIND":"md3qn_per",
             "DETECTOR_MODEL":str(ROOT/"results/training/live_detector/detector_live_calibrated.joblib"),"CONTROLLER_MODEL":str(ROOT/"results/training/live_controller_recalibrated/md3qn_per_live_candidate.pt"),"STATE_SCHEMA":str(ROOT/"results/training/live_controller_recalibrated_v2/state_schema.json"),
             "LIVE_XAPP_SELECTION_MODE":{"policy":"policy","guarded_policy":"safeguarded_policy","fixed_a2":"bounded_action_probe","no_action":"policy"}[arm]}
        if arm=="fixed_a2": env["LIVE_XAPP_FORCED_ACTION_ID"]="2"
        if AMENDMENT is not None: env["ORAN_RECEIVER_MEASUREMENT_PROFILE"]="native_positive_infinity_v1"
        if a.phase=="fault": env["ORAN_TEST_DROP_FIRST_NONTRIVIAL_ACK"]="1"
        started=time.time()
        log=OUT/f"{run_id}.log"
        with log.open("w") as stream:
            result=subprocess.run(["sudo","env",*[f"{k}={v}" for k,v in env.items()],"bash","tools/run_live_pilot_scenario.sh"],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,timeout=360,check=False)
        traffic_path=evidence/"protected-traffic-status.json"
        traffic=json.loads(traffic_path.read_text()) if traffic_path.exists() else {}
        decisions_path=evidence/"live-xapp-decisions.jsonl"
        decisions=[json.loads(line) for line in decisions_path.read_text().splitlines()] if decisions_path.exists() else []
        measurements_complete=result.returncode==0 and traffic.get("complete_radio_windows")==45
        row={"run_id":run_id,"block":block,"arm":arm,"seed":seed,"jammer":jammer,"jsr_db":jsr,"load":load,"attenuation_db":attenuation,"exit_code":result.returncode,"measurements_complete":measurements_complete,"recorded_decisions":len(decisions),"wall_seconds":time.time()-started,"source_sha256":identity,"freeze_sha256":sha(freeze)}
        if a.phase=="fault": row["fault_recovery_sequence"]=fault_sequence(decisions)
        write(evidence/"corrected_guard_attestation.json",row)
        state["completed"].append(row); state["updated_utc"]=datetime.now(timezone.utc).isoformat(); write(progress,state)
        print(json.dumps(row),flush=True)
        if not measurements_complete: raise RuntimeError(f"measurement failure; no retry performed: {run_id}")
    state["status"]="COMPLETE" if len({r["run_id"] for r in state["completed"]})==len(rows) else "PARTIAL_LIMIT"; write(progress,state)


if __name__=="__main__": main()
