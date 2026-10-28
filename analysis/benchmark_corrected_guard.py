"""Version-attested paired timing of the corrected local guard computation."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
import tracemalloc
import numpy as np
from src.controller.action_mask import Action
from src.controller.safety_guard import SafetyGuardConfig, SafetyObservation, UncertaintySafetyGuard
from analysis.statistics import bootstrap_ci

ROOT=Path(__file__).resolve().parents[1]


def trace(n):
    for i in range(n):
        joint=i>=10 and i%15<8
        yield SafetyObservation(i,"benchmark-sensor","run-1",i,str(i),(i+1)*10**9,True,True,.2 if joint else .01,.6 if joint else .1,.7 if joint else 1.,0.,1.,Action.NO_OP)


def bypass(item):
    return bool(item.protected_session_active and item.spectral_reference_ready and np.isfinite(item.throughput_mbps))


def worker(n,order):
    items=list(trace(n))
    warm=UncertaintySafetyGuard()
    for item in items[:min(1000,n)]: warm.decide(item); bypass(item)
    samples={}
    for name in order.split(","):
        guard=UncertaintySafetyGuard()
        fn=guard.decide if name=="guard" else bypass
        lat=[]
        for item in items:
            start=time.perf_counter_ns(); fn(item); lat.append((time.perf_counter_ns()-start)/1e6)
        samples[name]=lat
    return samples


def summary(values):
    v=np.asarray(values,float)
    return {"mean_ms":float(v.mean()),"median_ms":float(np.median(v)),"p95_ms":float(np.quantile(v,.95)),"p99_ms":float(np.quantile(v,.99))}


def main():
    p=argparse.ArgumentParser(); p.add_argument("--worker",action="store_true"); p.add_argument("--decisions",type=int,default=5000); p.add_argument("--order",default="guard,bypass"); p.add_argument("--repetitions",type=int,default=30); p.add_argument("--output",type=Path,default=ROOT/"revision_2026-10-23/analysis_outputs/corrected_guard_benchmark.json"); a=p.parse_args()
    if a.worker: print(json.dumps(worker(a.decisions,a.order))); return
    reps=[]; pooled={"guard":[],"bypass":[]}
    for i in range(a.repetitions):
        order="guard,bypass" if i%2==0 else "bypass,guard"
        result=subprocess.run([sys.executable,"-m","analysis.benchmark_corrected_guard","--worker","--decisions",str(a.decisions),"--order",order],cwd=ROOT,check=True,capture_output=True,text=True)
        raw=json.loads(result.stdout)
        g,b=np.mean(raw["guard"]),np.mean(raw["bypass"])
        reps.append({"process_repetition":i,"order":order,"guard_mean_ms":float(g),"bypass_mean_ms":float(b),"paired_increment_ms":float(g-b)})
        for name in pooled: pooled[name].extend(raw[name])
    tracemalloc.start(); before=tracemalloc.get_traced_memory()[1]; guard=UncertaintySafetyGuard()
    for item in trace(1000): guard.decide(item)
    peak=tracemalloc.get_traced_memory()[1]-before; tracemalloc.stop()
    files=["src/controller/safety_guard.py","src/xapp/adapters/actions.py","src/xapp/adapters/ocudu_websocket.py","experiments/live/run_xapp_loop.py"]
    out={"schema_version":1,"scope":"guard.decide only on identical unique measured-input harness observations; no policy, preprocessing, logging, adapter, network, readback, or radio effects included",
         "implementation_sha256":{f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files},"configuration":asdict(SafetyGuardConfig()),
         "repetitions":a.repetitions,"independent_unit":"fresh Python process, order counterbalanced","decisions_per_repetition":a.decisions,
         "individual_decision_timing":{name:summary(v) for name,v in pooled.items()},"raw_repetition_means":reps,
         "paired_increment_between_process_repetitions":summary([r["paired_increment_ms"] for r in reps]),
         "paired_increment_mean_95ci":list(bootstrap_ci([r["paired_increment_ms"] for r in reps],seed=20261023)),
         "allocation_peak_increment_bytes_1000_decisions":peak,
         "environment":{"python":sys.version,"platform":platform.platform(),"processor":platform.processor(),"clock":"perf_counter_ns"}}
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(out,indent=2)+"\n")
    print(json.dumps({k:out[k] for k in ("paired_increment_between_process_repetitions","paired_increment_mean_95ci","individual_decision_timing")},indent=2))


if __name__=="__main__": main()
