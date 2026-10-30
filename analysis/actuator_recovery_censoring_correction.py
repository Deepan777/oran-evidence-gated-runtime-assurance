"""Reanalyse the frozen 84-row actuator study at its actual five-second horizon."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from analysis.recovery_censoring_correction import paired_binary
from analysis.statistics import bootstrap_ci, holm_bonferroni, planned_wilcoxon


def derive(windows: list[dict], pre: int = 10, jam: int = 15, horizon: int = 5) -> dict:
    if len(windows) != pre+jam+horizon:
        raise ValueError("actuator run must preserve the frozen 10/15/5 phase split")
    baseline = np.mean([float(w["protected_throughput_mbps"]) for w in windows[:pre]])
    if not np.isfinite(baseline) or baseline <= 0:
        raise ValueError("recovery denominator is not a positive observed pre-interference mean")
    recovered = [i+1 for i,w in enumerate(windows[pre+jam:])
                 if np.isfinite(float(w["protected_throughput_mbps"])) and float(w["protected_throughput_mbps"]) >= .9*baseline]
    event_time = recovered[0] if recovered else None
    return {"recovery_event_by_5s": int(bool(recovered)), "event_time_seconds": event_time,
            "censor_time_seconds": None if recovered else horizon,
            "restricted_time_to_recovery_5s": horizon if event_time is None else event_time,
            "pre_goodput_mbps": float(baseline), "threshold_mbps": .9*float(baseline)}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--evidence-root", type=Path, default=Path("results/reviewer_validation_r1_release/extracted"))
    parser.add_argument("--output-dir", type=Path, default=Path("revision_2026-10-23/analysis_outputs"))
    args=parser.parse_args()
    source=args.evidence_root/"results/reviewer_validation_r1/REVIEWER_VALIDATION_R1_REPORT.json"
    old=json.loads(source.read_text())
    rows=[]
    for run in old["runs"]:
        if run["status"] != "PASS":
            raise ValueError("the frozen accepted actuator population must remain 84 rows")
        path=args.evidence_root/"results/raw"/f"iq-broker-live-{run['run_id']}"/"protected-traffic-status.json"
        windows=json.loads(path.read_text())["radio_windows"]
        rows.append({"run_id":run["run_id"], "block_id":run["block_id"], "method":run["method"],
                     "old_assigned_recovery_seconds":run["recovery_to_90pct_seconds"], **derive(windows)})
    frame=pd.DataFrame(rows)
    if len(frame)!=84 or frame.block_id.nunique()!=24:
        raise ValueError("actuator population cardinality changed")
    comparisons=[]
    for j,method in enumerate(("A1","A2","A3","A4")):
        matched=frame[frame.method.isin(["A0",method])].pivot(index="block_id",columns="method")
        matched=matched.dropna(subset=[("recovery_event_by_5s",method)])
        a=matched["recovery_event_by_5s"][method].to_numpy(dtype=int)
        b=matched["recovery_event_by_5s"]["A0"].to_numpy(dtype=int)
        delta=matched["restricted_time_to_recovery_5s"][method].to_numpy(float)-matched["restricted_time_to_recovery_5s"]["A0"].to_numpy(float)
        comparisons.append({"method":method,"reference":"A0", **paired_binary(a,b,seed=5200+j),
                            "restricted_mean_difference_seconds":float(delta.mean()),
                            "restricted_mean_paired_bootstrap_95ci":list(bootstrap_ci(delta,iterations=20000,seed=5300+j)),
                            "rmtr_wilcoxon":planned_wilcoxon(matched["restricted_time_to_recovery_5s"][method],matched["restricted_time_to_recovery_5s"]["A0"])})
    for item,adjusted in zip(comparisons,holm_bonferroni([c["p_value"] for c in comparisons]),strict=True):
        item["holm_within_recovery_endpoint_family"]=adjusted
    report={"schema_version":1,"analysis_status":"RETROSPECTIVE_CENSORING_CORRECTION",
            "source_report_sha256":hashlib.sha256(source.read_bytes()).hexdigest(),"runs":84,"matched_blocks":24,
            "definition":{"original_prespecified_threshold":"first recovery window >=90% of run-specific pre-interference mean goodput; throughput-only, no three-window or delay requirement",
                          "time_origin":"cessation of interference, start of first recovery window","event_time":"end of first qualifying one-second window",
                          "horizon_seconds":5,"non_event_handling":"right-censored at 5 s; former assigned 6-s values are not events",
                          "restricted_mean":"E[min(T,5s)], arithmetic mean under common complete five-second follow-up",
                          "retrospective_changes":"explicit event/censor records, horizon-correct RMTR, exact McNemar test for recovery probability; original threshold preserved",
                          "multiplicity":"four A1/A2/A3/A4 versus A0 exact paired recovery tests Holm-adjusted within the recovery endpoint family, as the frozen plan specifies endpoint-wise families; RMTR is supplementary",
                          "bootstrap":"Binary recovery probability: 10,000 paired matched-block resamples, seeds 5200..5203. Supplementary RMTR: 20,000 paired matched-block resamples, seeds 5300..5303. Both use 2.5/97.5 percentile bounds; all methods within each comparison remain paired."},
            "method_summaries":{method:{"runs":len(g),"events":int(g.recovery_event_by_5s.sum()),"censored":int((1-g.recovery_event_by_5s).sum()),"restricted_mean_5s":float(g.restricted_time_to_recovery_5s.mean())} for method,g in frame.groupby("method")},
            "paired_comparisons":comparisons,
            "preserved_other_endpoints":old["endpoint_summaries"],
            "plan_typo":"The plan's descriptive goodput sentence says 20 seconds, but the frozen phase manifest and actual records use 15 interference seconds. Analyses retain the actual 15 seconds."}
    args.output_dir.mkdir(parents=True,exist_ok=True)
    frame.to_csv(args.output_dir/"actuator_recovery_corrected_per_run.csv",index=False)
    (args.output_dir/"actuator_recovery_corrected_report.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({"methods":report["method_summaries"],"paired":comparisons},indent=2))


if __name__=="__main__": main()
