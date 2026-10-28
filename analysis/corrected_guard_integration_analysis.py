"""Audit a frozen corrected-guard radio study without creating observations.

All windows are repeated measurements within an independently executed run.
Matched scenario/seed blocks, rather than windows, are resampled for uncertainty.
Missing proposals, acknowledgements and measurements remain explicitly missing.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np

ARMS = ("no_action", "policy", "guarded_policy", "fixed_a2")
ACTION_NAMES = ("NO_OP", "PRB_AVOID_SHIFT", "MCS_BACKOFF", "SLICE_PRB_BOOST", "COMPOSITE_LOCAL_DEFENSE")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    # A malformed or unfinished record is an evidence error, not an omitted row.
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def action_name(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, int) and 0 <= value < len(ACTION_NAMES):
        return ACTION_NAMES[value]
    text = str(value)
    if text in ACTION_NAMES:
        return text
    if len(text) == 2 and text[0] == "A" and text[1].isdigit() and int(text[1]) < 5:
        return ACTION_NAMES[int(text[1])]
    return text


def finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (ValueError, TypeError):
        return None
    return number if math.isfinite(number) else None


def describe(values: Iterable[Any]) -> dict[str, Any]:
    supplied = list(values)
    observed = np.array([v for item in supplied if (v := finite(item)) is not None], dtype=float)
    result: dict[str, Any] = {"n_observed": len(observed), "n_missing_or_nonfinite": len(supplied)-len(observed)}
    if len(observed):
        result.update(mean=float(observed.mean()), median=float(np.median(observed)),
                      p95=float(np.quantile(observed,.95)), p99=float(np.quantile(observed,.99)),
                      minimum=float(observed.min()), maximum=float(observed.max()))
    else:
        result.update({name:None for name in ("mean","median","p95","p99","minimum","maximum")})
    return result


def _events(record: dict[str, Any]) -> Iterable[tuple[str, dict[str, Any]]]:
    # Runtime ordering: baseline reconciliation before inference, normal
    # execution, then any invalid-input restoration. These are separate commands.
    for name, value in (
        ("baseline_reconciliation", record.get("baseline_reconciliation")),
        ("decision_execution", record.get("decision",{}).get("execution")),
        ("invalid_evidence_restoration", record.get("restoration")),
    ):
        if isinstance(value,dict):
            yield name,value


def summarize_decisions(records: list[dict[str, Any]], *, expected_windows: int,
                        stage_timings: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    indexes=[int(row["window_index"]) for row in records]
    if len(set(indexes)) != len(indexes):
        raise ValueError("duplicate decision window indexes; raw deliveries cannot be independent events")
    if indexes != sorted(indexes) or any(i<0 or i>=expected_windows for i in indexes):
        raise ValueError("decision window indexes are out of order or outside the frozen run")
    counts: Counter[str] = Counter()
    reasons: Counter[str] = Counter()
    proposal_actions: Counter[str] = Counter()
    authorized_actions: Counter[str] = Counter()
    scopes: Counter[str] = Counter()
    latency: dict[str,list[float|None]] = defaultdict(list)
    state = "UNKNOWN"
    saw_initial_baseline = False
    pending_resumption = False
    unresolved_unknown_episode = False
    unknown_start_ns: int | None = None
    last_execution_ns: int | None = None
    unknown_intervals: list[dict[str,Any]] = []
    transition_rows=[]
    for record in records:
        index=int(record["window_index"])
        decision=record.get("decision") or {}
        guard=record.get("safety_guard")
        proposal=action_name(decision.get("policy_selected_action"))
        selected=action_name(decision.get("selected_action"))
        counts["recorded_windows"]+=1
        counts["decision_windows"]+=bool(decision)
        counts["proposal_unavailable_windows"]+=proposal is None
        counts["state_unverified_no_inference_windows"]+=record.get("status")=="STATE_UNVERIFIED_NO_INFERENCE"
        unbounded=(record.get("receiver_measurement_qualifiers") or {}).get("sinr_db")=="native_positive_infinity_unbounded"
        if unbounded:
            counts["native_unbounded_sinr_windows"]+=1
            provenance=(record.get("telemetry_provenance") or {}).get("sinr_db")
            counts["unbounded_sinr_policy_substitution_windows"]+=provenance in ("previous_measured","training_mean")
            counts["unbounded_sinr_incorrectly_called_measured"]+=provenance=="measured"
        if proposal is not None:
            counts["policy_proposals"]+=1
            proposal_actions[proposal]+=1
            counts["nontrivial_policy_proposals"]+=proposal!="NO_OP"
        if selected is not None and selected!="NO_OP":
            counts["nontrivial_selected_actions"]+=1
            authorized_actions[selected]+=1
        if isinstance(guard,dict):
            reasons[str(guard.get("reason","unrecorded"))]+=1
            counts["guard_allow_policy_windows"]+=guard.get("allow_policy") is True
            counts["guard_eligible_observation_windows"]+=guard.get("observation_eligible") is True
            if proposal is not None and proposal!="NO_OP":
                if guard.get("allow_policy") is True:
                    counts["operationally_supported_nontrivial_proposals"]+=1
                    counts["supported_nontrivial_authorizations"]+=selected==proposal
                elif guard.get("allow_policy") is False:
                    counts["operationally_unsupported_nontrivial_proposals"]+=1
                    counts["intercepted_nontrivial_proposals"]+=selected=="NO_OP"
                    counts["unsupported_nontrivial_selected_actions"]+=selected not in (None,"NO_OP")
        for key in ("guard_computation_ms","baseline_reconciliation_ms","full_loop_processing_ms"):
            if key in record:
                latency[key].append(finite(record[key]))
        for key,value in decision.get("latency_ms",{}).items():
            latency[key].append(finite(value))
        unknown_in_window=record.get("previous_action_state_status")=="UNKNOWN"
        for origin,event in _events(record):
            counts["execution_records"]+=1
            sent=event.get("command_sent") is True
            requested=action_name(event.get("requested_action"))
            counts["commands_sent"]+=sent
            counts["nontrivial_commands_sent"]+=sent and requested not in (None,"NO_OP")
            counts["nontrivial_commands_before_initial_baseline"]+=sent and requested not in (None,"NO_OP") and not saw_initial_baseline
            counts["acknowledged_commands"]+=sent and event.get("acknowledged") is True
            counts["commands_with_unknown_outcome"]+=event.get("command_outcome")=="UNKNOWN"
            counts["rejected_commands"]+=event.get("command_outcome")=="REJECTED"
            counts["parameter_mismatch_executions"]+=event.get("parameter_match") is False
            counts["action_mismatch_executions"]+=event.get("action_match") is False
            counts["restoration_requests"]+=event.get("restoration_requested") is True
            counts["restoration_acknowledgements"]+=event.get("restoration_acknowledged") is True
            counts["verified_restorations"]+=event.get("restoration_verified") is True
            readback=event.get("readback") or {}
            counts["successful_readback_records"]+=readback.get("successful") is True
            if readback:
                scopes[str(readback.get("verification_scope","unrecorded"))]+=1
            after=str(event.get("configuration_state_status","UNKNOWN"))
            applied=action_name(event.get("configuration_state_action",event.get("executed_action")))
            agreement=(after=="VERIFIED" and requested is not None and requested==applied
                       and event.get("requested_parameters",{})==event.get("configuration_state_parameters",event.get("applied_parameters",{})))
            counts["verified_software_configuration_executions"]+=after=="VERIFIED"
            counts["verified_nontrivial_software_configuration_executions"]+=after=="VERIFIED" and applied not in (None,"NO_OP")
            counts["verified_request_parameter_agreements"]+=agreement
            counts["verified_request_parameter_disagreements"]+=after=="VERIFIED" and not agreement
            observed_time=readback.get("timestamp_ns",readback.get("observed_timestamp_ns"))
            complete_identity=(readback.get("source_id") is not None and finite(observed_time) is not None
                               and readback.get("configuration_version") is not None)
            counts["readback_identity_complete_records"]+=bool(readback) and complete_identity
            counts["effective_software_execution_evidence"]+=(agreement and readback.get("successful") is True and complete_identity)
            counts["verified_status_without_successful_readback"]+=after=="VERIFIED" and readback.get("successful") is not True
            completed=event.get("completed_ns")
            if isinstance(completed,int):
                if last_execution_ns is not None and completed<last_execution_ns:
                    raise ValueError("execution monotonic completion timestamps moved backwards")
                last_execution_ns=completed
            if state!="UNKNOWN" and after=="UNKNOWN":
                counts["post_baseline_unknown_entries"]+=1
                if not unresolved_unknown_episode:
                    unknown_start_ns=completed if isinstance(completed,int) else None
                    unresolved_unknown_episode=True
                unknown_in_window=True
            elif after=="UNKNOWN":
                unknown_in_window=True
            if after=="VERIFIED" and applied=="NO_OP":
                if not saw_initial_baseline:
                    counts["initial_baseline_verified"]+=1
                    saw_initial_baseline=True
                elif unresolved_unknown_episode:
                    pending_resumption=True
                    if unknown_start_ns is not None and isinstance(completed,int):
                        unknown_intervals.append({"start_monotonic_ns":unknown_start_ns,"end_monotonic_ns":completed,
                                                  "observed_duration_ms":(completed-unknown_start_ns)/1e6,"right_censored":False})
                    unknown_start_ns=None
                    unresolved_unknown_episode=False
            if after=="VERIFIED" and applied not in (None,"NO_OP") and pending_resumption:
                counts["verified_nontrivial_resumptions_after_restoration"]+=1
                pending_resumption=False
            if after!=state:
                transition_rows.append({"window_index":index,"origin":origin,"before":state,"after":after,
                                        "requested_action":requested,"verified_action":applied if after=="VERIFIED" else None})
            state=after
            if isinstance(event.get("started_ns"),int) and isinstance(completed,int):
                latency[origin+"_actuator_and_readback_ms"].append((completed-event["started_ns"])/1e6)
        counts["windows_with_observed_unknown_state"]+=unknown_in_window
    if unknown_start_ns is not None:
        unknown_intervals.append({"start_monotonic_ns":unknown_start_ns,"end_monotonic_ns":None,
                                  "observed_duration_ms":None if last_execution_ns is None else (last_execution_ns-unknown_start_ns)/1e6,
                                  "right_censored":True})
    elif unresolved_unknown_episode:
        unknown_intervals.append({"start_monotonic_ns":None,"end_monotonic_ns":None,"observed_duration_ms":None,
                                  "right_censored":True,"reason":"UNKNOWN entry lacks a recorded monotonic timestamp"})
    stage_indexes=[]
    for row in stage_timings or []:
        stage_indexes.append(int(row["window_index"]))
        for key in ("logging_ms","full_loop_with_logging_ms"):
            if key in row: latency[key].append(finite(row[key]))
    if len(stage_indexes)!=len(set(stage_indexes)):
        raise ValueError("duplicate stage timing window indexes")
    return {"counts":dict(sorted(counts.items())),"guard_reason_counts":dict(reasons),
            "policy_proposal_action_counts":dict(proposal_actions),"nontrivial_selected_action_counts":dict(authorized_actions),
            "readback_scope_counts":dict(scopes),"decision_window_coverage":len(records)/expected_windows,
            "missing_decision_window_indexes":sorted(set(range(expected_windows))-set(indexes)),
            "final_configuration_state_status":state,"initial_unknown_duration":"left-censored; startup monotonic origin unrecorded",
            "functional_sequence_observations":{"initial_baseline_verified":saw_initial_baseline,
                "verified_nontrivial_execution_observed":counts["verified_nontrivial_software_configuration_executions"]>0,
                "verified_restoration_then_nontrivial_resumption_observed":counts["verified_nontrivial_resumptions_after_restoration"]>0},
            "unknown_intervals":unknown_intervals,"state_transitions":transition_rows,
            "latency_ms":{key:{**describe(values),"expected_run_radio_windows":expected_windows,
                                "windows_without_this_instrumented_stage":expected_windows-len(values)} for key,values in sorted(latency.items())},
            "support_definition":"operational frozen evidence/lockout predicate, not independent ground-truth interference classification",
            "duration_units":"unknown windows are radio windows; command-event intervals use the same monotonic nanosecond clock and represent wall processing time"}


def service_outcomes(windows: list[dict[str,Any]], phases: dict[str,int]) -> dict[str,Any]:
    total=sum(phases.values())
    if len(windows)!=total or [int(row["window_index"]) for row in windows]!=list(range(total)):
        raise ValueError("traffic windows do not match the exact ordered frozen phase population")
    pre,exposure=phases["pre"],phases["interference"]
    before=describe(row.get("protected_throughput_mbps") for row in windows[:pre])
    during=windows[pre:pre+exposure]
    goodput=describe(row.get("protected_throughput_mbps") for row in during)
    if before["n_observed"]!=pre or goodput["n_observed"]!=exposure:
        raise ValueError("missing throughput cannot be silently converted to an outage")
    denominator=before["mean"]
    return {"pre_goodput_mbps":denominator,"exposure_goodput_mbps":goodput["mean"],
            "exposure_goodput_retention":None if denominator is None or denominator<=0 else goodput["mean"]/denominator,
            "exposure_zero_goodput_windows":sum(float(row["protected_throughput_mbps"])==0 for row in during),
            "exposure_loss_percent":describe(row.get("protected_lost_percent") for row in during),
            "exposure_jitter_ms":describe(row.get("protected_jitter_ms") for row in during),
            "recovery_goodput_mbps":describe(row.get("protected_throughput_mbps") for row in windows[pre+exposure:]),
            "radio_windows":total,"exposure_radio_windows":exposure,
            "units":"goodput per measured one-second radio window; no conversion from virtual-radio wall elapsed time"}


def paired_estimate(left: list[float],right: list[float], *, iterations: int=10000,seed: int=20261026) -> dict[str,Any]:
    a,b=np.array(left,float),np.array(right,float)
    if iterations<=0 or a.shape!=b.shape or a.ndim!=1 or len(a)==0 or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("paired estimates require aligned finite run-level values")
    delta=a-b
    interval=None
    if len(delta)>=2:
        rng=np.random.default_rng(seed)
        samples=rng.choice(delta,size=(iterations,len(delta)),replace=True).mean(axis=1)
        interval=np.quantile(samples,[.025,.975]).tolist()
    return {"n_matched_blocks":len(delta),"mean_difference":float(delta.mean()),"median_difference":float(np.median(delta)),
            "paired_block_bootstrap_95ci":interval,"left_lower_blocks":int(sum(delta<0)),"equal_blocks":int(sum(delta==0)),
            "left_higher_blocks":int(sum(delta>0)),"bootstrap_iterations":iterations,"bootstrap_seed":seed,
            "bootstrap_unit":"independent matched scenario-seed block; all repeated windows reduced within run first"}


def _metric(run: dict[str,Any],name: str) -> float | None:
    if name.startswith("loop_"):
        key=name.removeprefix("loop_")
        summary=run.get("control",{}).get("latency_ms",{}).get("full_loop_with_logging_ms",{})
        if summary.get("n_observed")!=summary.get("expected_run_radio_windows"):
            return None
        return finite(summary.get(key))
    if name=="exposure_rlc_delay_p95_ms":
        return finite(run.get("rlc_delay",{}).get("p95"))
    return finite(run.get("service",{}).get(name))


def summarize_pairs(runs: list[dict[str,Any]],expected_blocks: Iterable[str], *, iterations: int=10000) -> dict[str,Any]:
    blocks: dict[str,dict[str,dict[str,Any]]] = defaultdict(dict)
    for run in runs:
        if run.get("measurement_and_identity_verified"):
            if run["arm"] in blocks[run["block"]]: raise ValueError("duplicate independently executed block/arm")
            blocks[run["block"]][run["arm"]]=run
    expected=sorted(set(expected_blocks))
    complete=[key for key in expected if set(blocks[key])==set(ARMS)]
    results=[]
    definitions=[("guarded_policy",baseline,metric) for baseline in ("policy","no_action","fixed_a2")
                 for metric in ("exposure_goodput_mbps","exposure_goodput_retention","exposure_rlc_delay_p95_ms")]
    definitions.extend(("guarded_policy","policy",metric) for metric in ("loop_mean","loop_p95","loop_p99"))
    for number,(left,right,metric) in enumerate(definitions):
        usable=[key for key in complete if _metric(blocks[key][left],metric) is not None and _metric(blocks[key][right],metric) is not None]
        item={"left":left,"right":right,"metric":metric,"used_blocks":usable,"unavailable_metric_blocks":sorted(set(complete)-set(usable))}
        if usable:
            item.update(paired_estimate([_metric(blocks[key][left],metric) for key in usable],
                                        [_metric(blocks[key][right],metric) for key in usable],iterations=iterations,seed=20261026+number))
        else: item.update(n_matched_blocks=0,mean_difference=None,paired_block_bootstrap_95ci=None)
        results.append(item)
    return {"complete_four_arm_blocks":complete,"incomplete_blocks":{key:sorted(set(ARMS)-set(blocks[key])) for key in expected if key not in complete},
            "comparisons":results,"interpretation":"exploratory paired estimation, no confirmatory significance claim or simultaneous coverage; negative goodput delta denotes observed service disadvantage",
            "latency_boundary":"paired per-run full local processing including logging; includes computation, communication and readback but excludes waiting for radio windows; stages are reported separately"}


def summarize_arms(runs: list[dict[str,Any]]) -> dict[str,Any]:
    arms={}
    for arm in ARMS:
        planned=[row for row in runs if row["arm"]==arm]
        available=[row for row in planned if row.get("measurement_and_identity_verified")]
        controlled=[row for row in available if "counts" in row.get("control",{}) and arm!="no_action" and not row.get("controller_evidence_error")]
        totals: Counter[str]=Counter()
        reasons: Counter[str]=Counter()
        for row in controlled:
            totals.update(row["control"]["counts"])
            reasons.update(row["control"]["guard_reason_counts"])
        unsupported=totals["operationally_unsupported_nontrivial_proposals"]
        supported=totals["operationally_supported_nontrivial_proposals"]
        arms[arm]={"planned_runs":len(planned),"measurement_identity_verified_runs":len(available),
                   "control_auditable_runs":len(controlled),"total_counts":dict(totals),"guard_reason_counts":dict(reasons),
                   "interception_fraction_of_operationally_unsupported_proposals":None if not unsupported else totals["intercepted_nontrivial_proposals"]/unsupported,
                   "authorization_fraction_of_operationally_supported_proposals":None if not supported else totals["supported_nontrivial_authorizations"]/supported,
                   "independent_runs_with_nontrivial_resumption":sum(row["control"]["counts"].get("verified_nontrivial_resumptions_after_restoration",0)>0 for row in controlled),
                   "guard_harm_boundary_activations":reasons["post_action_harm_boundary_crossed"],
                   "run_level_service_summaries":{metric:describe(_metric(row,metric) for row in available)
                       for metric in ("exposure_goodput_mbps","exposure_goodput_retention","exposure_rlc_delay_p95_ms")},
                   "rate_interpretation":"counts/proportions describe within-run opportunities; repeated windows are not independent replications"}
    return arms


def build_report(study_root: Path,raw_root: Path, *, phase: str="evaluation",iterations: int=10000) -> dict[str,Any]:
    freeze_path=study_root/f"{phase}_freeze.json"
    if not freeze_path.is_file(): raise FileNotFoundError(f"No frozen {phase} design: {freeze_path}")
    freeze=read_json(freeze_path)
    progress_path=study_root/f"{phase}_progress.json"
    progress=read_json(progress_path) if progress_path.is_file() else {"completed":[]}
    phases={key:int(freeze["phases_seconds"][key]) for key in ("pre","interference","recovery")}
    expected=freeze["rows"]
    keys=[(row[0],int(row[5]),row[6]) for row in expected]
    if len(keys)!=len(set(keys)): raise ValueError("duplicate frozen scenario-seed-arm rows")
    observed={}
    for row in progress.get("completed",[]):
        key=(row["block"],int(row["seed"]),row["arm"])
        if key in observed or key not in keys: raise ValueError("duplicate or unplanned progress row")
        observed[key]=row
    records=[]
    raw_hashes={str(freeze_path):sha256(freeze_path)}
    if progress_path.is_file(): raw_hashes[str(progress_path)]=sha256(progress_path)
    for frozen_row,key in zip(expected,keys,strict=True):
        block,jammer,jsr,load,attenuation,seed,arm=frozen_row
        row=observed.get(key)
        result={"block":block,"seed":seed,"arm":arm,"jammer":jammer,"jsr_db":jsr,"load":load,"attenuation_db":attenuation,
                "status":"NOT_RECORDED","measurement_and_identity_verified":False}
        records.append(result)
        if row is None:
            if freeze.get("run_id_prefix"):
                expected_id=f"{freeze['run_id_prefix']}-{block}-{arm}-s{seed}"
                attempt=raw_root/f"iq-broker-live-{expected_id}"
                if attempt.is_dir():
                    result.update(status="UNCOMPLETED_RAW_ATTEMPT_PRESERVED",run_id=expected_id,raw_attempt_path=str(attempt))
            continue
        if freeze.get("run_id_prefix") and row["run_id"]!=f"{freeze['run_id_prefix']}-{block}-{arm}-s{seed}":
            raise ValueError("progress run identifier differs from its frozen study identity")
        result.update(run_id=row["run_id"],runner_exit_code=row.get("exit_code"),runner_measurements_complete=row.get("measurements_complete"))
        evidence=raw_root/f"iq-broker-live-{row['run_id']}"
        attestation_path=evidence/"corrected_guard_attestation.json"
        traffic_path=evidence/"protected-traffic-status.json"
        if not attestation_path.is_file() or not traffic_path.is_file():
            result["status"]="MISSING_RAW_ATTESTATION_OR_TRAFFIC"; continue
        attest=read_json(attestation_path)
        identity_ok=(attest.get("freeze_sha256")==sha256(freeze_path) and attest.get("source_sha256")==freeze["source_sha256"]
                     and all(attest.get(name)==row.get(name) for name in ("run_id","block","seed","arm")))
        result["source_identity_verified"]=identity_ok
        for path in (attestation_path,traffic_path): raw_hashes[str(path)]=sha256(path)
        traffic=read_json(traffic_path)
        try:
            result["service"]=service_outcomes(traffic["radio_windows"],phases)
        except (KeyError,ValueError,TypeError) as exc:
            result.update(status="TRAFFIC_EVIDENCE_INVALID",error=str(exc)); continue
        result["measurement_and_identity_verified"]=(identity_ok and row.get("measurements_complete") is True
                                                      and traffic.get("complete_radio_windows")==sum(phases.values()) and row.get("exit_code")==0)
        result["status"]="MEASUREMENTS_VERIFIED" if result["measurement_and_identity_verified"] else "MEASUREMENT_OR_IDENTITY_AUDIT_FAILED"
        decision_path=evidence/"live-xapp-decisions.jsonl"
        timing_path=evidence/"loop-stage-timings.jsonl"
        if arm=="no_action":
            result["control"]={"controller_disabled_by_design":True,"counts":{},"latency_ms":{}}
        else:
            try:
                decisions=read_jsonl(decision_path)
                timings=read_jsonl(timing_path)
                result["control"]=summarize_decisions(decisions,expected_windows=sum(phases.values()),stage_timings=timings)
                result["controller_window_coverage_complete"]=len(decisions)==sum(phases.values())
                if not decisions: result["controller_evidence_error"]="no recorded decisions; no interception or execution success inferred"
            except (ValueError,KeyError,TypeError) as exc:
                result["controller_evidence_error"]=str(exc)
            for path in (decision_path,timing_path):
                if path.is_file(): raw_hashes[str(path)]=sha256(path)
        parquet=evidence/"records.parquet"
        if parquet.is_file():
            try:
                import pandas as pd
                frame=pd.read_parquet(parquet)
                if len(frame)!=sum(phases.values()) or sorted(frame.window_index.astype(int))!=list(range(sum(phases.values()))):
                    raise ValueError("RLC records do not preserve the frozen complete radio-window population")
                exposure=frame[(frame.window_index>=phases["pre"])&(frame.window_index<phases["pre"]+phases["interference"])]
                result["rlc_delay"]=describe(exposure.rlc_sdu_delay_ms.tolist())
                result["rlc_delay"]["observability_counts"]=dict(Counter(exposure.telemetry_observability.astype(str))) if "telemetry_observability" in exposure else {}
                if "sinr_positive_infinity_observed" in frame:
                    result["receiver_measurement_audit"]={
                        "native_unbounded_sinr_windows":int((frame.sinr_positive_infinity_observed==1).sum()),
                        "finite_sinr_windows":int(np.isfinite(frame.sinr_db).sum()),
                        "fully_finite_telemetry_windows":int(frame.complete_telemetry.sum()),
                        "evidence_complete_windows":int(frame.evidence_complete.sum()),
                        "rule":"unbounded native SNR is not a finite measurement; evidence completeness is not identical to a fully finite controller state"}
                raw_hashes[str(parquet)]=sha256(parquet)
            except (KeyError,ValueError,AttributeError,ImportError) as exc: result["rlc_delay_error"]=str(exc)
        resource_path=evidence/"container-resource-snapshot.jsonl"
        if resource_path.is_file():
            try:
                result["resource_snapshot"]={"scope":"terminal container snapshot, not continuous or per-controller process attribution",
                    "containers":[{key:item.get(key) for key in ("Name","Container","CPUPerc","MemPerc","MemUsage","PIDs")}
                                   for item in read_jsonl(resource_path)]}
                raw_hashes[str(resource_path)]=sha256(resource_path)
            except ValueError as exc: result["resource_snapshot_error"]=str(exc)
    return {"schema_version":1,"phase":phase,"analysis_role":"exploratory integrated corrected-guard analysis; historical controller and actuator cohorts remain separate",
            "population":{"expected_runs":len(expected),"recorded_runs":len(observed),"measurement_identity_verified_runs":sum(row["measurement_and_identity_verified"] for row in records),
                          "expected_radio_windows_per_run":sum(phases.values()),"expected_blocks":len({row[0] for row in expected}),"phases_radio_seconds":phases},
            "freeze_sha256":sha256(freeze_path),"frozen_analysis_plan":freeze.get("analysis_plan"),"input_sha256":raw_hashes,"runs":records,
            "arm_summaries":summarize_arms(records),
            "paired_effects":summarize_pairs(records,[row[0] for row in expected],iterations=iterations),
            "completeness":"COMPLETE" if len(observed)==len(expected) and all(row["measurement_and_identity_verified"] for row in records) else "INCOMPLETE_PRESERVED",
            "measurement_limits":["scheduler readback verifies consumed software configuration, not physical radio effect","missing/nonfinite BLER is not filled for the guard; skipped or uncertain windows remain controller outcomes","RLC delay distribution is conditional on finite observed values; outages/missingness reported separately","stage-specific readback and actuator communication times are combined by current instrumentation"]}


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root",type=Path,default=Path("results/corrected_guard_integration/20261027_final"))
    parser.add_argument("--raw-root",type=Path,default=Path("results/raw"))
    parser.add_argument("--phase",choices=("smoke","fault","evaluation"),default="evaluation")
    parser.add_argument("--output",type=Path,default=Path("revision_2026-10-23/analysis_outputs/corrected_guard_integration_report.json"))
    parser.add_argument("--bootstrap-iterations",type=int,default=10000)
    args=parser.parse_args()
    report=build_report(args.study_root,args.raw_root,phase=args.phase,iterations=args.bootstrap_iterations)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    csv_path=args.output.with_suffix(".per_run.csv")
    fields=("run_id","block","seed","arm","jammer","jsr_db","load","attenuation_db","status","measurement_and_identity_verified")
    with csv_path.open("w",encoding="utf-8",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=fields); writer.writeheader()
        writer.writerows({key:row.get(key) for key in fields} for row in report["runs"])
    print(json.dumps({"population":report["population"],"completeness":report["completeness"],"output":str(args.output)},indent=2))
    return 0


if __name__=="__main__": raise SystemExit(main())
