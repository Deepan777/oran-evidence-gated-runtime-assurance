"""Retrospective record-only audit. Never runs a radio, controller, or benchmark."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import tarfile
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def summary(values):
    a=np.asarray(values,dtype=float)
    return dict(n=len(a),mean=float(a.mean()),median=float(np.median(a)),p99=float(np.percentile(a,99)),maximum=float(a.max()))

def write_json(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def jsonlines(path):return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

def run(args):
    out=args.output;out.mkdir(parents=True,exist_ok=True)
    inputs={}
    def register(path):
        if path.is_relative_to(args.m1_root):
            name='data/local_measurement_amendment1/'+path.relative_to(args.m1_root).as_posix()
        else:
            name=str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else path.as_posix()
        inputs[name]=sha(path)
    columns=['run_id','method','source_run_id','phase','window_index','window_start_ns','window_end_ns','dl_throughput_mbps','rlc_sdu_delay_ms','raw_ocudu_rlc_json','raw_ocudu_scheduler_json']
    register(args.expanded_records)
    records=pd.read_parquet(args.expanded_records,columns=columns)
    records['native_rlc_tx_num_sdus']=records.raw_ocudu_rlc_json.map(lambda s:json.loads(s).get('payload',json.loads(s))['rlc_metrics']['tx']['num_sdus'])
    recovery=records.loc[records.phase.eq('recovery')]
    native_presence=[]
    for row in recovery.itertuples():
        data=json.loads(row.raw_ocudu_scheduler_json);data=data.get('payload',data)
        ues=[ue for cell in data.get('cells',[]) for ue in cell.get('ue_list',[])]
        native_presence.append(dict(run_id=row.run_id,window_index=row.window_index,method=row.method,scheduler_ue_entries=len(ues),rntis=','.join(str(ue.get('rnti')) for ue in ues),dl_bs_sum=sum(float(ue.get('dl_bs',0)) for ue in ues)))
    scheduler=pd.DataFrame(native_presence)
    scheduler.to_csv(out/'a1_recovery_scheduler_presence.csv',index=False)
    per_run=[];arms=[]
    for method in ['B0','B2','B5','P']:
        manifest=args.manifests/f'{method}_live_validation_runs.csv';register(manifest)
        m=pd.read_csv(manifest)
        group=records.loc[records.method.eq(method)];r=group.loc[group.phase.eq('recovery')]
        assert len(m)==90 and group.run_id.nunique()==90 and len(group)==10800 and len(r)==1800
        info=dict(method=method,runs=90,recovery_windows=len(r),live_xapp_flags=sorted(m.ENABLE_LIVE_XAPP.unique().astype(int).tolist()),selection_modes=sorted(m.LIVE_XAPP_SELECTION_MODE.dropna().unique().tolist()),controller_kinds=sorted(m.CONTROLLER_KIND.dropna().unique().tolist()),positive_throughput_windows=int((r.dl_throughput_mbps>0).sum()),positive_native_rlc_sdu_windows=int((r.native_rlc_tx_num_sdus>0).sum()),finite_delay_windows=int(np.isfinite(r.rlc_sdu_delay_ms).sum()),recovery_windows_with_scheduler_ue_entry=int((scheduler.loc[scheduler.method.eq(method)].scheduler_ue_entries>0).sum()))
        arms.append(info)
        for rid,run in group.groupby('run_id'):
            run=run.sort_values('window_index');assert run.window_index.tolist()==list(range(120))
            rec=run.loc[run.phase.eq('recovery')]
            positive=(run.dl_throughput_mbps.to_numpy()>0)|(run.native_rlc_tx_num_sdus.to_numpy()>0)
            tail=int(np.where(positive)[0][-1]+1) if positive.any() else 0
            per_run.append(dict(run_id=rid,method=method,last_positive_throughput_or_sdu_window=int(np.where(positive)[0][-1]) if positive.any() else None,zero_throughput_and_sdu_tail_start_window=tail if tail<120 else None,pre_mean_throughput_mbps=float(run.loc[run.phase.eq('pre_jam'),'dl_throughput_mbps'].mean()),recovery_positive_throughput_windows=int((rec.dl_throughput_mbps>0).sum()),recovery_positive_sdu_windows=int((rec.native_rlc_tx_num_sdus>0).sum()),measured_window_wall_span_seconds=float((run.window_end_ns.max()-run.window_end_ns.min())/1e9)))
    pd.DataFrame(per_run).to_csv(out/'a1_per_run.csv',index=False)
    register(args.raw_manifest);raw=json.loads(args.raw_manifest.read_text())
    source_run_counts={method:sum(r['method']==method for r in raw['runs']) for method in ['B0','B2','B5','P']}
    available={method:0 for method in source_run_counts}
    for r in raw['runs']:
        available[r['method']]+=int((ROOT/r['evidence_directory']/'run_metadata.json').is_file())
    register(args.frozen_loop)
    assert sha(args.frozen_loop)=='da537592f7894b6aeed49dd13849eaeacb221f5ca7427babaa81a30f3374c572'
    frozen=args.frozen_loop.read_text()
    a1=dict(arms=arms,raw_manifest_run_counts=source_run_counts,locally_available_original_run_metadata_counts=available,frozen_loop_sha256=sha(args.frozen_loop),frozen_loop_has_phase_input=('phase' in frozen),frozen_loop_has_explicit_post_loop_reset=False,causal_conclusion='Shared live-xApp-path association, not isolated learned-policy failure. Specific UE/PDU, scheduler, traffic, or harness cause cannot be determined from locally available records. Historical receiver attachment flag is invalid for NR. Frozen loop has no interference-phase restoration boundary; actual recovery-period applied state is unavailable locally.',frozen_outputs_modified=False)
    write_json(out/'a1_expanded_common_cause.json',a1)

    fault=args.m1_root/'results/raw/iq-broker-live-cg-20261027m1-fault-partial-guarded_policy-s9401'
    decision_path=fault/'live-xapp-decisions.jsonl';timing_path=fault/'loop-stage-timings.jsonl'
    attestation=json.loads((fault/'corrected_guard_attestation.json').read_text())
    register(fault/'corrected_guard_attestation.json')
    for relative in ('experiments/live/run_xapp_loop.py','src/xapp/adapters/ocudu_websocket.py','src/controller/safety_guard.py'):
        assert sha(ROOT/relative)==attestation['source_sha256'][relative], relative
    register(decision_path);register(timing_path)
    decisions=jsonlines(decision_path);timings={r['window_index']:r for r in jsonlines(timing_path)}
    stages=[]
    for d in decisions:
        t=d['decision']['latency_ms'];logging=timings[d['window_index']]['logging_ms']
        row=dict(window_index=d['window_index'],full_with_logging_ms=timings[d['window_index']]['full_loop_with_logging_ms'],logging_ms=logging,guard_ms=d['guard_computation_ms'],preprocessing_ms=t['preprocessing_latency'],inference_ms=t['controller_inference_latency'],masking_ms=t['action_masking_latency'],runtime_total_ms=t['total_end_to_end_loop_latency'],command_send_ack_readback_ms=t['control_command_latency'],reconciliation_ms=d.get('baseline_reconciliation_ms',0.0))
        row['unpartitioned_outer_processing_ms']=d['full_loop_processing_ms']-row['reconciliation_ms']-row['guard_ms']-row['runtime_total_ms']
        row['runtime_other_ms']=row['runtime_total_ms']-row['preprocessing_ms']-row['inference_ms']-row['masking_ms']-row['command_send_ack_readback_ms']
        stages.append(row)
    stages=pd.DataFrame(stages);assert len(stages)==45
    assert (stages.unpartitioned_outer_processing_ms>=0).all()
    stages.to_csv(out/'a2_fault_per_window_timing.csv',index=False)
    a2=dict(run=fault.name,windows=45,measured_stage_summaries_ms={c:summary(stages[c]) for c in stages.columns if c!='window_index'},unpartitioned_outer_fraction_of_total=float(stages.unpartitioned_outer_processing_ms.sum()/stages.full_with_logging_ms.sum()),unavailable_individual_stage_durations=['telemetry wait','telemetry parse','detector evaluation','protected-session subprocess checks','command send alone','acknowledgement wait alone','consumer readback polling alone'],fixed_polling=dict(radio_window_poll_seconds=.05,readback_poll_seconds=.01,readback_timeout_seconds=.75,command_timeout_seconds=1.0),timing_boundary='Radio-window waiting is outside full_loop_processing_ms. Logging is separately measured. Send, acknowledgement and readback are combined. Outer residual includes telemetry file reload/parse, detector, two protected-session docker exec pgrep calls and other bookkeeping; these cannot be separated retrospectively.',dominance='The measured outer residual dominates; individual causes inside it are uninstrumented, so no measured parse/subprocess split is claimed.',historical_1_721_ms_boundary='Historical DecisionRuntime timing; excludes the outer telemetry/session/guard work and logging, hence not comparable with corrected complete local processing.')
    write_json(out/'a2_timing_decomposition.json',a2)

    partial=args.m1_root/'results/raw/iq-broker-live-cg-20261027m1-smoke-partial-guarded_policy-s9302'
    path=partial/'live-xapp-decisions.jsonl';register(path)
    rows=jsonlines(path);by_window={r['window_index']:r for r in rows}
    register(partial/'run_metadata.json');meta=json.loads((partial/'run_metadata.json').read_text())
    events=[]
    for d in rows:
        if d['safety_guard']['reason']!='post_action_harm_boundary_crossed':continue
        previous=by_window[d['window_index']-1];g=d['safety_guard']
        latency_threshold=max(g['baseline_latency_ms']*3,g['baseline_latency_ms']+1)
        throughput_threshold=g['baseline_throughput_mbps']*.5
        before=previous['decision']['raw_state'];after=d['decision']['raw_state']
        events.append(dict(action_window=previous['window_index'],rollback_window=d['window_index'],action=previous['decision']['execution']['requested_action'],authorized=previous['safety_guard']['allow_policy'],throughput_before_action_mbps=before['dl_throughput_mbps'],delay_before_action_ms=before['rlc_sdu_delay_ms'],throughput_at_rollback_mbps=after['dl_throughput_mbps'],delay_at_rollback_ms=after['rlc_sdu_delay_ms'],throughput_threshold_mbps=throughput_threshold,delay_threshold_ms=latency_threshold,boundary_already_crossed_before_action=bool(before['dl_throughput_mbps']<throughput_threshold or before['rlc_sdu_delay_ms']>latency_threshold),interference_ongoing=bool(meta['pre_jam_seconds']<=d['window_index']<meta['pre_jam_seconds']+meta['jammer_on_seconds']),action_readback_after_input_window_end=bool(previous['decision']['execution']['readback']['timestamp_ns']>previous['window_end_ns']),attribution='cannot distinguish action-caused harm from ongoing interference; the threshold was already violated before authorization'))
    assert len(events)==3
    pd.DataFrame(events).to_csv(out/'a3_rollback_event_windows.csv',index=False)
    write_json(out/'a3_rollback_attribution.json',dict(run=partial.name,events=events,conclusion='All three are persistence of a pre-existing threshold breach during ongoing interference. Temporal succession is not causal action harm, and a counterfactual ongoing-interference/no-action sequence is unavailable for these windows.'))
    register(ROOT/'experiments/live/run_xapp_loop.py');register(ROOT/'src/xapp/adapters/ocudu_websocket.py');register(ROOT/'src/controller/safety_guard.py')
    write_json(out/'audit_followup_inputs_sha256.json',inputs)
    lines=['# Existing-record follow-up audit - 28 October 2026','','No new cloud/radio runs, replayed timing benchmarks, altered frozen outputs, or modified raw measurements. New tables below are retrospective derived audit outputs.','','## A1. Shared zero-recovery pattern','', '| Arm | Live xApp | Runs | Positive gNB recovery windows / 1,800 | Positive native RLC SDU recovery windows / 1,800 | Scheduler UE entry windows / 1,800 |','|---|---|---|---|---|---|']
    lines += [f"| {r['method']} | {r['live_xapp_flags']} | 90 | {r['positive_throughput_windows']} | {r['positive_native_rlc_sdu_windows']} | {r['recovery_windows_with_scheduler_ue_entry']} |" for r in arms]
    lines += ['', 'Sources: `results/aggregate/live_validation_v10_records.parquet`: `method`, `phase`, `window_index`, `dl_throughput_mbps`, `raw_ocudu_rlc_json.rlc_metrics.tx.num_sdus`, `raw_ocudu_scheduler_json.cells[].ue_list`; manifest files `datasets/metadata/manifests/live_validation_v10/{B0,B2,B5,P}_live_validation_runs.csv`: `ENABLE_LIVE_XAPP`, `LIVE_XAPP_SELECTION_MODE`, `CONTROLLER_KIND`. Per-run findings are in `a1_per_run.csv`; scheduler UE presence is in `a1_recovery_scheduler_presence.csv`.', '', 'All three live-xApp arms share zero network-side service throughout recovery; B0 has some positive intervals. This is an association with the live-control execution path, not evidence uniquely implicating the learned policy. A scheduler UE entry is not proof of a functioning NR/PDU/application session. The historical receiver flag was exported from LTE state and cannot adjudicate NR detachment.', '', f"The raw evidence manifest (`results/release/live_validation_v10_raw_evidence_manifest.json`, `runs[].files[].path/sha256`) indexes {raw['raw_evidence_files']} files, but no original per-run `run_metadata.json` or original decision/traffic/container log is locally available for any of the 360 rows. The existing `revision_2026-10-23/v10-protected-traffic-status.tar.gz` has zero members. Manifest hashes are not log contents. Therefore UE/PDU session loss, persisted scheduler state, traffic-client cessation and harness ordering cannot be separately confirmed or excluded. No cloud fetch was performed.", '', 'The recovered exact frozen loop `revision_2026-10-22/reproducibility/controller_package_360_run/run_xapp_loop.py` matches the frozen digest in `LIVE_VALIDATION_FREEZE_MANIFEST_V10.yaml`. It runs direct decisions through the full duration, without a phase-aware recovery restoration or an explicit reset after the loop. This is a code-supported potential persistent-control pathway, not proof of a particular consumed scheduler state or failure cause. The later `tools/run_live_pilot_scenario.sh` describes traffic termination after the sample target, but lacks a per-run historical attestation and is not used to establish what happened in the expanded cohort.', '', '**Interpretation:** retain the reported counts and paired tests as outcomes of these evaluated controller-plus-harness configurations. Reframe the negative result as no demonstrated benefit and shared live-control-path non-recovery; do not attribute the recovery contrast to intrinsic learned-policy quality, verified NR detachment, or a proven harness defect. A common harness/controller integration cause remains unresolved.', '', '## A2. Complete local processing timing','', '| Measured interval | Median ms | p99 ms | Maximum ms |','|---|---|---|---|']
    for key in ['full_with_logging_ms','unpartitioned_outer_processing_ms','command_send_ack_readback_ms','inference_ms','guard_ms','logging_ms','reconciliation_ms']:
        s=a2['measured_stage_summaries_ms'][key];lines.append(f"| {key} | {s['median']:.6f} | {s['p99']:.6f} | {s['maximum']:.6f} |")
    lines += ['', f"The unpartitioned outer processing accounts for {100*a2['unpartitioned_outer_fraction_of_total']:.2f}% of summed complete local processing time. This is an additive per-window residual, not subtraction of marginal medians. Exact per-window components and residuals are in `a2_fault_per_window_timing.csv`.", '', 'Sources: corrected fault run `results/raw/iq-broker-live-cg-20261027m1-fault-partial-guarded_policy-s9401/live-xapp-decisions.jsonl`: `full_loop_processing_ms`, `guard_computation_ms`, `baseline_reconciliation_ms`, `decision.latency_ms.{preprocessing_latency,controller_inference_latency,action_masking_latency,control_command_latency,total_end_to_end_loop_latency}`; `loop-stage-timings.jsonl`: `logging_ms`, `full_loop_with_logging_ms`. These files are the unchanged members of `revision_2026-10-23/corrected_guard_measurement_amendment1_data_20261027.tar.gz`, read from an existing local extraction.', '', 'Source boundaries: `experiments/live/run_xapp_loop.py` starts `processing_start` AFTER the radio-window polling loop (50-ms sleep). `_telemetry_window` reloads/parses accumulated telemetry files. `_protected_session_active` launches synchronous `sudo docker exec ... pgrep -x iperf3`, called for action context and again for guard observation in each guarded decision. These operations and detector evaluation are in the residual; none has an individual historical timer. The measured `control_command_latency` includes send, acknowledgement and scheduler readback; those stages cannot be separated. `src/xapp/adapters/ocudu_websocket.py` specifies a 10-ms readback poll, 0.75-s readback deadline and 1-s command timeout; these are configured limits/poll intervals, not measured stage costs or fixed per-window waits. The planned dropped acknowledgement is discarded after actual application/response; a full 1-s timeout is not assumed.', '', 'The historical 1.721-ms median describes the inner runtime interval, not the new complete outer-loop boundary. Telemetry waiting is excluded in the latter as well. No retrospective claim assigns the residual specifically to parse time, inference, acknowledgement wait or polling. Exact stage attribution is unavailable.', '', '## A3. Three rollback events','', '| Authorized window -> rollback window | Action | Pre-action throughput Mbit/s | Pre-action delay ms | Rollback delay ms | Boundary already breached | Interference ongoing |','|---|---|---|---|---|---|']
    lines += [f"| {e['action_window']} -> {e['rollback_window']} | {e['action']} | {e['throughput_before_action_mbps']:.6f} | {e['delay_before_action_ms']:.6f} | {e['delay_at_rollback_ms']:.6f} | {e['boundary_already_crossed_before_action']} | {e['interference_ongoing']} |" for e in events]
    lines += ['', f"Guard thresholds were throughput < {events[0]['throughput_threshold_mbps']:.6f} Mbit/s or delay > {events[0]['delay_threshold_ms']:.6f} ms. Every authorization input already crossed those same thresholds. Input windows close before the ensuing action readback; a pre-action measurement is not an action effect. All rollback windows occurred within exposure windows 15--34 (zero-based).", '', 'Sources: corrected partial-band preflight `results/raw/iq-broker-live-cg-20261027m1-smoke-partial-guarded_policy-s9302/live-xapp-decisions.jsonl`, windows 17/18, 24/25, 31/32: `safety_guard.{allow_policy,reason,baseline_throughput_mbps,baseline_latency_ms}`, `decision.raw_state.{dl_throughput_mbps,rlc_sdu_delay_ms}`, `window_end_ns`, `decision.execution.{requested_action,readback.timestamp_ns}`; `run_metadata.json`: `pre_jam_seconds=15`, `jammer_on_seconds=20`; `src/controller/safety_guard.py`: previous-action nontrivial check and rollback thresholds. Exact event values are retained in `a3_rollback_event_windows.csv`.', '', '**Attribution for each event:** cannot distinguish action-caused harm from continuing interference. The observed threshold breach pre-exists each action and remains during exposure; there is no matched simultaneous counterfactual for the affected windows. Report these as harm-boundary activations and baseline requests, not three proved harmful actions or proved prevention of additional harm.', '', '## Unresolved records and publication boundaries','', '- A1: original 360-run traffic/decision/container logs are indexed but not locally readable; a specific common cause cannot be established from the available records.', '- A2: separate telemetry parse/session-check/send/ack/readback stage timers were not recorded; only the combined intervals and residual can be reported.', '- A3: three temporal action/rollback pairs do not identify a causal harm mechanism.', '- No new DOI, GitHub tag, deployment result, physical safety outcome or experimental result is inferred.','']
    (out/'audit_followup_2026-10-28.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(dict(arms=arms,timing=a2['measured_stage_summaries_ms'],residual_fraction=a2['unpartitioned_outer_fraction_of_total'],events=events),indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--expanded-records',type=Path,default=ROOT/'results/aggregate/live_validation_v10_records.parquet')
    parser.add_argument('--manifests',type=Path,default=ROOT/'datasets/metadata/manifests/live_validation_v10')
    parser.add_argument('--raw-manifest',type=Path,default=ROOT/'results/release/live_validation_v10_raw_evidence_manifest.json')
    parser.add_argument('--frozen-loop',type=Path,default=ROOT/'revision_2026-10-22/reproducibility/controller_package_360_run/run_xapp_loop.py')
    parser.add_argument('--m1-root',type=Path,default=ROOT/'data/local_measurement_amendment1')
    parser.add_argument('--output',type=Path,default=ROOT/'analysis_outputs')
    run(parser.parse_args())
