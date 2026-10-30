# Existing-record follow-up audit - 28 October 2026

No new cloud/radio runs, replayed timing benchmarks, altered frozen outputs, or modified raw measurements. New tables below are retrospective derived audit outputs.

## A1. Shared zero-recovery pattern

| Arm | Live xApp | Runs | Positive gNB recovery windows / 1,800 | Positive native RLC SDU recovery windows / 1,800 | Scheduler UE entry windows / 1,800 |
|---|---|---|---|---|---|
| B0 | [0] | 90 | 720 | 720 | 1800 |
| B2 | [1] | 90 | 0 | 0 | 1800 |
| B5 | [1] | 90 | 0 | 0 | 1800 |
| P | [1] | 90 | 0 | 0 | 1800 |

Sources: `results/aggregate/live_validation_v10_records.parquet`: `method`, `phase`, `window_index`, `dl_throughput_mbps`, `raw_ocudu_rlc_json.rlc_metrics.tx.num_sdus`, `raw_ocudu_scheduler_json.cells[].ue_list`; manifest files `datasets/metadata/manifests/live_validation_v10/{B0,B2,B5,P}_live_validation_runs.csv`: `ENABLE_LIVE_XAPP`, `LIVE_XAPP_SELECTION_MODE`, `CONTROLLER_KIND`. Per-run findings are in `a1_per_run.csv`; scheduler UE presence is in `a1_recovery_scheduler_presence.csv`.

All three live-xApp arms share zero network-side service throughout recovery; B0 has some positive intervals. This is an association with the live-control execution path, not evidence uniquely implicating the learned policy. A scheduler UE entry is not proof of a functioning NR/PDU/application session. The historical receiver flag was exported from LTE state and cannot adjudicate NR detachment.

The raw evidence manifest (`results/release/live_validation_v10_raw_evidence_manifest.json`, `runs[].files[].path/sha256`) indexes 11888 files, but no original per-run `run_metadata.json` or original decision/traffic/container log is locally available for any of the 360 rows. The existing `revision_2026-10-23/v10-protected-traffic-status.tar.gz` has zero members. Manifest hashes are not log contents. Therefore UE/PDU session loss, persisted scheduler state, traffic-client cessation and harness ordering cannot be separately confirmed or excluded. No cloud fetch was performed.

The recovered exact frozen loop `revision_2026-10-22/reproducibility/controller_package_360_run/run_xapp_loop.py` matches the frozen digest in `LIVE_VALIDATION_FREEZE_MANIFEST_V10.yaml`. It runs direct decisions through the full duration, without a phase-aware recovery restoration or an explicit reset after the loop. This is a code-supported potential persistent-control pathway, not proof of a particular consumed scheduler state or failure cause. The later `tools/run_live_pilot_scenario.sh` describes traffic termination after the sample target, but lacks a per-run historical attestation and is not used to establish what happened in the expanded cohort.

**Interpretation:** retain the reported counts and paired tests as outcomes of these evaluated controller-plus-harness configurations. Reframe the negative result as no demonstrated benefit and shared live-control-path non-recovery; do not attribute the recovery contrast to intrinsic learned-policy quality, verified NR detachment, or a proven harness defect. A common harness/controller integration cause remains unresolved.

## A2. Complete local processing timing

| Measured interval | Median ms | p99 ms | Maximum ms |
|---|---|---|---|
| full_with_logging_ms | 435.760180 | 820.579124 | 831.838099 |
| unpartitioned_outer_processing_ms | 418.936102 | 802.574363 | 812.783128 |
| command_send_ack_readback_ms | 12.742911 | 33.469532 | 33.732802 |
| inference_ms | 0.922830 | 4.101658 | 4.685380 |
| guard_ms | 0.786000 | 3.397415 | 3.518930 |
| logging_ms | 0.373950 | 2.002024 | 2.484220 |
| reconciliation_ms | 0.000000 | 13.333973 | 14.042563 |

The unpartitioned outer processing accounts for 96.07% of summed complete local processing time. This is an additive per-window residual, not subtraction of marginal medians. Exact per-window components and residuals are in `a2_fault_per_window_timing.csv`.

Sources: corrected fault run `results/raw/iq-broker-live-cg-20261027m1-fault-partial-guarded_policy-s9401/live-xapp-decisions.jsonl`: `full_loop_processing_ms`, `guard_computation_ms`, `baseline_reconciliation_ms`, `decision.latency_ms.{preprocessing_latency,controller_inference_latency,action_masking_latency,control_command_latency,total_end_to_end_loop_latency}`; `loop-stage-timings.jsonl`: `logging_ms`, `full_loop_with_logging_ms`. These files are the unchanged members of `revision_2026-10-23/corrected_guard_measurement_amendment1_data_20261027.tar.gz`, read from an existing local extraction.

Source boundaries: `experiments/live/run_xapp_loop.py` starts `processing_start` AFTER the radio-window polling loop (50-ms sleep). `_telemetry_window` reloads/parses accumulated telemetry files. `_protected_session_active` launches synchronous `sudo docker exec ... pgrep -x iperf3`, called for action context and again for guard observation in each guarded decision. These operations and detector evaluation are in the residual; none has an individual historical timer. The measured `control_command_latency` includes send, acknowledgement and scheduler readback; those stages cannot be separated. `src/xapp/adapters/ocudu_websocket.py` specifies a 10-ms readback poll, 0.75-s readback deadline and 1-s command timeout; these are configured limits/poll intervals, not measured stage costs or fixed per-window waits. The planned dropped acknowledgement is discarded after actual application/response; a full 1-s timeout is not assumed.

The historical 1.721-ms median describes the inner runtime interval, not the new complete outer-loop boundary. Telemetry waiting is excluded in the latter as well. No retrospective claim assigns the residual specifically to parse time, inference, acknowledgement wait or polling. Exact stage attribution is unavailable.

## A3. Three rollback events

| Authorized window -> rollback window | Action | Pre-action throughput Mbit/s | Pre-action delay ms | Rollback delay ms | Boundary already breached | Interference ongoing |
|---|---|---|---|---|---|
| 17 -> 18 | MCS_BACKOFF | 0.000000 | 2409.772006 | 2771.543127 | True | True |
| 24 -> 25 | MCS_BACKOFF | 0.000000 | 5343.744124 | 4123.554536 | True | True |
| 31 -> 32 | MCS_BACKOFF | 0.000000 | 5357.144170 | 3679.814905 | True | True |

Guard thresholds were throughput < 2.375020 Mbit/s or delay > 5.305052 ms. Every authorization input already crossed those same thresholds. Input windows close before the ensuing action readback; a pre-action measurement is not an action effect. All rollback windows occurred within exposure windows 15--34 (zero-based).

Sources: corrected partial-band preflight `results/raw/iq-broker-live-cg-20261027m1-smoke-partial-guarded_policy-s9302/live-xapp-decisions.jsonl`, windows 17/18, 24/25, 31/32: `safety_guard.{allow_policy,reason,baseline_throughput_mbps,baseline_latency_ms}`, `decision.raw_state.{dl_throughput_mbps,rlc_sdu_delay_ms}`, `window_end_ns`, `decision.execution.{requested_action,readback.timestamp_ns}`; `run_metadata.json`: `pre_jam_seconds=15`, `jammer_on_seconds=20`; `src/controller/safety_guard.py`: previous-action nontrivial check and rollback thresholds. Exact event values are retained in `a3_rollback_event_windows.csv`.

**Attribution for each event:** cannot distinguish action-caused harm from continuing interference. The observed threshold breach pre-exists each action and remains during exposure; there is no matched simultaneous counterfactual for the affected windows. Report these as harm-boundary activations and baseline requests, not three proved harmful actions or proved prevention of additional harm.

## Unresolved records and publication boundaries

- A1: original 360-run traffic/decision/container logs are indexed but not locally readable; a specific common cause cannot be established from the available records.
- A2: separate telemetry parse/session-check/send/ack/readback stage timers were not recorded; only the combined intervals and residual can be reported.
- A3: three temporal action/rollback pairs do not identify a causal harm mechanism.
- No new DOI, GitHub tag, deployment result, physical safety outcome or experimental result is inferred.
