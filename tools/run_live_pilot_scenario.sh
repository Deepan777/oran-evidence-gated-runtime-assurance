#!/usr/bin/env bash
# Run one immutable pilot manifest row against the native OCUDU/srsUE radio.
set -euo pipefail

PROJECT=/home/OPERATOR_USER/oran-cloud-project
RUN_ID=${RUN_ID:?RUN_ID is required}
SCENARIO_ID=${SCENARIO_ID:?SCENARIO_ID is required}
JAMMER=${JAMMER:?JAMMER is required}
JSR_DB=${JSR_DB:-0}
SEED=${SEED:?SEED is required}
TRAFFIC_LOAD=${TRAFFIC_LOAD:?TRAFFIC_LOAD is required}
SERVICE_MIX=${SERVICE_MIX:?SERVICE_MIX is required}
ATTENUATION_DB=${ATTENUATION_DB:?ATTENUATION_DB is required}
RUN_SECONDS=${RUN_SECONDS:?RUN_SECONDS is required}
PRE_JAM_SECONDS=${PRE_JAM_SECONDS:?PRE_JAM_SECONDS is required}
JAMMER_ON_SECONDS=${JAMMER_ON_SECONDS:?JAMMER_ON_SECONDS is required}
RECOVERY_SECONDS=${RECOVERY_SECONDS:?RECOVERY_SECONDS is required}
PROTECTED_MBPS=${PROTECTED_MBPS:?PROTECTED_MBPS is required}
BACKGROUND_MBPS=${BACKGROUND_MBPS:-0}
EVIDENCE_CLASS=${EVIDENCE_CLASS:-development_pilot}
GNB_IMAGE=${GNB_IMAGE:-ocudu-gnb-zmq:research-actions-fa4c9fa445a3}
UE_IMAGE=${UE_IMAGE:?UE_IMAGE is required}
GNB_CONFIG=${GNB_CONFIG:-ocudu_gnb_zmq_e2.yml}
SAMPLE_RATE_HZ=${SAMPLE_RATE_HZ:-23040000}
# The virtual radio is deliberately paced but can run slower than wall time
# under a high-rate metrics workload. This cap is only a safety limit; the
# authoritative stop condition is the forwarded-radio sample count below.
MAX_WALL_SECONDS=${MAX_WALL_SECONDS:-$((RUN_SECONDS * 20 + 120))}
# A pilot must always preserve its raw evidence, even when an optional client
# exits non-zero.  Confirmatory orchestration enables this explicitly after
# checking the generated record summary.
REQUIRE_COMPLETE_TELEMETRY=${REQUIRE_COMPLETE_TELEMETRY:-0}
REQUIRE_EVIDENCE_COMPLETE=${REQUIRE_EVIDENCE_COMPLETE:-0}
REQUIRE_PROTECTED_TRAFFIC=${REQUIRE_PROTECTED_TRAFFIC:-0}
ACCEPTANCE_PROFILE=${ACCEPTANCE_PROFILE:-strict_complete_telemetry_v1}
ENABLE_LIVE_XAPP=${ENABLE_LIVE_XAPP:-0}
LIVE_XAPP_XAI=${LIVE_XAPP_XAI:-0}
DETECTOR_MODEL=${DETECTOR_MODEL:-$PROJECT/results/training/detector/detector_offline.joblib}
CONTROLLER_MODEL=${CONTROLLER_MODEL:-$PROJECT/results/training/controller/md3qn_per_offline.pt}
CONTROLLER_KIND=${CONTROLLER_KIND:-md3qn_per}
STATE_SCHEMA=${STATE_SCHEMA:-$PROJECT/results/training/controller/state_schema.json}
LIVE_XAPP_SELECTION_MODE=${LIVE_XAPP_SELECTION_MODE:-policy}
LIVE_XAPP_FORCED_ACTION_ID=${LIVE_XAPP_FORCED_ACTION_ID:-}
OOD_GUARD_PROFILE=${OOD_GUARD_PROFILE:-training_quantile_fail_closed}
UE_TIMEOUT_MS=${UE_TIMEOUT_MS:-1000}
CAPTURE_FULL_CONTAINER_LOGS=${CAPTURE_FULL_CONTAINER_LOGS:-0}
ALLOW_UPLINK_SILENCE=${ALLOW_UPLINK_SILENCE:-0}

if [ $((PRE_JAM_SECONDS + JAMMER_ON_SECONDS + RECOVERY_SECONDS)) -ne "$RUN_SECONDS" ]; then
  echo "phase durations must sum to RUN_SECONDS" >&2; exit 2
fi
case "$SERVICE_MIX" in protected_only|protected_plus_background) ;; *) echo "invalid service mix" >&2; exit 2;; esac

E="$PROJECT/results/raw/iq-broker-live-$RUN_ID"
START_MARKER="$E/experiment.start"
sudo install -d -o deepanramkumar_p -g deepanramkumar_p "$E"
GNB_LOG_PID=""
UE_LOG_PID=""

# The radio containers and broker are disposable per-run infrastructure. Keep
# their raw logs before finalization, then remove only these named trial
# resources so they cannot continue consuming VM CPU/disk after a completed
# (or failed) pilot. Core Open5GS and near-RT RIC services are never touched.
cleanup_trial_runtime() {
  # Preserve the terminal container state before removing disposable radio
  # resources. This remains available even when set -e or an interrupt exits
  # before the normal finalisation block.
  sudo docker inspect oran-ocudu-gnb >"$E/gnb-container-inspect.json" 2>/dev/null || true
  sudo docker inspect oran-srsue >"$E/srsue-container-inspect.json" 2>/dev/null || true
  # Always retain a bounded causal tail on an interrupted/failed pilot.  A
  # full log follower is optional because it can itself perturb real-time
  # execution, but the final tail is cheap and essential for diagnosis.
  if [ ! -s "$E/gnb.log" ]; then
    sudo docker logs --tail 5000 oran-ocudu-gnb >"$E/gnb.log" 2>&1 || true
  fi
  if [ ! -s "$E/srsue-container.log" ]; then
    sudo docker logs --tail 5000 oran-srsue >"$E/srsue-container.log" 2>&1 || true
  fi
  for log_pid in "$GNB_LOG_PID" "$UE_LOG_PID"; do
    if [ -n "$log_pid" ]; then
      kill "$log_pid" >/dev/null 2>&1 || true
      wait "$log_pid" >/dev/null 2>&1 || true
    fi
  done
  sudo docker rm -f oran-ocudu-gnb oran-srsue >/dev/null 2>&1 || true
  if [ -f "$E/broker.pid" ]; then
    sudo kill "$(cat "$E/broker.pid")" >/dev/null 2>&1 || true
  fi
}
trap cleanup_trial_runtime EXIT

RUN_ID="$RUN_ID" JAMMER="$JAMMER" JSR_DB="$JSR_DB" SEED="$SEED" \
JAMMER_START_SECONDS="$PRE_JAM_SECONDS" \
JAMMER_STOP_SECONDS="$((PRE_JAM_SECONDS + JAMMER_ON_SECONDS))" \
ATTENUATION_DB="$ATTENUATION_DB" EXPERIMENT_START_FILE="$START_MARKER" \
GNB_IMAGE="$GNB_IMAGE" UE_IMAGE="$UE_IMAGE" GNB_CONFIG="$GNB_CONFIG" \
ALLOW_UPLINK_SILENCE="$ALLOW_UPLINK_SILENCE" UE_TIMEOUT_MS="$UE_TIMEOUT_MS" \
bash "$PROJECT/tools/run_live_registration.sh" | tee "$E/registration.log"

# Follow complete logs while containers exist. Unlike a bounded tail taken at
# finalisation, these files retain the first causal RRC/NGAP messages and are
# still saved when a container exits unexpectedly.
if [ "$CAPTURE_FULL_CONTAINER_LOGS" = 1 ]; then
  sudo docker logs -f --since 0s oran-ocudu-gnb >"$E/gnb.full.log" 2>&1 &
  GNB_LOG_PID=$!
  sudo docker logs -f --since 0s oran-srsue >"$E/srsue.full.log" 2>&1 &
  UE_LOG_PID=$!
fi

UE_IP=""
for _ in $(seq 1 600); do
  UE_IP=$(sudo docker exec oran-srsue sh -lc "ip -4 -o addr show tun_srsue 2>/dev/null | awk '{print \$4}' | cut -d/ -f1" || true)
  [ -n "$UE_IP" ] && break
  sleep 0.1
done
[ -n "$UE_IP" ] || { echo "UE PDU interface did not become ready" >&2; exit 1; }

command -v iperf3 >/dev/null 2>&1 || sudo env DEBIAN_FRONTEND=noninteractive apt-get install -y -qq iperf3
# The pinned calibration UE image already contains iperf3.  Retain a
# self-healing fallback for a clean image, but do not perform package setup in
# every disposable radio run.
sudo docker exec oran-srsue sh -lc 'command -v iperf3 >/dev/null 2>&1' \
  || { sudo docker exec -e DEBIAN_FRONTEND=noninteractive oran-srsue apt-get update -qq; sudo docker exec -e DEBIAN_FRONTEND=noninteractive oran-srsue apt-get install -y -qq iperf3; }
sudo pkill -x iperf3 >/dev/null 2>&1 || true
sudo iperf3 -s -B 10.45.0.1 -p 5201 -D --logfile "$E/iperf-protected-server.log"
if [ "$SERVICE_MIX" = protected_plus_background ]; then
  sudo iperf3 -s -B 10.45.0.1 -p 5202 -D --logfile "$E/iperf-background-server.log"
fi

cd "$PROJECT"
PYTHONPATH=. envs/oran-action-xai/bin/python tools/capture_ocudu_metrics.py \
  --output "$E/ocudu-metrics.jsonl" --duration-seconds "$MAX_WALL_SECONDS" \
  >"$E/metrics-capture.log" 2>&1 &
METRICS_PID=$!
sleep 1

MEASUREMENT_START_NS=$(date +%s%N)
cat >"$E/run_metadata.json" <<EOF
{"run_id":"$RUN_ID","scenario_id":"$SCENARIO_ID","seed":$SEED,"evidence_class":"$EVIDENCE_CLASS","acceptance_profile":"$ACCEPTANCE_PROFILE","ood_guard_profile":"$OOD_GUARD_PROFILE","measurement_start_ns":$MEASUREMENT_START_NS,"sample_rate_hz":$SAMPLE_RATE_HZ,"run_length_seconds":$RUN_SECONDS,"pre_jam_seconds":$PRE_JAM_SECONDS,"jammer_on_seconds":$JAMMER_ON_SECONDS,"recovery_seconds":$RECOVERY_SECONDS,"jammer_type":"$JAMMER","requested_jsr_db":$JSR_DB,"traffic_load":"$TRAFFIC_LOAD","service_mix":"$SERVICE_MIX","attenuation_db":$ATTENUATION_DB,"protected_mbps":$PROTECTED_MBPS,"background_mbps":$BACKGROUND_MBPS,"ue_ip":"$UE_IP","gnb_image":"$GNB_IMAGE","ue_image":"$UE_IMAGE","gnb_config":"$GNB_CONFIG","max_wall_seconds":$MAX_WALL_SECONDS,"ue_timeout_ms":$UE_TIMEOUT_MS,"allow_uplink_silence":$ALLOW_UPLINK_SILENCE,"capture_full_container_logs":$CAPTURE_FULL_CONTAINER_LOGS,"live_xapp_enabled":$ENABLE_LIVE_XAPP,"live_xapp_controller_kind":"$CONTROLLER_KIND","live_xapp_selection_mode":"$LIVE_XAPP_SELECTION_MODE","live_xapp_forced_action_id":"$LIVE_XAPP_FORCED_ACTION_ID","live_xapp_xai_requested":$LIVE_XAPP_XAI,"xai_execution_mode":"$([ "$LIVE_XAPP_XAI" = 1 ] && echo post_run_nonblocking_worker || echo disabled)"}
EOF
touch "$START_MARKER"

if [ "$ENABLE_LIVE_XAPP" = 1 ]; then
  [ -f "$DETECTOR_MODEL" ] && [ -f "$CONTROLLER_MODEL" ] && [ -f "$STATE_SCHEMA" ] \
    || { echo "frozen detector/controller/state schema is missing" >&2; exit 2; }
  XAPP_ARGS=(
    --ocudu-metrics "$E/ocudu-metrics.jsonl"
    --srsue-metrics "$E/srsue_metrics.csv"
    --iq-sensor "$E/iq-spectral-estimator.jsonl"
    --detector "$DETECTOR_MODEL"
    --controller "$CONTROLLER_MODEL"
    --controller-kind "$CONTROLLER_KIND"
    --state-schema "$STATE_SCHEMA"
    --output "$E/live-xapp-decisions.jsonl"
    --duration-seconds "$RUN_SECONDS"
    --radio-window-driven
    --wall-timeout-seconds "$MAX_WALL_SECONDS"
    --ood-guard-profile "$OOD_GUARD_PROFILE"
  )
  # Explanations are deliberately excluded from the one-second control path.
  # When requested, an isolated worker consumes the immutable acknowledged
  # decisions after the real-radio measurement has ended.
  if [ "$LIVE_XAPP_SELECTION_MODE" = "forced_exploration" ] || [ "$LIVE_XAPP_SELECTION_MODE" = "fixed_action" ]; then
    [ -n "$LIVE_XAPP_FORCED_ACTION_ID" ] || { echo "forced exploration needs LIVE_XAPP_FORCED_ACTION_ID" >&2; exit 2; }
    XAPP_ARGS+=(--selection-mode "$LIVE_XAPP_SELECTION_MODE" --forced-action-id "$LIVE_XAPP_FORCED_ACTION_ID")
  elif [ "$LIVE_XAPP_SELECTION_MODE" = "bounded_action_probe" ]; then
    [ -n "$LIVE_XAPP_FORCED_ACTION_ID" ] || { echo "bounded action probe needs LIVE_XAPP_FORCED_ACTION_ID" >&2; exit 2; }
    XAPP_ARGS+=(
      --selection-mode bounded_action_probe
      --forced-action-id "$LIVE_XAPP_FORCED_ACTION_ID"
      --probe-start-window "$PRE_JAM_SECONDS"
      --probe-stop-window "$((PRE_JAM_SECONDS + JAMMER_ON_SECONDS))"
    )
  elif [ "$LIVE_XAPP_SELECTION_MODE" = "ppo_sampling" ]; then
    XAPP_ARGS+=(--selection-mode ppo_sampling)
  elif [ "$LIVE_XAPP_SELECTION_MODE" = "safeguarded_policy" ]; then
    XAPP_ARGS+=(--selection-mode safeguarded_policy)
  elif [ "$LIVE_XAPP_SELECTION_MODE" != "policy" ]; then
    echo "invalid LIVE_XAPP_SELECTION_MODE" >&2; exit 2
  fi
  ORAN_SCHEDULER_READBACK_LOG="$E/gnb.full.log" PYTHONPATH=. envs/oran-action-xai/bin/python experiments/live/run_xapp_loop.py "${XAPP_ARGS[@]}" \
    >"$E/live-xapp.log" 2>&1 &
  XAPP_PID=$!
else
  XAPP_PID=""
fi

PROTECTED_TRAFFIC_START_NS=$(date +%s%N)
printf '%s\n' "$PROTECTED_TRAFFIC_START_NS" >"$E/protected-client-start-ns.txt"
sudo docker exec oran-srsue iperf3 -c 10.45.0.1 -B "$UE_IP" -p 5201 -u -b "${PROTECTED_MBPS}M" -l 256 -t "$MAX_WALL_SECONDS" -R -J >"$E/protected_udp.json" 2>&1 &
PROTECTED_PID=$!
if [ "$SERVICE_MIX" = protected_plus_background ]; then
  sudo docker exec oran-srsue iperf3 -c 10.45.0.1 -B "$UE_IP" -p 5202 -u -b "${BACKGROUND_MBPS}M" -l 1200 -t "$MAX_WALL_SECONDS" -R -J >"$E/background_udp.json" 2>&1 &
  BACKGROUND_PID=$!
fi
# Ping is diagnostic only. It must never hold evidence finalisation open when
# a strongly jammed radio drops replies, so give it a finite wall-clock cap.
sudo docker exec oran-srsue timeout "$((RUN_SECONDS + 5))" ping -I tun_srsue -i 0.2 -W 2 -c "$((RUN_SECONDS * 5))" 10.45.0.1 >"$E/protected_ping.txt" 2>&1 &
PING_PID=$!

set +e
PYTHONPATH=. envs/oran-action-xai/bin/python tools/wait_for_radio_samples.py \
  --evidence "$E" --target-samples "$((RUN_SECONDS * SAMPLE_RATE_HZ))" \
  --timeout-seconds "$MAX_WALL_SECONDS" >"$E/radio-clock-status.json"
RADIO_CLOCK_STATUS=$?
# The traffic clients are intentionally ended only after the exact radio-time
# target is reached (or its explicit safety timeout). Their partial JSON is
# retained as raw evidence and the termination status is recorded below.
sudo docker exec oran-srsue pkill -x iperf3 >/dev/null 2>&1 || true
sudo pkill -x iperf3 >/dev/null 2>&1 || true
kill "$METRICS_PID" >/dev/null 2>&1 || true
wait "$PROTECTED_PID"; PROTECTED_STATUS=$?
if [ "$SERVICE_MIX" = protected_plus_background ]; then
  wait "$BACKGROUND_PID"; BACKGROUND_STATUS=$?
else
  BACKGROUND_STATUS=0
fi
sudo docker exec oran-srsue pkill -x ping >/dev/null 2>&1 || true
wait "$PING_PID"; PING_STATUS=$?
wait "$METRICS_PID"; METRICS_STATUS=$?
if [ -n "$XAPP_PID" ]; then
  # The final radio window may appear just after the sample counter reaches
  # its target. Give the controller a bounded grace interval to append the
  # final immutable decision before declaring the row incomplete.
  for _ in $(seq 1 300); do
    kill -0 "$XAPP_PID" >/dev/null 2>&1 || break
    sleep 0.1
  done
  kill "$XAPP_PID" >/dev/null 2>&1 || true
  wait "$XAPP_PID"; XAPP_STATUS=$?
else
  XAPP_STATUS=0
fi
set -e
printf 'protected_iperf_exit=%s\nbackground_iperf_exit=%s\nping_exit=%s\nmetrics_capture_exit=%s\nlive_xapp_exit=%s\n' \
  "$PROTECTED_STATUS" "$BACKGROUND_STATUS" "$PING_STATUS" "$METRICS_STATUS" "$XAPP_STATUS" >"$E/client-exit-status.txt"
printf 'radio_clock_exit=%s\n' "$RADIO_CLOCK_STATUS" >>"$E/client-exit-status.txt"
# The live containers can produce verbose logs continuously. Retain a bounded
# diagnostic tail so collection cannot stall on unbounded Docker log reads.
sudo docker logs --tail 5000 oran-ocudu-gnb >"$E/gnb.log" 2>&1 || true
sudo docker logs --tail 5000 oran-srsue >"$E/srsue.log" 2>&1 || true
sudo docker stats --no-stream --format '{{json .}}' oran-ocudu-gnb oran-srsue >"$E/container-resource-snapshot.jsonl" || true
sudo pkill -x iperf3 >/dev/null 2>&1 || true

BUILD_ARGS=(--evidence "$E" --metadata "$E/run_metadata.json")
if [ "$REQUIRE_COMPLETE_TELEMETRY" = 1 ]; then BUILD_ARGS+=(--require-complete); fi
if [ "$REQUIRE_EVIDENCE_COMPLETE" = 1 ]; then BUILD_ARGS+=(--require-evidence-complete); fi
PYTHONPATH=. envs/oran-action-xai/bin/python experiments/collection/build_live_run_records.py "${BUILD_ARGS[@]}" \
  | tee "$E/record-build-summary.json"
PROTECTED_TRAFFIC_STATUS=0
if [ "$REQUIRE_PROTECTED_TRAFFIC" = 1 ]; then
  set +e
  PYTHONPATH=. envs/oran-action-xai/bin/python experiments/collection/iperf_evidence.py \
    --iperf "$E/protected_udp.json" \
    --sensor "$E/iq-spectral-estimator.jsonl" \
    --metadata "$E/run_metadata.json" \
    --traffic-start-ns-file "$E/protected-client-start-ns.txt" \
    --expected-rows "$RUN_SECONDS" \
    --output "$E/protected-traffic-status.json" \
    >"$E/protected-traffic-validator.log" 2>&1
  PROTECTED_TRAFFIC_STATUS=$?
  set -e
  printf 'protected_traffic_validator_exit=%s\n' "$PROTECTED_TRAFFIC_STATUS" >>"$E/client-exit-status.txt"
fi
XAI_WORKER_STATUS=0
if [ "$LIVE_XAPP_XAI" = 1 ]; then
  # Release the disposable radio workload before explanation generation. The
  # worker can read evidence but cannot affect the already completed actions,
  # acknowledgements, or radio records.
  cleanup_trial_runtime
  set +e
  PYTHONPATH=. envs/oran-action-xai/bin/python analysis/enrich_live_xai.py \
    --decisions "$E/live-xapp-decisions.jsonl" \
    --controller "$CONTROLLER_MODEL" \
    --state-schema "$STATE_SCHEMA" \
    --output "$E/live-xapp-xai.jsonl" \
    --expected-rows "$RUN_SECONDS" \
    >"$E/live-xapp-xai-worker.log" 2>&1
  XAI_WORKER_STATUS=$?
  set -e
  printf 'xai_worker_exit=%s\n' "$XAI_WORKER_STATUS" >>"$E/client-exit-status.txt"
fi
if [ "$RADIO_CLOCK_STATUS" -ne 0 ]; then
  echo "LIVE_PILOT_SCENARIO=INCOMPLETE_RADIO_DURATION run_id=$RUN_ID evidence=$E" >&2
  exit 1
fi
if [ "$XAI_WORKER_STATUS" -ne 0 ]; then
  echo "LIVE_PILOT_SCENARIO=XAI_INCOMPLETE run_id=$RUN_ID evidence=$E" >&2
  exit 1
fi
if [ "$PROTECTED_TRAFFIC_STATUS" -ne 0 ]; then
  echo "LIVE_PILOT_SCENARIO=PROTECTED_TRAFFIC_INCOMPLETE run_id=$RUN_ID evidence=$E" >&2
  exit 1
fi
echo "LIVE_PILOT_SCENARIO=COLLECTED run_id=$RUN_ID evidence=$E"
