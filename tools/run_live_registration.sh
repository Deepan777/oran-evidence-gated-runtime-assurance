#!/usr/bin/env bash
set -euo pipefail

PROJECT=/home/OPERATOR_USER/oran-cloud-project
JAMMER="${JAMMER:-none}"
JSR_DB="${JSR_DB:-0}"
SEED="${SEED:-1001}"
JAMMER_START_SECONDS="${JAMMER_START_SECONDS:-0}"
JAMMER_STOP_SECONDS="${JAMMER_STOP_SECONDS:-}"
ATTENUATION_DB="${ATTENUATION_DB:-0}"
EXPERIMENT_START_FILE="${EXPERIMENT_START_FILE:-}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)-${JAMMER}-jsr${JSR_DB}-seed${SEED}}"
EVIDENCE="$PROJECT/results/raw/iq-broker-live-$RUN_ID"
BROKER_PID_FILE="$EVIDENCE/broker.pid"
GNB_IMAGE="${GNB_IMAGE:-ocudu-gnb-zmq:git-fa4c9fa445a3-radiofix-macfix}"
GNB_CONFIG="${GNB_CONFIG:-ocudu_gnb_zmq_e2.yml}"
# This image is derived once from the pinned telemetry image with the normal
# traffic client installed.  Keeping it locally avoids an apt transaction in
# every disposable calibration UE while preserving the identical radio binary.
UE_IMAGE="${UE_IMAGE:-srsue-zmq:git-6bcbd9e5bf86-telemetry-v1-iperf3}"

# A VM restart can expose MongoDB's TCP listener before it is ready for real
# operations. The Open5GS container entrypoint starts UDR/PCF only once, so
# both services can exit during that narrow window and later cause an AUSF 504
# plus immediate RRC release. Refuse radio registration until the complete
# core dependency chain is healthy, restarting only missing stateless NFs.
bash "$PROJECT/tools/ensure_open5gs_core_ready.sh"
bash "$PROJECT/tools/ensure_flexric_ready.sh"

# A manifest retry may intentionally reuse the locked run_id.  Never append to
# the previous attempt's broker/truth logs: wait_for_radio_samples treats any
# historical uplink_timeout as terminal.  Preserve the old raw evidence in a
# timestamped sibling archive, then create a clean directory for this attempt.
if [ -e "$EVIDENCE" ]; then
  ARCHIVE_ROOT="$PROJECT/results/raw/retry-archive"
  ARCHIVE_TAG=$(date -u +%Y%m%dT%H%M%SZ)
  mkdir -p "$ARCHIVE_ROOT"
  mv "$EVIDENCE" "$ARCHIVE_ROOT/$(basename "$EVIDENCE")-$ARCHIVE_TAG"
fi
mkdir -p "$EVIDENCE"

# Remove only the two disposable radio containers from the previous trial.
sudo docker rm -f oran-ocudu-gnb oran-srsue >/dev/null 2>&1 || true
# A unique evidence directory means its PID file cannot identify a broker from
# an earlier run.  Match only this project tool so stale brokers cannot retain
# the four ZMQ endpoints and silently receive part of the next radio stream.
sudo pkill -f '[r]un_zmq_radio_broker.py' >/dev/null 2>&1 || true
if [ -f "$BROKER_PID_FILE" ]; then
  kill "$(cat "$BROKER_PID_FILE")" >/dev/null 2>&1 || true
fi

cd "$PROJECT"
# Start the broker before the endpoints. It waits for both sockets and holds
# the virtual clock at startup until it receives a real downlink buffer.
UE_TIMEOUT_MS="${UE_TIMEOUT_MS:-1000}"
BROKER_ARGS=(
  --jammer "$JAMMER"
  --jsr-db "$JSR_DB"
  --seed "$SEED"
  --truth-log "$EVIDENCE/jammer_truth.jsonl"
  --status-log "$EVIDENCE/broker_status.jsonl"
  --sensor-log "$EVIDENCE/iq-spectral-estimator.jsonl"
  --silence-samples 23040
  --startup-discard-limit 10000
  --sample-rate-hz 23040000
  --ue-timeout-ms "$UE_TIMEOUT_MS"
  --jammer-start-seconds "$JAMMER_START_SECONDS"
  --attenuation-db "$ATTENUATION_DB"
)
if [ -n "$JAMMER_STOP_SECONDS" ]; then
  BROKER_ARGS+=(--jammer-stop-seconds "$JAMMER_STOP_SECONDS")
fi
if [ -n "$EXPERIMENT_START_FILE" ]; then
  rm -f "$EXPERIMENT_START_FILE"
  BROKER_ARGS+=(--experiment-start-file "$EXPERIMENT_START_FILE")
fi
# Locked evidence never substitutes a synthetic UE response.  The optional
# recovery mode remains available only for explicitly labelled diagnostics;
# confirmatory runs fail fast if the physical UE stops advancing the clock.
ALLOW_UPLINK_SILENCE="${ALLOW_UPLINK_SILENCE:-0}"
if [ "$ALLOW_UPLINK_SILENCE" = 1 ]; then
  BROKER_ARGS+=(--allow-uplink-silence)
fi

# The legacy ZMQ UE maintains an approximately 1 ms transmit deadline.  The
# broker is part of that clock, so ordinary best-effort scheduling can add
# enough host jitter to release RRC despite correct IQ sample processing.
# Round-robin real-time scheduling yields whenever the broker's paced frame
# completes; it changes no waveform, scenario factor, or sample stream.
BROKER_SCHEDULER="cpu0_round_robin_priority_20"
nohup taskset -c 0 chrt -r 20 env PYTHONPATH=. envs/oran-action-xai/bin/python3 \
  tools/run_zmq_radio_broker.py \
  "${BROKER_ARGS[@]}" \
  >"$EVIDENCE/broker.log" 2>&1 &
echo $! >"$BROKER_PID_FILE"
printf 'broker_scheduler=%s\n' "$BROKER_SCHEDULER" >"$EVIDENCE/broker-scheduling.txt"
for _ in $(seq 1 50); do
  if ps -o cls= -p "$(cat "$BROKER_PID_FILE")" 2>/dev/null | awk '$1 == "RR" { found=1 } END { exit !found }'; then
    break
  fi
  sleep 0.02
done
if ! ps -o cls= -p "$(cat "$BROKER_PID_FILE")" 2>/dev/null | awk '$1 == "RR" { found=1 } END { exit !found }'; then
  echo "broker did not enter required round-robin real-time scheduling class" >&2
  exit 1
fi

# The patched gNB and UE are started against an already-listening broker. The
# UE must start before the cell-activation gate so the two ZMQ radio directions
# can establish their normal request/reply clock without synthetic IQ.
# Run the UE PHY/gateway with the same bounded real-time class as the IQ
# broker; otherwise a 0.013 ms host scheduling slip can release RRC.
sudo docker run -d \
  --name oran-ocudu-gnb \
  --network host \
  --cpuset-cpus 1,2 \
  --cap-add IPC_LOCK \
  --cap-add SYS_NICE \
  --ulimit memlock=-1 \
  --ulimit rtprio=99 \
  -v "$PROJECT/config/baseline:/config:ro" \
  -v "$EVIDENCE:/evidence" \
  "$GNB_IMAGE" -c "/config/$GNB_CONFIG" \
  >"$EVIDENCE/gnb_container_id.txt"

for _ in $(seq 1 50); do
  if grep -q '"event": "downlink_ready"' "$EVIDENCE/broker_status.jsonl" 2>/dev/null && \
     grep -q '"event": "uplink_ready"' "$EVIDENCE/broker_status.jsonl" 2>/dev/null; then
    break
  fi
  sleep 0.1
done

# The 8-vCPU cloud VM reserves two cores for the UE.  A single pinned core
# can starve the srsUE gateway thread during long intermittent-JSR runs and
# produce a non-radio PDU-allocation exit despite a complete IQ clock.
sudo docker run -d \
  --name oran-srsue \
  --network bridge \
  --cpuset-cpus 3,4 \
  --add-host host.docker.internal:host-gateway \
  --publish 127.0.0.1:2003:2003 \
  --device /dev/net/tun \
  --cap-add NET_ADMIN \
  --cap-add SYS_NICE \
  --ulimit memlock=-1 \
  --ulimit rtprio=99 \
  --entrypoint chrt \
  -v "$PROJECT/config/baseline:/config:ro" \
  -v "$EVIDENCE:/evidence" \
  "$UE_IMAGE" -r 20 taskset -c 3,4 /usr/local/bin/srsue /config/srsue_nr_sa_zmq_proxy.conf \
  >"$EVIDENCE/srsue_container_id.txt"

# Forward only the Open5GS UE address pool.  These rules are idempotent and
# retain Docker's default isolation for every other subnet.
sudo iptables -t nat -C POSTROUTING -s 10.45.0.0/16 -o ens4 -j MASQUERADE 2>/dev/null \
  || sudo iptables -t nat -A POSTROUTING -s 10.45.0.0/16 -o ens4 -j MASQUERADE
sudo iptables -C FORWARD -i ogstun -o ens4 -j ACCEPT 2>/dev/null \
  || sudo iptables -I FORWARD 1 -i ogstun -o ens4 -j ACCEPT
sudo iptables -C FORWARD -i ens4 -o ogstun -m conntrack --ctstate ESTABLISHED,RELATED -j ACCEPT 2>/dev/null \
  || sudo iptables -I FORWARD 1 -i ens4 -o ogstun -m conntrack --ctstate ESTABLISHED,RELATED -j ACCEPT

GNB_LOG_PATH=$(sudo docker inspect oran-ocudu-gnb --format '{{.LogPath}}')
gnb_log_has() {
  sudo grep -aFqm1 "$1" "$GNB_LOG_PATH" 2>/dev/null
}
radio_signal_acquired() {
  grep -q '"event": "downlink_signal_acquired"' "$EVIDENCE/broker_status.jsonl" 2>/dev/null
}
for _ in $(seq 1 1200); do
  # Production logging can intentionally suppress the informational
  # "Cell was activated" text. A non-zero gNB IQ frame observed by the broker
  # is an equivalent runtime readiness signal; the caller separately requires
  # the UE PDU interface before starting the immutable experiment clock.
  if gnb_log_has 'Cell was activated' || radio_signal_acquired; then
    break
  fi
  sleep 0.1
done
if ! gnb_log_has 'Cell was activated' && ! radio_signal_acquired; then
  sudo tail -n 5000 "$GNB_LOG_PATH" >"$EVIDENCE/gnb_startup.log" 2>&1
  echo "gNB scheduler did not activate with both radio endpoints present" >&2
  exit 1
fi

echo "LIVE_REGISTRATION_STARTED jammer=$JAMMER jsr_db=$JSR_DB seed=$SEED evidence=$EVIDENCE"
