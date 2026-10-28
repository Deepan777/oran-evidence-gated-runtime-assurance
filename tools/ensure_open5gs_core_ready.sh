#!/usr/bin/env bash
# Idempotent Open5GS preflight for VM boot races.
set -euo pipefail

CORE_CONTAINER=${CORE_CONTAINER:-oran-open5gs-core}
MONGO_CONTAINER=${MONGO_CONTAINER:-oran-open5gs-mongodb}
MAX_ATTEMPTS=${MAX_ATTEMPTS:-90}

sudo docker inspect "$CORE_CONTAINER" >/dev/null
sudo docker inspect "$MONGO_CONTAINER" >/dev/null

mongo_ready=0
for _ in $(seq 1 "$MAX_ATTEMPTS"); do
  if sudo docker exec "$MONGO_CONTAINER" mongosh --quiet --eval \
      'quit(db.adminCommand({ping:1}).ok === 1 ? 0 : 1)' >/dev/null 2>&1; then
    mongo_ready=1
    break
  fi
  sleep 1
done
[ "$mongo_ready" = 1 ] || { echo "Open5GS MongoDB readiness check timed out" >&2; exit 1; }

ensure_nf() {
  name="$1"
  binary="$2"
  address="$3"
  port="$4"
  if ! sudo docker exec "$CORE_CONTAINER" pgrep -x "$binary" >/dev/null 2>&1; then
    sudo docker exec -d "$CORE_CONTAINER" sh -lc \
      "exec /usr/local/bin/$binary >>/evidence/open5gs-runtime/${name}-preflight.stdout.log 2>&1"
  fi
  for _ in $(seq 1 "$MAX_ATTEMPTS"); do
    if sudo docker exec "$CORE_CONTAINER" pgrep -x "$binary" >/dev/null 2>&1 && \
       sudo docker exec "$CORE_CONTAINER" timeout 1 bash -lc \
         "</dev/tcp/$address/$port" >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done
  echo "Open5GS service $name failed readiness at $address:$port" >&2
  sudo docker exec "$CORE_CONTAINER" sh -lc \
    "tail -n 200 /evidence/open5gs-runtime/${name}-preflight.stdout.log" >&2 || true
  return 1
}

ensure_nf udr open5gs-udrd 127.0.0.20 7777
ensure_nf pcf open5gs-pcfd 127.0.0.13 7777

# Authentication traverses AMF -> AUSF -> UDM -> UDR through SCP/NRF. Verify
# every listener in that path rather than accepting process existence alone.
for endpoint in \
  127.0.0.5:7777 \
  127.0.0.11:7777 \
  127.0.0.12:7777 \
  127.0.0.20:7777 \
  127.0.0.200:7777 \
  127.0.0.10:7777; do
  address=${endpoint%:*}
  port=${endpoint##*:}
  sudo docker exec "$CORE_CONTAINER" timeout 1 bash -lc \
    "</dev/tcp/$address/$port" >/dev/null 2>&1 \
    || { echo "Open5GS dependency listener unavailable: $endpoint" >&2; exit 1; }
done

echo "OPEN5GS_CORE_PREFLIGHT=PASS"
