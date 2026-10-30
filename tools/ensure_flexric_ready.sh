#!/usr/bin/env bash
# Ensure the pinned near-RT RIC is listening before a disposable radio run.
set -euo pipefail

PROJECT=/home/OPERATOR_USER/oran-cloud-project
CONTAINER=oran-nearrt-ric
IMAGE=oai-flexric:git-ef6d722f2219
CONFIG="$PROJECT/external/flexric/flexric.conf"

sudo docker image inspect "$IMAGE" >/dev/null
test -f "$CONFIG"

if ! sudo docker inspect "$CONTAINER" >/dev/null 2>&1; then
  sudo docker run -d \
    --name "$CONTAINER" \
    --restart unless-stopped \
    --network host \
    -v "$CONFIG:/usr/local/etc/flexric/flexric.conf:ro" \
    "$IMAGE" stdbuf -o0 nearRT-RIC >/dev/null
elif [ "$(sudo docker inspect "$CONTAINER" --format '{{.State.Running}}')" != true ]; then
  sudo docker start "$CONTAINER" >/dev/null
fi

for _ in $(seq 1 100); do
  if sudo ss -lna 2>/dev/null | grep -q '127.0.0.1:36421'; then
    echo "FLEXRIC_PREFLIGHT=PASS"
    exit 0
  fi
  sleep 0.1
done

sudo docker logs --tail 200 "$CONTAINER" >&2 || true
echo "pinned near-RT RIC did not expose the E2 SCTP listener" >&2
exit 1
