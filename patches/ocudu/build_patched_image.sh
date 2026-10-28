#!/usr/bin/env bash
set -euo pipefail

docker build \
  -f /tmp/Dockerfile.ocudu-patched \
  -t ocudu-gnb-zmq:git-fa4c9fa445a3-radiofix-txrebase \
  /home/deepanramkumar_p/oran-cloud-project/external/ocudu
