#!/usr/bin/env bash
set -euo pipefail

# The original local-WSL path is retained as the default.  Cloud Ubuntu
# callers can override it with ORAN_PROJECT_ROOT (for example,
# ORAN_PROJECT_ROOT="$HOME/O-RAN").
PROJECT_ROOT="${ORAN_PROJECT_ROOT:-/mnt/d/VIT Vellore Research one/O-RAN}"
SNAPSHOT_DIR="$PROJECT_ROOT/results/raw"
mkdir -p "$SNAPSHOT_DIR" "$PROJECT_ROOT/.cache/apt" "$PROJECT_ROOT/.tmp"

if [[ "$(. /etc/os-release && echo "$ID-$VERSION_ID")" != "ubuntu-24.04" ]]; then
  echo "This script requires Ubuntu 24.04; refusing to provision a different host." >&2
  exit 2
fi

export TMPDIR="$PROJECT_ROOT/.tmp"
export PIP_CACHE_DIR="$PROJECT_ROOT/.cache/pip"
export HF_HOME="$PROJECT_ROOT/.cache/huggingface"
export TORCH_HOME="$PROJECT_ROOT/.cache/torch"
export MPLCONFIGDIR="$PROJECT_ROOT/.cache/matplotlib"

sudo apt-get update
sudo apt-get install -y \
  ca-certificates curl gnupg git build-essential cmake ninja-build pkg-config \
  libsctp-dev lksctp-tools libzmq3-dev libfftw3-dev libmbedtls-dev \
  libboost-program-options-dev libconfig++-dev software-properties-common \
  python3-pip jq

# Ubuntu 24.04 defaults to Python 3.12, while the frozen research environment
# requires Python 3.11. Add the Noble-compatible deadsnakes PPA only when the
# requested interpreter is absent from the configured repositories.
if ! apt-cache show python3.11 >/dev/null 2>&1; then
  sudo add-apt-repository -y ppa:deadsnakes/ppa
  sudo apt-get update
fi
sudo apt-get install -y python3.11 python3.11-venv

sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc
. /etc/os-release
ARCH="$(dpkg --print-architecture)"
sudo tee /etc/apt/sources.list.d/docker.sources >/dev/null <<EOF
Types: deb
URIs: https://download.docker.com/linux/ubuntu
Suites: ${UBUNTU_CODENAME:-$VERSION_CODENAME}
Components: stable
Architectures: $ARCH
Signed-By: /etc/apt/keyrings/docker.asc
EOF
sudo apt-get update
DOCKER_VERSION="$(apt-cache madison docker-ce | awk 'NR==1 {print $3}')"
if [[ -z "$DOCKER_VERSION" ]]; then
  echo "No Docker CE version was found in the official repository." >&2
  exit 3
fi
sudo apt-get install -y \
  "docker-ce=$DOCKER_VERSION" "docker-ce-cli=$DOCKER_VERSION" \
  containerd.io docker-buildx-plugin docker-compose-plugin

python3.11 -m venv "$PROJECT_ROOT/envs/oran-action-xai"
"$PROJECT_ROOT/envs/oran-action-xai/bin/python" -m pip install --upgrade pip
"$PROJECT_ROOT/envs/oran-action-xai/bin/python" -m pip install -r "$PROJECT_ROOT/requirements.in"

{
  echo '{'
  printf '  "captured_at_utc": "%s",\n' "$(date -u +%FT%TZ)"
  printf '  "os": "%s",\n' "$(. /etc/os-release && echo "$PRETTY_NAME")"
  printf '  "kernel": "%s",\n' "$(uname -r)"
  printf '  "docker_ce_version": "%s",\n' "$DOCKER_VERSION"
  printf '  "python": "%s",\n' "$(python3.11 --version 2>&1)"
  printf '  "python_deb_version": "%s",\n' "$(dpkg-query -W -f='${Version}' python3.11)"
  printf '  "gcc": "%s",\n' "$(gcc --version | head -n1)"
  printf '  "cmake": "%s"\n' "$(cmake --version | head -n1)"
  echo '}'
} > "$SNAPSHOT_DIR/bootstrap_environment_snapshot.json"

echo "Ubuntu bootstrap complete. External repositories are not cloned until commits are selected and frozen."
