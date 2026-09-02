#!/usr/bin/env bash
set -euo pipefail

# Cloud entry point.  The project is expected at $HOME/O-RAN unless the
# caller supplies ORAN_PROJECT_ROOT.  The shared Ubuntu bootstrap performs
# the pinned dependency installation and records an environment snapshot.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export ORAN_PROJECT_ROOT="${ORAN_PROJECT_ROOT:-$HOME/O-RAN}"

if [[ ! -f "$ORAN_PROJECT_ROOT/requirements.in" ]]; then
  echo "Project root not found or incomplete: $ORAN_PROJECT_ROOT" >&2
  exit 2
fi

exec bash "$SCRIPT_DIR/bootstrap_ubuntu.sh"
