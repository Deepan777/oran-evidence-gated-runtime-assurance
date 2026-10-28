#!/usr/bin/env python3
"""Start the auditable OCUDU downlink IQ jammer proxy."""
from __future__ import annotations

import argparse
from pathlib import Path

from src.jammer.waveforms import JammerConfig, JammerType
from src.jammer.zmq_proxy import proxy_loop


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--listen", default="tcp://127.0.0.1:2002", help="srsUE-facing REP address")
    parser.add_argument("--ocudu", default="tcp://127.0.0.1:2000", help="OCUDU TX REP address")
    parser.add_argument("--truth-log", type=Path, required=True)
    parser.add_argument("--jammer", choices=[item.value for item in JammerType], default="none")
    parser.add_argument("--jsr-db", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=1001)
    args = parser.parse_args()
    return proxy_loop(args.listen, args.ocudu, JammerConfig(JammerType(args.jammer), args.jsr_db, args.seed), args.truth_log)


if __name__ == "__main__": raise SystemExit(main())
