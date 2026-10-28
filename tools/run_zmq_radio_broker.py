#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from src.jammer.waveforms import JammerConfig, JammerType
from src.jammer.zmq_radio_broker import ZmqRadioBroker


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jammer", choices=[item.value for item in JammerType], default="none")
    parser.add_argument("--jsr-db", type=float, default=0.0); parser.add_argument("--seed", type=int, default=1001)
    parser.add_argument("--truth-log", type=Path, required=True); parser.add_argument("--status-log", type=Path, required=True)
    parser.add_argument("--sensor-log", type=Path, help="measured-IQ spectral estimator output; never jammer truth")
    parser.add_argument("--silence-samples", type=int, default=23040)
    parser.add_argument("--startup-discard-limit", type=int, default=10000)
    parser.add_argument("--sample-rate-hz", type=float, default=23.04e6)
    parser.add_argument("--ue-timeout-ms", type=int, default=100)
    parser.add_argument("--jammer-start-seconds", type=float, default=0.0)
    parser.add_argument("--jammer-stop-seconds", type=float)
    parser.add_argument("--attenuation-db", type=float, default=0.0)
    parser.add_argument("--experiment-start-file", type=Path)
    parser.add_argument(
        "--allow-uplink-silence",
        action="store_true",
        help="continue the radio clock with explicit zero-IQ uplink after RRC release",
    )
    args = parser.parse_args()
    # The UE runs in an isolated Docker network namespace so its PDU-session
    # TUN route cannot overlap the Open5GS ogstun route on the host.  The
    # downlink broker binds only the Docker bridge gateway; UE uplink is
    # published back to host loopback by the launcher.
    ZmqRadioBroker(gnb_dl="tcp://127.0.0.1:2000", gnb_ul_listen="tcp://127.0.0.1:2001",
                   ue_dl_listen="tcp://172.17.0.1:2002", ue_ul="tcp://127.0.0.1:2003",
                   jammer=JammerConfig(JammerType(args.jammer), args.jsr_db, args.seed),
                   truth_log=args.truth_log, status_log=args.status_log,
                   sensor_log=args.sensor_log,
                   silence_samples=args.silence_samples,
                   startup_discard_limit=args.startup_discard_limit,
                   sample_rate_hz=args.sample_rate_hz,
                   ue_timeout_ms=args.ue_timeout_ms,
                   jammer_start_seconds=args.jammer_start_seconds,
                   jammer_stop_seconds=args.jammer_stop_seconds,
                   attenuation_db=args.attenuation_db,
                   experiment_start_file=args.experiment_start_file,
                   allow_uplink_silence=args.allow_uplink_silence).run()
    return 0


if __name__ == "__main__": raise SystemExit(main())
