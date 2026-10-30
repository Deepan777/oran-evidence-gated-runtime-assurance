#!/usr/bin/env python3
"""Apply A0--A4 to live OCUDU and persist exact request/response evidence."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from tools.send_ocudu_command import send


ACTIONS: tuple[dict[str, object], ...] = (
    {"cmd": "research_action_set", "action": "A0"},
    {"cmd": "research_action_set", "action": "A1", "prb_start": 26, "prb_stop": 52},
    {"cmd": "research_action_set", "action": "A2", "mcs_cap": 4},
    {"cmd": "research_action_set", "action": "A3", "slice_min_pct": 30, "slice_max_pct": 70},
    {"cmd": "research_action_set", "action": "A4", "prb_start": 26, "prb_stop": 52, "mcs_cap": 4},
    {"cmd": "research_action_set", "action": "A0"},
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--hold-seconds", type=float, default=3.0)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    for index, request in enumerate(ACTIONS):
        sent_ns = time.time_ns()
        response = send(args.host, args.port, request, 2.0)
        received_ns = time.time_ns()
        record = {
            "sequence": index,
            "sent_ns": sent_ns,
            "received_ns": received_ns,
            "latency_ms": (received_ns - sent_ns) / 1_000_000,
            "request": request,
            "response": response,
        }
        records.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)
        accepted = (
            isinstance(response, dict)
            and response.get("cmd") == "research_action_set"
            and response.get("status") != "error"
        )
        if not accepted:
            args.output.write_text("\n".join(json.dumps(item) for item in records) + "\n", encoding="utf-8")
            return 1
        time.sleep(args.hold_seconds)
    args.output.write_text("\n".join(json.dumps(item) for item in records) + "\n", encoding="utf-8")
    print("LIVE_ACTION_ACKNOWLEDGEMENTS=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
