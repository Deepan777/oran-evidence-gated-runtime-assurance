#!/usr/bin/env python3
"""Exercise the production WebSocket client under controlled Linux netem faults."""
from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import os
import signal
import socket
import struct
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.send_ocudu_command import _recv_frame, send


CONDITIONS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("baseline", ()),
    ("delay_50ms", ("delay", "50ms")),
    ("delay_100ms_jitter_20ms", ("delay", "100ms", "20ms", "distribution", "normal")),
    ("loss_1pct", ("loss", "1%")),
    ("loss_5pct", ("loss", "5%")),
    ("loss_10pct", ("loss", "10%")),
    ("reorder_10pct", ("delay", "20ms", "reorder", "10%", "50%")),
    ("duplicate_5pct", ("duplicate", "5%")),
)


def _read_headers(conn: socket.socket) -> dict[str, str]:
    raw = bytearray()
    while b"\r\n\r\n" not in raw:
        chunk = conn.recv(4096)
        if not chunk:
            raise ConnectionError("client closed during handshake")
        raw.extend(chunk)
    lines = raw.decode("latin-1").split("\r\n")
    headers: dict[str, str] = {}
    for line in lines[1:]:
        if ":" in line:
            key, value = line.split(":", 1)
            headers[key.strip().lower()] = value.strip()
    return headers


def _send_server_frame(conn: socket.socket, payload: bytes) -> None:
    header = bytearray([0x81])
    if len(payload) < 126:
        header.append(len(payload))
    elif len(payload) <= 0xFFFF:
        header.append(126)
        header.extend(struct.pack("!H", len(payload)))
    else:
        header.append(127)
        header.extend(struct.pack("!Q", len(payload)))
    conn.sendall(bytes(header) + payload)


def serve(host: str, port: int) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((host, port))
        server.listen(32)
        while True:
            conn, _ = server.accept()
            with conn:
                try:
                    conn.settimeout(3.0)
                    headers = _read_headers(conn)
                    key = headers["sec-websocket-key"]
                    accept = base64.b64encode(hashlib.sha1(
                        (key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11").encode("ascii")
                    ).digest()).decode("ascii")
                    conn.sendall((
                        "HTTP/1.1 101 Switching Protocols\r\n"
                        "Upgrade: websocket\r\nConnection: Upgrade\r\n"
                        f"Sec-WebSocket-Accept: {accept}\r\n\r\n"
                    ).encode("ascii"))
                    opcode, payload = _recv_frame(conn)
                    request = json.loads(payload.decode("utf-8"))
                    response = {
                        "cmd": request.get("cmd"),
                        "status": "ok",
                        "action": request.get("action"),
                        "timestamp_ns": time.time_ns(),
                    }
                    _send_server_frame(conn, json.dumps(response, separators=(",", ":")).encode("utf-8"))
                except Exception:
                    continue


def _run(command: list[str], check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=check, text=True, capture_output=True)


def _set_netem(arguments: tuple[str, ...]) -> None:
    _run(["tc", "qdisc", "del", "dev", "lo", "root"], check=False)
    if arguments:
        _run(["tc", "qdisc", "add", "dev", "lo", "root", "netem", *arguments])


def _qdisc() -> str:
    return _run(["tc", "-s", "qdisc", "show", "dev", "lo"], check=False).stdout.strip()


def orchestrate(output: Path, attempts: int, port: int, timeout: float) -> int:
    if os.geteuid() != 0:
        raise SystemExit("run this study with sudo so netem can be applied and removed")
    if attempts <= 0:
        raise SystemExit("attempts must be positive")
    initial_qdisc = _qdisc()
    if "netem" in initial_qdisc:
        raise SystemExit(f"refusing to replace a pre-existing netem qdisc: {initial_qdisc}")
    output.parent.mkdir(parents=True, exist_ok=True)
    process = subprocess.Popen([
        sys.executable, str(Path(__file__).resolve()), "--serve", "--port", str(port)
    ], cwd=str(Path(__file__).resolve().parents[1]))
    rows: list[dict[str, Any]] = []
    condition_meta: list[dict[str, Any]] = []
    try:
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=0.1):
                    break
            except OSError:
                time.sleep(0.05)
        else:
            raise RuntimeError("acknowledgement server did not start")
        for condition, arguments in CONDITIONS:
            _set_netem(arguments)
            before = _qdisc()
            for attempt in range(attempts):
                request = {
                    "cmd": "research_action_set",
                    "action": "A2",
                    "mcs_cap": 8,
                    "attempt": attempt,
                }
                started = time.perf_counter_ns()
                success = False
                response: object | None = None
                error: str | None = None
                try:
                    response = send("127.0.0.1", port, request, timeout)
                    success = isinstance(response, dict) and response.get("cmd") == "research_action_set" and response.get("status") != "error"
                    if not success:
                        error = "invalid_acknowledgement"
                except Exception as exc:
                    error = f"{type(exc).__name__}:{exc}"
                latency_ms = (time.perf_counter_ns() - started) / 1_000_000.0
                rows.append({
                    "condition": condition,
                    "attempt": attempt,
                    "success": success,
                    "latency_ms": latency_ms,
                    "error": error,
                    "response": response,
                })
                time.sleep(0.01)
            condition_meta.append({"condition": condition, "netem": list(arguments), "qdisc_after_attempts": _qdisc(), "qdisc_before_attempts": before})
    finally:
        _set_netem(())
        process.send_signal(signal.SIGTERM)
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)

    summaries: list[dict[str, Any]] = []
    for condition, _ in CONDITIONS:
        selected = [row for row in rows if row["condition"] == condition]
        successful = [float(row["latency_ms"]) for row in selected if row["success"]]
        summaries.append({
            "condition": condition,
            "attempts": len(selected),
            "acknowledged": len(successful),
            "success_rate": len(successful) / len(selected),
            "median_success_latency_ms": float(np.median(successful)) if successful else None,
            "mean_success_latency_ms": float(np.mean(successful)) if successful else None,
            "p95_success_latency_ms": float(np.quantile(successful, 0.95)) if successful else None,
            "errors": sorted({str(row["error"]) for row in selected if row["error"]}),
        })
    document = {
        "schema_version": 1,
        "evidence_role": "COMMAND_TRANSPORT_IMPAIRMENT_CHARACTERIZATION",
        "scope_limitation": "isolated WebSocket command/acknowledgement transport; not a radio-service impairment result",
        "client": "tools.send_ocudu_command.send",
        "attempts_per_condition": attempts,
        "timeout_seconds": timeout,
        "initial_qdisc": initial_qdisc,
        "final_qdisc": _qdisc(),
        "conditions": condition_meta,
        "summaries": summaries,
        "attempts": rows,
    }
    output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    csv_path = output.with_suffix(".csv")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("condition", "attempt", "success", "latency_ms", "error"))
        writer.writeheader()
        writer.writerows({key: row[key] for key in writer.fieldnames} for row in rows)
    print(json.dumps(summaries, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18081)
    parser.add_argument("--output", type=Path, default=Path("results/reviewer_validation_r1/command_impairment.json"))
    parser.add_argument("--attempts", type=int, default=40)
    parser.add_argument("--timeout", type=float, default=0.75)
    args = parser.parse_args()
    if args.serve:
        return serve(args.host, args.port)
    return orchestrate(args.output.resolve(), args.attempts, args.port, args.timeout)


if __name__ == "__main__":
    raise SystemExit(main())

