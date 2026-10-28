#!/usr/bin/env python3
"""Capture OCUDU remote-control JSON metrics without third-party packages."""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import socket
import struct
import time
from pathlib import Path


GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"


def _read_exact(sock: socket.socket, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = sock.recv(remaining)
        if not chunk:
            raise ConnectionError("WebSocket closed while reading a frame")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _send_frame(sock: socket.socket, payload: bytes, opcode: int = 1) -> None:
    mask = os.urandom(4)
    length = len(payload)
    header = bytearray([0x80 | opcode])
    if length < 126:
        header.append(0x80 | length)
    elif length <= 0xFFFF:
        header.append(0x80 | 126)
        header.extend(struct.pack("!H", length))
    else:
        header.append(0x80 | 127)
        header.extend(struct.pack("!Q", length))
    masked = bytes(value ^ mask[index % 4] for index, value in enumerate(payload))
    sock.sendall(bytes(header) + mask + masked)


def _recv_frame(sock: socket.socket) -> tuple[int, bytes]:
    first, second = _read_exact(sock, 2)
    opcode = first & 0x0F
    masked = bool(second & 0x80)
    length = second & 0x7F
    if length == 126:
        length = struct.unpack("!H", _read_exact(sock, 2))[0]
    elif length == 127:
        length = struct.unpack("!Q", _read_exact(sock, 8))[0]
    mask = _read_exact(sock, 4) if masked else b""
    payload = _read_exact(sock, length)
    if masked:
        payload = bytes(value ^ mask[index % 4] for index, value in enumerate(payload))
    return opcode, payload


def capture(
    host: str,
    port: int,
    output: Path,
    count: int | None,
    timeout: float,
) -> int:
    key = base64.b64encode(os.urandom(16)).decode("ascii")
    with socket.create_connection((host, port), timeout=5.0) as sock:
        request = (
            f"GET / HTTP/1.1\r\nHost: {host}:{port}\r\nUpgrade: websocket\r\n"
            f"Connection: Upgrade\r\nSec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n"
        )
        sock.sendall(request.encode("ascii"))
        response = bytearray()
        while b"\r\n\r\n" not in response:
            response.extend(sock.recv(4096))
        headers = response.decode("latin-1")
        expected_accept = base64.b64encode(hashlib.sha1((key + GUID).encode()).digest()).decode()
        if " 101 " not in headers or expected_accept.lower() not in headers.lower():
            raise RuntimeError(f"WebSocket handshake failed: {headers!r}")
        _send_frame(sock, json.dumps({"cmd": "metrics_subscribe"}).encode())
        sock.settimeout(1.0)
        deadline = time.monotonic() + timeout
        captured = 0
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8", newline="\n") as stream:
            while (count is None or captured < count) and time.monotonic() < deadline:
                try:
                    opcode, payload = _recv_frame(sock)
                except socket.timeout:
                    continue
                if opcode == 9:
                    _send_frame(sock, payload, opcode=10)
                    continue
                if opcode == 8:
                    break
                if opcode != 1:
                    continue
                text = payload.decode("utf-8")
                document = json.loads(text)
                if document.get("cmd") == "metrics_subscribe":
                    continue
                stream.write(json.dumps(document, separators=(",", ":")) + "\n")
                stream.flush()
                captured += 1
        _send_frame(sock, json.dumps({"cmd": "metrics_unsubscribe"}).encode())
        return captured


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument(
        "--duration-seconds",
        type=float,
        help="Capture all metric envelopes for this duration instead of stopping at --count.",
    )
    parser.add_argument("--timeout", type=float, default=20.0)
    args = parser.parse_args()
    if args.duration_seconds is not None:
        if args.duration_seconds <= 0:
            parser.error("--duration-seconds must be positive")
        count = None
        timeout = args.duration_seconds
    else:
        count = args.count
        timeout = args.timeout
    captured = capture(args.host, args.port, args.output, count, timeout)
    print(f"OCUDU_JSON_CAPTURED={captured}")
    return 0 if (count is None and captured > 0) or captured == count else 1


if __name__ == "__main__":
    raise SystemExit(main())
