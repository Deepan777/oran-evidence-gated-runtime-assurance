#!/usr/bin/env python3
"""Send one JSON command to OCUDU's loopback WebSocket control port."""
from __future__ import annotations

import argparse
import json
import os
import socket
import struct


def _read_exact(sock: socket.socket, size: int) -> bytes:
    chunks: list[bytes] = []
    while size:
        chunk = sock.recv(size)
        if not chunk:
            raise ConnectionError("WebSocket closed while reading a frame")
        chunks.append(chunk)
        size -= len(chunk)
    return b"".join(chunks)


def _send_frame(sock: socket.socket, payload: bytes, opcode: int = 1) -> None:
    mask = os.urandom(4)
    header = bytearray([0x80 | opcode])
    if len(payload) < 126:
        header.append(0x80 | len(payload))
    elif len(payload) <= 0xFFFF:
        header.extend((0x80 | 126,))
        header.extend(struct.pack("!H", len(payload)))
    else:
        header.extend((0x80 | 127,))
        header.extend(struct.pack("!Q", len(payload)))
    masked = bytes(value ^ mask[index % 4] for index, value in enumerate(payload))
    sock.sendall(bytes(header) + mask + masked)


def _recv_frame(sock: socket.socket) -> tuple[int, bytes]:
    first, second = _read_exact(sock, 2)
    length = second & 0x7F
    if length == 126:
        length = struct.unpack("!H", _read_exact(sock, 2))[0]
    elif length == 127:
        length = struct.unpack("!Q", _read_exact(sock, 8))[0]
    mask = _read_exact(sock, 4) if second & 0x80 else b""
    payload = _read_exact(sock, length)
    if mask:
        payload = bytes(value ^ mask[index % 4] for index, value in enumerate(payload))
    return first & 0x0F, payload


def send(host: str, port: int, document: object, timeout: float) -> object:
    import base64
    import hashlib
    key = base64.b64encode(os.urandom(16)).decode("ascii")
    with socket.create_connection((host, port), timeout=timeout) as sock:
        request = (
            f"GET / HTTP/1.1\r\nHost: {host}:{port}\r\nUpgrade: websocket\r\n"
            f"Connection: Upgrade\r\nSec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n"
        )
        sock.sendall(request.encode("ascii"))
        response = bytearray()
        while b"\r\n\r\n" not in response:
            response.extend(sock.recv(4096))
        expected = base64.b64encode(
            hashlib.sha1((key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11").encode()).digest()
        ).decode()
        headers = response.decode("latin-1")
        if " 101 " not in headers or expected.lower() not in headers.lower():
            raise RuntimeError(f"WebSocket handshake failed: {headers!r}")
        _send_frame(sock, json.dumps(document, separators=(",", ":")).encode())
        sock.settimeout(timeout)
        opcode, payload = _recv_frame(sock)
        if opcode != 1:
            raise RuntimeError(f"unexpected WebSocket opcode {opcode}")
        return json.loads(payload.decode("utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--timeout", type=float, default=3.0)
    parser.add_argument("--json", required=True)
    args = parser.parse_args()
    response = send(args.host, args.port, json.loads(args.json), args.timeout)
    print(json.dumps(response, sort_keys=True))
    return 0 if response.get("status") != "error" else 1


if __name__ == "__main__":
    raise SystemExit(main())
