#!/usr/bin/env python3
"""Bounded diagnostic probe for one-byte srsRAN/OCUDU ZMQ IQ requests."""
from __future__ import annotations

import argparse

import zmq


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("endpoint")
    parser.add_argument("--timeout-ms", type=int, default=5000)
    args = parser.parse_args()
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.setsockopt(zmq.RCVTIMEO, args.timeout_ms)
    socket.setsockopt(zmq.LINGER, 0)
    socket.connect(args.endpoint)
    socket.send(b"\xff")
    data = socket.recv()
    print(f"endpoint={args.endpoint} bytes={len(data)} complex64_samples={len(data) // 8}")
    socket.close()
    context.term()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
