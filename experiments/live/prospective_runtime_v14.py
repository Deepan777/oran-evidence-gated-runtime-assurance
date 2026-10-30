"""Real-runtime wrapper; no generated telemetry, acknowledgements or readback.

The original frozen runtime and guard are imported unchanged. Only the
explicitly exploratory G-pre arm changes the harm reference. Fault injection
acts on actual live command traffic and records the upstream observations.
"""
from __future__ import annotations
from collections import deque
from dataclasses import replace
import json
import math
import os
from pathlib import Path
import socket
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.live import run_xapp_loop as runtime
from src.controller.action_mask import Action
from src.controller.safety_guard import UncertaintySafetyGuard
from src.xapp.adapters import ocudu_websocket as backend_module
from tools import send_ocudu_command as transport

WINDOW = -1
INJECTED = False
OUT = Path(os.environ.get('ORAN_V14_EVIDENCE', '.'))
LOCK = threading.Lock()


def log(name, record):
    with LOCK, (OUT / name).open('a', encoding='utf-8') as stream:
        stream.write(json.dumps({'timestamp_ns': time.time_ns(), **record}, allow_nan=False) + '\n')


class ExploratoryPreActionGuard(UncertaintySafetyGuard):
    """Only the post-action harm test uses three measured pre-action windows."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.history = deque(maxlen=3)
        self.action_reference = None

    def decide(self, observation, *, applied_state_verified=True):
        # The base guard's only use of previous_action is its harm test.
        # Preserve all admission, identity, calibration and duration logic.
        original_previous = observation.previous_action
        decision = super().decide(replace(observation, previous_action=Action.NO_OP),
                                  applied_state_verified=applied_state_verified)
        ref = self.action_reference
        if (original_previous is not Action.NO_OP and decision.observation_eligible
                and ref is not None and (
                    observation.throughput_mbps < ref['throughput_mbps'] * self.config.rollback_throughput_ratio
                    or observation.latency_ms > max(ref['latency_ms'] * self.config.rollback_latency_ratio,
                                                    ref['latency_ms'] + 1.0))):
            decision = self._fallback('post_action_harm_boundary_crossed', cooldown=True,
                                      eligible=True, identity=decision.observation_identity)
        if decision.observation_eligible and all(math.isfinite(v) for v in
                (observation.throughput_mbps, observation.latency_ms)):
            if self.history and observation.window_index != self.history[-1][0] + 1:
                self.history.clear()
            self.history.append((observation.window_index, observation.throughput_mbps, observation.latency_ms))
        if decision.allow_policy:
            if len(self.history) != 3:
                decision = self.reject_invalid('exploratory_pre_action_reference_unavailable')
            elif original_previous is Action.NO_OP:
                self.action_reference = {
                    'windows': [v[0] for v in self.history],
                    'throughput_mbps': sum(v[1] for v in self.history) / 3,
                    'latency_ms': sum(v[2] for v in self.history) / 3,
                }
        elif original_previous is Action.NO_OP:
            self.action_reference = None
        log('exploratory-gpre-reference.jsonl', {'exploratory': True,
            'window_index': observation.window_index, 'reference_evaluated': ref,
            'reference_for_authorized_episode': self.action_reference,
            'decision_reason': decision.reason})
        return decision


def headers(sock):
    data = bytearray()
    while not data.endswith(b'\r\n\r\n'):
        part = sock.recv(1)
        if not part:
            raise ConnectionError('Live peer closed during handshake')
        data.extend(part)
        if len(data) > 65536:
            raise ValueError('Oversize live handshake')
    return bytes(data)


def impaired_send(host, port, document, timeout, fault, window):
    """One-command loopback proxy: drop a real ack or a real command."""
    listener = socket.socket()
    listener.bind(('127.0.0.1', 0))
    listener.listen(1)
    listener.settimeout(3)
    proxy_port = listener.getsockname()[1]
    def worker():
        try:
            with listener, listener.accept()[0] as client, socket.create_connection((host, port), timeout=3) as upstream:
                client.settimeout(3); upstream.settimeout(3)
                upstream.sendall(headers(client))
                client.sendall(headers(upstream))
                opcode, payload = transport._recv_frame(client)
                received = json.loads(payload.decode())
                if opcode != 1:
                    raise ValueError('Unexpected live command frame')
                event = {'fault': fault, 'window_index': window, 'proxy_received_command': received,
                         'upstream_command_sent': False, 'acknowledgement_withheld': True}
                if fault == 'lost_ack':
                    transport._send_frame(upstream, payload)
                    event['upstream_command_sent'] = True
                    reply_opcode, reply = transport._recv_frame(upstream)
                    event['upstream_ack_opcode'] = reply_opcode
                    event['actual_upstream_acknowledgement'] = json.loads(reply.decode())
                # command_timeout deliberately withholds the request, after
                # a genuine handshake, until the actual client socket times out.
                log('fault-injection.jsonl', event)
                time.sleep(timeout + .25)
        except Exception as exc:
            log('fault-injection.jsonl', {'fault': fault, 'window_index': window,
                'proxy_error': type(exc).__name__ + ':' + str(exc)})
    threading.Thread(target=worker, daemon=True).start()
    return transport.send('127.0.0.1', proxy_port, document, timeout)


ORIGINAL_SEND = backend_module.send


def measured_send(host, port, document, timeout):
    global INJECTED
    fault = os.environ.get('ORAN_V14_FAULT', 'none')
    target = int(os.environ.get('ORAN_V14_FAULT_WINDOW', '-1'))
    is_mismatch = fault == 'applied_parameter_mismatch'
    trigger = not INJECTED and ((WINDOW >= target and document.get('action') != 'A0')
                               if is_mismatch else WINDOW == target)
    trigger = trigger and fault != 'none'
    wire = dict(document)
    if trigger and is_mismatch:
        keys = [k for k in ('mcs_cap','prb_start','slice_min_pct') if k in wire]
        if not keys:
            trigger = False
        else:
            key = keys[0]
            wire[key] = max(0, int(wire[key]) - 1) if int(wire[key]) > 0 else 1
    event = {'window_index': WINDOW, 'requested_document': document,
             'wire_document': wire, 'injection_requested': fault,
             'injection_triggered': trigger, 'upstream_host': host, 'upstream_port': port}
    if trigger:
        INJECTED = True
        if is_mismatch:
            log('fault-injection.jsonl', {'fault': fault, 'window_index': WINDOW,
                'intended_document': document, 'actual_wire_document': wire})
    try:
        if trigger and fault in ('lost_ack','command_timeout'):
            response = impaired_send(host, port, wire, timeout, fault, WINDOW)
        else:
            response = ORIGINAL_SEND(host, port, wire, timeout)
        event['actual_response'] = response
        return response
    except Exception as exc:
        event['actual_exception'] = type(exc).__name__ + ':' + str(exc)
        raise
    finally:
        log('command-transport.jsonl', event)


def main():
    global WINDOW
    original_telemetry = runtime._telemetry_window
    def telemetry(*args, **kwargs):
        global WINDOW
        result = original_telemetry(*args, **kwargs)
        if result is not None:
            WINDOW = int(result[1]['sensor']['window_index'])
        else:
            # A missing receiver window still has a real sensor completion.
            WINDOW = next((int(r['window_index']) for r in runtime._read_json_lines(args[2])
                           if int(r['timestamp_ns']) == int(args[4])-1), -1)
        return result
    runtime._telemetry_window = telemetry
    backend_module.send = measured_send
    if os.environ.get('ORAN_V14_GPRE') == '1':
        runtime.UncertaintySafetyGuard = ExploratoryPreActionGuard
    return runtime.main()


if __name__ == '__main__':
    raise SystemExit(main())
