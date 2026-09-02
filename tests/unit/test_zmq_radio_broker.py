import json
import socket
import threading
import time
from pathlib import Path

import numpy as np
import pytest
import zmq

from src.jammer.waveforms import JammerConfig, JammerType
from src.jammer.zmq_radio_broker import ZmqRadioBroker, pacing_delay_seconds


def test_frame_pacing_counts_processing_time_once() -> None:
    assert pacing_delay_seconds(23_040, 23_040_000.0, 0.0002) == pytest.approx(0.0008)
    assert pacing_delay_seconds(23_040, 23_040_000.0, 0.0015) == 0.0
    with pytest.raises(ValueError):
        pacing_delay_seconds(0, 23_040_000.0, 0.0)


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def test_broker_forwards_startup_zeros_without_losing_timeline(tmp_path: Path) -> None:
    if not hasattr(zmq, "Context"):
        pytest.skip("real PyZMQ sockets are unavailable in this test environment")
    ports = [_free_port() for _ in range(4)]
    gnb_dl, gnb_ul, ue_dl, ue_ul = [f"tcp://127.0.0.1:{port}" for port in ports]
    context = zmq.Context()
    gnb_rep = context.socket(zmq.REP); gnb_rep.bind(gnb_dl)
    ue_tx_rep = context.socket(zmq.REP); ue_tx_rep.bind(ue_ul)
    broker = ZmqRadioBroker(
        gnb_dl=gnb_dl, gnb_ul_listen=gnb_ul, ue_dl_listen=ue_dl, ue_ul=ue_ul,
        jammer=JammerConfig(JammerType.NONE), truth_log=tmp_path / "truth.jsonl",
        status_log=tmp_path / "status.jsonl", silence_samples=8,
    )
    broker_thread = threading.Thread(target=broker.run, daemon=True); broker_thread.start()
    time.sleep(0.1)

    received_requests = []
    real = np.arange(8, dtype=np.float32).astype(np.complex64).tobytes()

    def serve_gnb() -> None:
        for response in (np.zeros(8, np.complex64).tobytes(), real):
            received_requests.append(gnb_rep.recv())
            gnb_rep.send(response)

    gnb_thread = threading.Thread(target=serve_gnb); gnb_thread.start()
    ue_req = context.socket(zmq.REQ); ue_req.connect(ue_dl)
    ue_req.send(b"clock")
    assert ue_req.recv() == np.zeros(8, np.complex64).tobytes()
    ue_req.send(b"clock")
    assert ue_req.recv() == real
    gnb_thread.join(timeout=2)

    assert received_requests == [b"clock", b"clock"]
    assert broker.counters.startup_zero_frames == 1
    assert broker.counters.downlink_frames == 2
    events = [json.loads(line)["event"] for line in (tmp_path / "status.jsonl").read_text().splitlines()]
    assert "downlink_signal_acquired" in events

    broker.stop_event.set(); broker_thread.join(timeout=2)
    for sock in (ue_req, ue_tx_rep, gnb_rep): sock.close(linger=0)
    context.term()


def test_broker_keeps_late_uplink_reply(tmp_path: Path) -> None:
    if not hasattr(zmq, "Context"):
        pytest.skip("real PyZMQ sockets are unavailable in this test environment")
    ports = [_free_port() for _ in range(4)]
    gnb_dl, gnb_ul, ue_dl, ue_ul = [f"tcp://127.0.0.1:{port}" for port in ports]
    context = zmq.Context()
    ue_rep = context.socket(zmq.REP); ue_rep.bind(ue_ul)
    broker = ZmqRadioBroker(
        gnb_dl=gnb_dl, gnb_ul_listen=gnb_ul, ue_dl_listen=ue_dl, ue_ul=ue_ul,
        jammer=JammerConfig(JammerType.NONE), truth_log=tmp_path / "truth.jsonl",
        status_log=tmp_path / "status.jsonl", silence_samples=8, ue_timeout_ms=2,
    )
    broker_thread = threading.Thread(target=broker.run, daemon=True); broker_thread.start()
    time.sleep(0.1)

    late = np.ones(8, np.complex64).tobytes()
    def serve_ue_late() -> None:
        ue_rep.recv()
        time.sleep(0.008)
        ue_rep.send(late)

    ue_thread = threading.Thread(target=serve_ue_late); ue_thread.start()
    gnb_req = context.socket(zmq.REQ); gnb_req.connect(gnb_ul)
    gnb_req.send(b"clock")
    assert gnb_req.recv() == late
    ue_thread.join(timeout=2)
    assert broker.counters.uplink_forwarded_frames == 1
    assert broker.counters.uplink_silence_frames == 0

    broker.stop_event.set(); broker_thread.join(timeout=2)
    for sock in (gnb_req, ue_rep): sock.close(linger=0)
    context.term()


def test_legitimate_zero_uplink_is_not_synthetic_silence(tmp_path: Path) -> None:
    if not hasattr(zmq, "Context"):
        pytest.skip("real PyZMQ sockets are unavailable in this test environment")
    ports = [_free_port() for _ in range(4)]
    gnb_dl, gnb_ul, ue_dl, ue_ul = [f"tcp://127.0.0.1:{port}" for port in ports]
    context = zmq.Context()
    ue_rep = context.socket(zmq.REP); ue_rep.bind(ue_ul)
    broker = ZmqRadioBroker(
        gnb_dl=gnb_dl, gnb_ul_listen=gnb_ul, ue_dl_listen=ue_dl, ue_ul=ue_ul,
        jammer=JammerConfig(JammerType.NONE), truth_log=tmp_path / "truth.jsonl",
        status_log=tmp_path / "status.jsonl", silence_samples=8, ue_timeout_ms=2,
    )
    broker_thread = threading.Thread(target=broker.run, daemon=True); broker_thread.start()
    time.sleep(0.1)

    legitimate_zero = np.zeros(8, np.complex64).tobytes()

    def serve_ue_zero() -> None:
        ue_rep.recv()
        ue_rep.send(legitimate_zero)

    ue_thread = threading.Thread(target=serve_ue_zero); ue_thread.start()
    gnb_req = context.socket(zmq.REQ); gnb_req.connect(gnb_ul)
    gnb_req.send(b"clock")
    assert gnb_req.recv() == legitimate_zero
    ue_thread.join(timeout=2)
    assert broker.counters.uplink_forwarded_frames == 1
    assert broker.counters.uplink_silence_frames == 0

    broker.stop_event.set(); broker_thread.join(timeout=2)
    for sock in (gnb_req, ue_rep): sock.close(linger=0)
    context.term()


def test_experiment_clock_waits_for_start_marker(tmp_path: Path) -> None:
    if not hasattr(zmq, "Context"):
        pytest.skip("real PyZMQ sockets are unavailable in this test environment")
    ports = [_free_port() for _ in range(4)]
    gnb_dl, gnb_ul, ue_dl, ue_ul = [f"tcp://127.0.0.1:{port}" for port in ports]
    context = zmq.Context()
    gnb_rep = context.socket(zmq.REP); gnb_rep.bind(gnb_dl)
    ue_tx_rep = context.socket(zmq.REP); ue_tx_rep.bind(ue_ul)
    marker = tmp_path / "start.marker"
    broker = ZmqRadioBroker(
        gnb_dl=gnb_dl, gnb_ul_listen=gnb_ul, ue_dl_listen=ue_dl, ue_ul=ue_ul,
        jammer=JammerConfig(JammerType.BARRAGE, jsr_db=0.0, seed=3),
        truth_log=tmp_path / "truth.jsonl", status_log=tmp_path / "status.jsonl",
        silence_samples=8, sample_rate_hz=8.0, jammer_start_seconds=1.0,
        experiment_start_file=marker,
    )
    broker_thread = threading.Thread(target=broker.run, daemon=True); broker_thread.start()
    time.sleep(0.1)
    legitimate = np.ones(8, np.complex64).tobytes()

    def serve_gnb() -> None:
        for _ in range(3):
            gnb_rep.recv(); gnb_rep.send(legitimate)

    server = threading.Thread(target=serve_gnb); server.start()
    ue_req = context.socket(zmq.REQ); ue_req.connect(ue_dl)
    ue_req.send(b"clock"); assert ue_req.recv() == legitimate
    marker.touch()
    ue_req.send(b"clock"); assert ue_req.recv() == legitimate
    ue_req.send(b"clock"); assert ue_req.recv() != legitimate
    server.join(timeout=4)
    events = [json.loads(line)["event"] for line in (tmp_path / "status.jsonl").read_text().splitlines()]
    assert events.count("experiment_clock_started") == 1
    broker.stop_event.set(); broker_thread.join(timeout=2)
    for sock in (ue_req, ue_tx_rep, gnb_rep): sock.close(linger=0)
    context.term()


def test_downlink_clock_never_advances_without_a_ue_request(tmp_path: Path) -> None:
    if not hasattr(zmq, "Context"):
        pytest.skip("real PyZMQ sockets are unavailable in this test environment")
    ports = [_free_port() for _ in range(4)]
    gnb_dl, gnb_ul, ue_dl, ue_ul = [f"tcp://127.0.0.1:{port}" for port in ports]
    context = zmq.Context()
    gnb_rep = context.socket(zmq.REP); gnb_rep.bind(gnb_dl)
    ue_tx_rep = context.socket(zmq.REP); ue_tx_rep.bind(ue_ul)
    broker = ZmqRadioBroker(
        gnb_dl=gnb_dl, gnb_ul_listen=gnb_ul, ue_dl_listen=ue_dl, ue_ul=ue_ul,
        jammer=JammerConfig(JammerType.NONE), truth_log=tmp_path / "truth.jsonl",
        status_log=tmp_path / "status.jsonl", silence_samples=8,
        experiment_start_file=tmp_path / "not-created.marker",
    )
    broker_thread = threading.Thread(target=broker.run, daemon=True); broker_thread.start()
    time.sleep(0.1)
    legitimate = np.ones(8, np.complex64).tobytes()

    def serve_one() -> None:
        gnb_rep.recv(); gnb_rep.send(legitimate)

    server = threading.Thread(target=serve_one); server.start()
    ue_req = context.socket(zmq.REQ); ue_req.connect(ue_dl)
    ue_req.send(b"clock"); assert ue_req.recv() == legitimate
    server.join(timeout=2)
    marker = tmp_path / "not-created.marker"
    marker.touch()
    time.sleep(0.05)
    assert broker.counters.downlink_frames == 1
    assert broker.counters.downlink_clock_fill_frames == 0

    broker.stop_event.set(); broker_thread.join(timeout=2)
    for sock in (ue_req, ue_tx_rep, gnb_rep): sock.close(linger=0)
    context.term()
