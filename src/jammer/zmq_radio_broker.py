"""Bidirectional virtual-radio broker for OCUDU/srsUE startup interoperation.

The patched OCUDU runtime and legacy srsUE are started concurrently. Every
uplink request is paired with the UE reply to preserve the radio sample
timeline exactly; an explicit zero-IQ uplink fallback can be enabled for a UE
that has released RRC under severe jamming. The fallback is recorded as radio
silence (never as fabricated telemetry) so the downlink clock and windowed
observability continue while preserving the outage in raw evidence. Downlink
IQ always passes through the methodology-locked signal-domain jammer.
"""
from __future__ import annotations

import json
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import zmq

from src.jammer.waveforms import JammerConfig
from src.jammer.zmq_proxy import append_truth, process_frame
from src.detector.localization import BaselineRelativeSpectralMonitor


@dataclass
class BrokerCounters:
    downlink_frames: int = 0
    downlink_samples: int = 0
    downlink_clock_fill_frames: int = 0
    downlink_clock_fill_samples: int = 0
    startup_zero_frames: int = 0
    startup_zero_samples: int = 0
    uplink_forwarded_frames: int = 0
    uplink_silence_frames: int = 0
    uplink_samples: int = 0


def pacing_delay_seconds(sample_count: int, sample_rate_hz: float, elapsed_seconds: float) -> float:
    """Return only the unspent wall-clock budget for one radio frame.

    Frame processing is part of the frame interval. Sleeping a full sample
    duration *after* expensive impairment processing steadily drives the
    virtual radio behind its ZMQ timing contract and can release the UE's RRC
    session. This helper preserves the configured sample-rate pace while
    avoiding that accumulated extra delay.
    """
    if sample_count <= 0 or sample_rate_hz <= 0.0 or elapsed_seconds < 0.0:
        raise ValueError("invalid pacing inputs")
    return max(0.0, sample_count / sample_rate_hz - elapsed_seconds)


class ZmqRadioBroker:
    def __init__(self, *, gnb_dl: str, gnb_ul_listen: str, ue_dl_listen: str, ue_ul: str,
                 jammer: JammerConfig, truth_log: Path, status_log: Path,
                 sensor_log: Path | None = None,
                 silence_samples: int = 23040, ue_timeout_ms: int = 2,
                 startup_discard_limit: int = 10000,
                 sample_rate_hz: float = 23.04e6,
                 jammer_start_seconds: float = 0.0,
                 jammer_stop_seconds: float | None = None,
                 attenuation_db: float = 0.0,
                 experiment_start_file: Path | None = None,
                 allow_uplink_silence: bool = False,
                 spectral_reference_windows: int = 10) -> None:
        self.gnb_dl, self.gnb_ul_listen = gnb_dl, gnb_ul_listen
        self.ue_dl_listen, self.ue_ul = ue_dl_listen, ue_ul
        self.jammer, self.truth_log, self.status_log, self.sensor_log = jammer, truth_log, status_log, sensor_log
        self.silence_samples, self.ue_timeout_ms = silence_samples, ue_timeout_ms
        self.startup_discard_limit = startup_discard_limit
        self.sample_rate_hz = sample_rate_hz
        if jammer_start_seconds < 0 or (jammer_stop_seconds is not None and jammer_stop_seconds <= jammer_start_seconds):
            raise ValueError("invalid jammer activity interval")
        if attenuation_db < 0:
            raise ValueError("attenuation_db must be non-negative")
        self.jammer_start_samples = round(jammer_start_seconds * sample_rate_hz)
        self.jammer_stop_samples = None if jammer_stop_seconds is None else round(jammer_stop_seconds * sample_rate_hz)
        self.attenuation_linear = 10.0 ** (-attenuation_db / 20.0)
        self.experiment_start_file = experiment_start_file
        self.allow_uplink_silence = allow_uplink_silence
        if spectral_reference_windows < 1:
            raise ValueError("spectral_reference_windows must be positive")
        self.spectral_reference_windows = spectral_reference_windows
        self.stop_event = threading.Event()
        # Recovery frames are valid only after the immutable experiment start
        # marker. Before registration completes they would advance the gNB
        # without a UE and create a false radio clock.
        self.experiment_started_event = threading.Event()
        self.counters = BrokerCounters()
        self._lock = threading.Lock()

    def _write_status(self, event: str) -> None:
        self.status_log.parent.mkdir(parents=True, exist_ok=True)
        payload = {"timestamp_ns": time.time_ns(), "event": event, **asdict(self.counters)}
        with self.status_log.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")

    def _write_sensor(self, payload: dict[str, object]) -> None:
        if self.sensor_log is None:
            return
        self.sensor_log.parent.mkdir(parents=True, exist_ok=True)
        with self.sensor_log.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")

    def _new_ue_request(self, context: zmq.Context) -> zmq.Socket:
        socket = context.socket(zmq.REQ)
        socket.setsockopt(zmq.RCVTIMEO, self.ue_timeout_ms)
        socket.setsockopt(zmq.SNDTIMEO, self.ue_timeout_ms)
        socket.setsockopt(zmq.LINGER, 0)
        socket.connect(self.ue_ul)
        return socket

    def _downlink(self) -> None:
        context = zmq.Context()
        ue_rep = context.socket(zmq.REP); ue_rep.setsockopt(zmq.LINGER, 0); ue_rep.bind(self.ue_dl_listen)
        gnb_req = context.socket(zmq.REQ); gnb_req.setsockopt(zmq.LINGER, 0); gnb_req.connect(self.gnb_dl)
        sample_index = 0
        signal_acquired = False
        experiment_started = False
        experiment_sample_index = 0
        sensor_window_index = 0
        sensor_count = 0
        sensor_values = {"clean_prb_fraction": 0.0, "jammer_occupancy_fraction": 0.0,
                         "jammer_subband_start_norm": 0.0, "jammer_subband_end_norm": 0.0,
                         "localized_jammer_region_available": 0.0,
                         "spectral_reference_ready": 0.0,
                         "spectral_reference_observations": 0.0,
                         "spectral_divergence": 0.0, "spectral_excess_fraction": 0.0}
        spectral_monitor = BaselineRelativeSpectralMonitor()
        self._write_status("downlink_ready")
        try:
            while not self.stop_event.is_set():
                # Advance the downlink only for an observed UE request.  A
                # prior recovery implementation replayed the last request
                # after a 1 ms scheduling pause. That advanced the gNB by one
                # slot without advancing the UE and deterministically caused
                # the UE to report out-of-sync. In a virtual radio, pausing
                # both sides is valid; inventing an unobserved frame is not.
                if not ue_rep.poll(100):
                    continue
                request = ue_rep.recv()
                gnb_req.send(request); legitimate = gnb_req.recv()
                frame_started = time.perf_counter()
                count = len(legitimate) // np.dtype(np.complex64).itemsize
                nonzero = bool(np.any(np.frombuffer(legitimate, dtype=np.complex64)))
                if not signal_acquired and nonzero:
                    signal_acquired = True
                    experiment_sample_index = 0
                    self._write_status("downlink_signal_acquired")
                elif not signal_acquired:
                    with self._lock:
                        self.counters.startup_zero_frames += 1
                        self.counters.startup_zero_samples += count
                    if self.counters.startup_zero_frames in {1, 10, 100} or self.counters.startup_zero_frames % 1000 == 0:
                        self._write_status("downlink_startup_zero_forwarded")
                    if self.counters.startup_zero_frames >= self.startup_discard_limit:
                        raise RuntimeError(
                            f"downlink remained all-zero for {self.startup_discard_limit} startup frames"
                        )
                start_requested = self.experiment_start_file is None or self.experiment_start_file.is_file()
                if signal_acquired and not experiment_started and start_requested:
                    experiment_started = True
                    self.experiment_started_event.set()
                    experiment_sample_index = 0
                    self._write_status("experiment_clock_started")
                impaired = (
                    np.frombuffer(legitimate, dtype=np.complex64) * np.float32(self.attenuation_linear)
                ).astype(np.complex64).tobytes()
                jammer_enabled = (
                    experiment_started
                    and experiment_sample_index >= self.jammer_start_samples
                    and (self.jammer_stop_samples is None or experiment_sample_index < self.jammer_stop_samples)
                )
                altered, _, records = process_frame(
                    impaired, self.jammer, experiment_sample_index, jammer_enabled=jammer_enabled
                )
                append_truth(records, self.truth_log)
                ue_rep.send(altered)
                # This estimate reads only the radio frame delivered to the
                # UE.  The injected waveform/result and truth record stay
                # outside this path so live control cannot leak ground truth.
                if experiment_started and self.counters.downlink_frames % 20 == 0:
                    estimate = spectral_monitor.observe(
                        np.frombuffer(altered, dtype=np.complex64),
                        calibration=sensor_window_index < self.spectral_reference_windows,
                    )
                    for name, value in estimate.as_dict().items():
                        if name != "sampled_bins":
                            sensor_values[name] += float(value)
                    sensor_count += 1
                sample_index += count
                if experiment_started:
                    experiment_sample_index += count
                    while experiment_sample_index >= (sensor_window_index + 1) * int(self.sample_rate_hz):
                        if sensor_count:
                            averaged = {name: value / sensor_count for name, value in sensor_values.items()}
                            localized = averaged["localized_jammer_region_available"] >= 0.5
                            averaged["localized_jammer_region_available"] = float(localized)
                            if not localized:
                                averaged["jammer_subband_start_norm"] = 0.0
                                averaged["jammer_subband_end_norm"] = 0.0
                            self._write_sensor({
                                "timestamp_ns": time.time_ns(),
                                "source": "measured_iq_baseline_relative_spectral_estimator",
                                "spectral_reference_windows": self.spectral_reference_windows,
                                "window_index": sensor_window_index,
                                "radio_window_start_sample": sensor_window_index * int(self.sample_rate_hz),
                                "radio_window_end_sample": (sensor_window_index + 1) * int(self.sample_rate_hz),
                                "sampled_frames": sensor_count, **averaged,
                            })
                        sensor_window_index += 1
                        sensor_count = 0
                        sensor_values = {name: 0.0 for name in sensor_values}
                with self._lock:
                    self.counters.downlink_frames += 1; self.counters.downlink_samples += count
                if self.counters.downlink_frames in {1, 10, 100} or self.counters.downlink_frames % 1000 == 0:
                    self._write_status("downlink_progress")
                # ZMQ has no hardware sample clock. Pace every normal frame at
                # its sample duration so the gNB and legacy UE ring buffers do
                # not overflow and drop the RRC response after Msg3.
                time.sleep(pacing_delay_seconds(
                    count, self.sample_rate_hz, time.perf_counter() - frame_started
                ))
        finally:
            ue_rep.close(); gnb_req.close(); context.term()

    def _uplink(self) -> None:
        context = zmq.Context()
        gnb_rep = context.socket(zmq.REP); gnb_rep.setsockopt(zmq.LINGER, 0); gnb_rep.bind(self.gnb_ul_listen)
        ue_req = self._new_ue_request(context)
        silence = np.zeros(self.silence_samples, dtype=np.complex64).tobytes()
        silence_mode = False
        self._write_status("uplink_ready")
        try:
            while not self.stop_event.is_set():
                if not gnb_rep.poll(100): continue
                request = gnb_rep.recv(); response = None
                synthetic_response = False
                frame_started = time.perf_counter()
                # Once the UE has explicitly released RRC, never probe the
                # disconnected UE during the frozen radio window. A failed
                # REQ probe can block for up to the bounded recovery timeout
                # and cause a real-time gNB overflow. Continue the exact
                # gNB radio clock with explicit zero-IQ uplink frames; the
                # first silence transition and every frame remain logged.
                if silence_mode:
                    response = silence
                    synthetic_response = True
                else:
                    ue_req.send(request)
                # Never replace a late response with silence: doing so would
                # forward that response one request later and move
                # PRACH/PUSCH to the wrong slot.
                # Intermittent high-JSR runs can briefly starve the UE's
                # request thread for several seconds.  Keep the strict
                # no-synthetic-silence policy, but allow a bounded 30-second
                # recovery window before classifying the trial as incomplete.
                if silence_mode:
                    # This branch is retained for the initial transition;
                    # after entering silence mode no UE probe is attempted.
                    response_deadline = time.monotonic() + max(
                        0.1, min(1.0, self.ue_timeout_ms / 1000.0)
                    )
                else:
                    response_deadline = time.monotonic() + max(
                        30.0, 50.0 * self.ue_timeout_ms / 1000.0
                    )
                if response is None:
                    while not self.stop_event.is_set():
                        if ue_req.poll(max(100, self.ue_timeout_ms)):
                            response = ue_req.recv()
                            silence_mode = False
                            break
                        if time.monotonic() >= response_deadline:
                            if not self.allow_uplink_silence or not self.experiment_started_event.is_set():
                                event = (
                                    "uplink_timeout"
                                    if self.experiment_started_event.is_set()
                                    else "uplink_timeout_before_experiment"
                                )
                                self._write_status(event)
                                break
                            # A timed-out REQ socket cannot be reused safely.
                            # Recreate it before the next periodic probe.
                            ue_req.close(linger=0)
                            ue_req = self._new_ue_request(context)
                            silence_mode = True
                            self._write_status("uplink_silence_start")
                            response = silence
                            synthetic_response = True
                            break
                if response is None:
                    break
                # A legitimate UE transmit frame is frequently all-zero IQ.
                # Comparing its bytes to the recovery buffer misclassifies
                # normal idle uplink frames as synthetic silence.  Track the
                # provenance explicitly instead.
                if synthetic_response:
                    with self._lock: self.counters.uplink_silence_frames += 1
                else:
                    with self._lock: self.counters.uplink_forwarded_frames += 1
                gnb_rep.send(response)
                with self._lock: self.counters.uplink_samples += len(response) // 8
                total = self.counters.uplink_forwarded_frames + self.counters.uplink_silence_frames
                if total in {1, 10, 100} or total % 1000 == 0: self._write_status("uplink_progress")
                count = len(response) // np.dtype(np.complex64).itemsize
                time.sleep(pacing_delay_seconds(
                    count, self.sample_rate_hz, time.perf_counter() - frame_started
                ))
        finally:
            ue_req.close(); gnb_rep.close(); context.term()

    def run(self) -> None:
        self._write_status("broker_start")
        threads = [threading.Thread(target=self._downlink, daemon=True), threading.Thread(target=self._uplink, daemon=True)]
        for thread in threads: thread.start()
        try:
            while all(thread.is_alive() for thread in threads): time.sleep(0.25)
        finally:
            self.stop_event.set()
            for thread in threads: thread.join(timeout=2)
            self._write_status("broker_stop")
