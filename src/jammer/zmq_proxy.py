"""Transparent ZeroMQ IQ forwarding and reproducible jammer injection.

The proxy deliberately works on raw interleaved ``complex64`` IQ frames.  It
does not generate or alter KPIs: every impairment is applied before the UE
receives the downlink sample stream.  The module is usable both as a bounded
test harness and as the production ZMQ forwarding worker.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import BinaryIO

import numpy as np
import zmq

from src.jammer.waveforms import JammerConfig, JammerResult, inject_jammer


@dataclass(frozen=True)
class TruthRecord:
    timestamp_ns: int
    sample_index_start: int
    sample_index_end: int
    jammer_type: str
    requested_jsr_db: float
    realized_jsr_db: float | None
    active: bool
    subband_start_norm: float
    subband_end_norm: float
    instantaneous_center_norm: float | None
    duty_cycle: float
    seed: int
    active_sample_count: int
    frame_sample_count: int


def process_frame(
    frame: bytes,
    config: JammerConfig,
    sample_index: int,
    *,
    jammer_enabled: bool = True,
) -> tuple[bytes, JammerResult, list[TruthRecord]]:
    """Forward one complex64 frame with a deterministic signal-domain jammer."""
    if len(frame) == 0 or len(frame) % np.dtype(np.complex64).itemsize:
        raise ValueError("IQ frame must contain a non-empty integral number of complex64 samples")
    legitimate = np.frombuffer(frame, dtype=np.complex64).copy()
    if not jammer_enabled:
        zeros = np.zeros_like(legitimate)
        result = JammerResult(
            output_iq=legitimate.copy(), jammer_iq=zeros,
            active_mask=np.zeros(legitimate.size, dtype=np.bool_),
            legitimate_rms=float(np.sqrt(np.mean(np.abs(legitimate.astype(np.complex128)) ** 2))),
            jammer_rms_active=0.0, realized_jsr_db=None,
            instantaneous_center_norm=np.full(legitimate.size, np.nan),
        )
        return result.output_iq.tobytes(), result, [TruthRecord(
            timestamp_ns=time.time_ns(), sample_index_start=sample_index,
            sample_index_end=sample_index + legitimate.size,
            jammer_type=config.jammer_type.value, requested_jsr_db=config.jsr_db,
            realized_jsr_db=None, active=False,
            subband_start_norm=config.subband_start_norm,
            subband_end_norm=config.subband_end_norm,
            instantaneous_center_norm=None, duty_cycle=config.duty_cycle, seed=config.seed,
            active_sample_count=0, frame_sample_count=legitimate.size,
        )]
    if config.jammer_type.value != "none" and not np.any(legitimate):
        # Startup/idle buffers contain no legitimate energy, so JSR is
        # undefined and no calibrated jammer may be added. Preserve the bytes
        # and explicitly record an inactive interval; the next non-zero frame
        # is calibrated normally.
        zeros = np.zeros_like(legitimate)
        result = JammerResult(
            output_iq=legitimate.copy(), jammer_iq=zeros,
            active_mask=np.zeros(legitimate.size, dtype=np.bool_),
            legitimate_rms=0.0, jammer_rms_active=0.0,
            realized_jsr_db=None,
            instantaneous_center_norm=np.full(legitimate.size, np.nan),
        )
        return result.output_iq.tobytes(), result, [TruthRecord(
            timestamp_ns=time.time_ns(), sample_index_start=sample_index,
            sample_index_end=sample_index + legitimate.size,
            jammer_type=config.jammer_type.value,
            requested_jsr_db=config.jsr_db, realized_jsr_db=None,
            active=False, subband_start_norm=config.subband_start_norm,
            subband_end_norm=config.subband_end_norm,
            instantaneous_center_norm=None, duty_cycle=config.duty_cycle,
            seed=config.seed, active_sample_count=0, frame_sample_count=legitimate.size,
        )]
    result = inject_jammer(legitimate, config, sample_offset=sample_index)
    records: list[TruthRecord] = []
    active = result.active_mask
    if active.any():
        start = sample_index + int(np.flatnonzero(active)[0])
        end = sample_index + int(np.flatnonzero(active)[-1]) + 1
        centers = result.instantaneous_center_norm[active]
        finite_centers = centers[np.isfinite(centers)]
        records.append(TruthRecord(
            timestamp_ns=time.time_ns(), sample_index_start=start, sample_index_end=end,
            jammer_type=config.jammer_type.value, requested_jsr_db=config.jsr_db,
            realized_jsr_db=result.realized_jsr_db, active=True,
            subband_start_norm=config.subband_start_norm, subband_end_norm=config.subband_end_norm,
            instantaneous_center_norm=float(finite_centers[-1]) if finite_centers.size else None,
            duty_cycle=config.duty_cycle, seed=config.seed,
            active_sample_count=int(np.count_nonzero(active)), frame_sample_count=legitimate.size,
        ))
    else:
        records.append(TruthRecord(
            timestamp_ns=time.time_ns(), sample_index_start=sample_index,
            sample_index_end=sample_index + legitimate.size, jammer_type=config.jammer_type.value,
            requested_jsr_db=config.jsr_db, realized_jsr_db=None, active=False,
            subband_start_norm=config.subband_start_norm, subband_end_norm=config.subband_end_norm,
            instantaneous_center_norm=None, duty_cycle=config.duty_cycle, seed=config.seed,
            active_sample_count=0, frame_sample_count=legitimate.size,
        ))
    return result.output_iq.tobytes(), result, records


def append_truth(records: list[TruthRecord], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(asdict(record), sort_keys=True) + "\n")


def proxy_loop(
    upstream: str,
    downstream: str,
    config: JammerConfig,
    truth_log: Path,
    max_frames: int | None = None,
) -> int:
    """Run the downlink OCUDU REP--srsUE REQ bridge.

    OCUDU owns a REP transmitter socket and sends a frame only after the peer
    requests it.  The proxy therefore accepts the UE request on its REP
    socket, forwards that *unmodified* request to OCUDU through a REQ socket,
    receives the legitimate IQ response, injects the jammer, then replies to
    the UE.  This preserves the wire protocol and makes the jammer location
    unambiguous: only the returned downlink complex-IQ frame is altered.
    """
    context = zmq.Context.instance()
    ingress = context.socket(zmq.REP)
    egress = context.socket(zmq.REQ)
    ingress.bind(upstream)
    egress.connect(downstream)
    sample_index, frames = 0, 0
    try:
        while max_frames is None or frames < max_frames:
            request = ingress.recv()
            egress.send(request)
            legitimate_frame = egress.recv()
            altered, _, records = process_frame(legitimate_frame, config, sample_index)
            append_truth(records, truth_log)
            ingress.send(altered)
            sample_index += len(legitimate_frame) // np.dtype(np.complex64).itemsize
            frames += 1
    finally:
        ingress.close(linger=0)
        egress.close(linger=0)
    return frames
