from __future__ import annotations

import json

import numpy as np

from src.jammer.waveforms import JammerConfig, JammerType
from src.jammer.zmq_proxy import append_truth, process_frame


def test_none_proxy_is_bitwise_passthrough(tmp_path) -> None:
    samples = (np.arange(64, dtype=np.float32) + 1j).astype(np.complex64)
    altered, _, truth = process_frame(samples.tobytes(), JammerConfig(JammerType.NONE), 11)
    assert altered == samples.tobytes()
    assert truth[0].active is False
    destination = tmp_path / "jammer_truth.jsonl"
    append_truth(truth, destination)
    assert json.loads(destination.read_text().strip())["sample_index_start"] == 11


def test_proxy_records_signal_domain_jamming() -> None:
    samples = np.ones(4096, dtype=np.complex64)
    altered, result, truth = process_frame(
        samples.tobytes(), JammerConfig(JammerType.BARRAGE, jsr_db=5.0, seed=7), 100
    )
    assert altered != samples.tobytes()
    assert abs((result.realized_jsr_db or 0.0) - 5.0) <= 0.5
    assert truth[0].active is True
    assert truth[0].sample_index_start == 100
    assert truth[0].active_sample_count == 4096
    assert truth[0].frame_sample_count == 4096


def test_non_none_jammer_passes_zero_energy_startup_frame() -> None:
    frame = np.zeros(128, dtype=np.complex64).tobytes()
    output, result, records = process_frame(
        frame, JammerConfig(JammerType.BARRAGE, jsr_db=10.0, seed=9), 2048
    )
    assert output == frame
    assert result.realized_jsr_db is None
    assert not result.active_mask.any()
    assert records[0].active is False
    assert records[0].requested_jsr_db == 10.0
    assert records[0].active_sample_count == 0


def test_disabled_jammer_preserves_signal_and_configured_truth() -> None:
    samples = np.ones(128, dtype=np.complex64)
    output, result, records = process_frame(
        samples.tobytes(), JammerConfig(JammerType.SWEEP, jsr_db=5.0, seed=9), 0,
        jammer_enabled=False,
    )
    assert output == samples.tobytes()
    assert not result.active_mask.any()
    assert records[0].jammer_type == "sweep"
    assert records[0].active is False
