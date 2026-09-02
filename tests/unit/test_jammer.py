from __future__ import annotations

import numpy as np
import pytest

from src.jammer.waveforms import JammerConfig, JammerType, inject_jammer


def legitimate_signal(size: int = 65536) -> np.ndarray:
    indexes = np.arange(size, dtype=np.float64)
    return np.exp(2j * np.pi * 0.05 * indexes).astype(np.complex64)


def test_none_mode_is_bit_exact_pass_through() -> None:
    legitimate = legitimate_signal(4096)
    result = inject_jammer(legitimate, JammerConfig(JammerType.NONE, seed=7))
    np.testing.assert_array_equal(result.output_iq, legitimate)
    np.testing.assert_array_equal(result.jammer_iq, np.zeros_like(legitimate))
    assert not result.active_mask.any()
    assert result.realized_jsr_db is None


@pytest.mark.parametrize("jammer_type", list(JammerType)[1:])
def test_deterministic_replay_by_seed(jammer_type: JammerType) -> None:
    legitimate = legitimate_signal()
    config = JammerConfig(jammer_type, jsr_db=5.0, seed=23)
    first = inject_jammer(legitimate, config)
    second = inject_jammer(legitimate, config)
    np.testing.assert_array_equal(first.jammer_iq, second.jammer_iq)
    np.testing.assert_array_equal(first.active_mask, second.active_mask)


@pytest.mark.parametrize("requested_jsr", [-5.0, 0.0, 5.0, 10.0])
def test_realized_jsr_is_within_half_db(requested_jsr: float) -> None:
    result = inject_jammer(
        legitimate_signal(),
        JammerConfig(JammerType.BARRAGE, jsr_db=requested_jsr, seed=11),
    )
    assert result.realized_jsr_db is not None
    assert abs(result.realized_jsr_db - requested_jsr) <= 0.5


def test_intermittent_duty_cycle_error_is_within_two_percentage_points() -> None:
    config = JammerConfig(
        JammerType.INTERMITTENT,
        seed=4,
        duty_cycle=0.37,
        intermittent_period_samples=1000,
    )
    result = inject_jammer(legitimate_signal(100_000), config)
    assert abs(float(result.active_mask.mean()) - 0.37) <= 0.02


def test_partial_band_energy_is_spectrally_localized() -> None:
    size = 65536
    config = JammerConfig(
        JammerType.PARTIAL_BAND,
        seed=3,
        subband_start_norm=0.25,
        subband_end_norm=0.50,
    )
    result = inject_jammer(legitimate_signal(size), config)
    spectrum = np.abs(np.fft.fft(result.jammer_iq)) ** 2
    frequencies = np.fft.fftfreq(size)
    inside = (frequencies >= -0.25) & (frequencies < 0.0)
    energy_fraction = float(spectrum[inside].sum() / spectrum.sum())
    assert energy_fraction >= 0.99


def test_sweep_trajectory_is_monotonic_and_bounded() -> None:
    config = JammerConfig(
        JammerType.SWEEP,
        sweep_start_norm=-0.4,
        sweep_end_norm=0.4,
        sweep_period_samples=8192,
    )
    result = inject_jammer(legitimate_signal(8192), config)
    center = result.instantaneous_center_norm
    assert np.all(np.diff(center) > 0.0)
    assert center[0] == pytest.approx(-0.4)
    assert center[-1] < 0.4


def test_reactive_jammer_respects_minimum_delay() -> None:
    legitimate = np.zeros(4096, dtype=np.complex64)
    legitimate[500:] = 1.0 + 0.0j
    config = JammerConfig(
        JammerType.REACTIVE,
        reactive_energy_threshold=0.5,
        reactive_min_delay_samples=128,
    )
    result = inject_jammer(legitimate, config)
    assert not result.active_mask[:628].any()
    assert result.active_mask[628:].all()


def test_stream_offset_advances_noise_and_intermittent_schedule() -> None:
    signal = np.ones(512, dtype=np.complex64)
    barrage = JammerConfig(JammerType.BARRAGE, jsr_db=0.0, seed=123)
    first = inject_jammer(signal, barrage, sample_offset=0)
    second = inject_jammer(signal, barrage, sample_offset=512)
    replay = inject_jammer(signal, barrage, sample_offset=512)
    assert not np.array_equal(first.jammer_iq, second.jammer_iq)
    assert np.array_equal(second.jammer_iq, replay.jammer_iq)

    intermittent = JammerConfig(
        JammerType.INTERMITTENT,
        seed=7,
        duty_cycle=0.25,
        intermittent_period_samples=128,
    )
    whole = inject_jammer(np.ones(256, dtype=np.complex64), intermittent, sample_offset=0)
    left = inject_jammer(np.ones(80, dtype=np.complex64), intermittent, sample_offset=0)
    right = inject_jammer(np.ones(176, dtype=np.complex64), intermittent, sample_offset=80)
    assert np.array_equal(whole.active_mask, np.concatenate((left.active_mask, right.active_mask)))
