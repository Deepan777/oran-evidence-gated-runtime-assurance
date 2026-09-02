import numpy as np

from src.detector.localization import BaselineRelativeSpectralMonitor


def _frame(rng, *, jam_start=None, jam_stop=None, jam_amplitude=0.0, bins=1024):
    # Stable occupied OFDM-like spectrum with independent symbol phases.
    frequency = np.zeros(bins, dtype=np.complex64)
    active = slice(120, 904)
    phase = rng.uniform(0.0, 2.0 * np.pi, 784)
    frequency[active] = np.exp(1j * phase)
    if jam_start is not None:
        jam_phase = rng.uniform(0.0, 2.0 * np.pi, jam_stop - jam_start)
        frequency[jam_start:jam_stop] += jam_amplitude * np.exp(1j * jam_phase)
    return np.fft.ifft(np.fft.ifftshift(frequency)).astype(np.complex64)


def test_calibration_never_reports_jammer_and_clean_service_stays_quiet():
    rng = np.random.default_rng(17)
    monitor = BaselineRelativeSpectralMonitor(minimum_reference_observations=12)
    for _ in range(12):
        estimate = monitor.observe(_frame(rng), calibration=True)
        assert not estimate.localized_jammer_region_available
        assert estimate.jammer_occupancy_fraction == 0.0
    for _ in range(16):
        clean = monitor.observe(_frame(rng), calibration=False)
    assert clean.spectral_reference_ready
    assert clean.jammer_occupancy_fraction < 0.03


def test_localized_excess_is_detected_relative_to_clean_service():
    rng = np.random.default_rng(19)
    monitor = BaselineRelativeSpectralMonitor(
        minimum_reference_observations=16, excess_ratio=3.0,
    )
    for _ in range(16):
        monitor.observe(_frame(rng), calibration=True)
    for _ in range(16):
        jammed = monitor.observe(
            _frame(rng, jam_start=430, jam_stop=570, jam_amplitude=8.0),
            calibration=False,
        )
    assert jammed.spectral_reference_ready
    assert jammed.localized_jammer_region_available
    assert 0.35 < jammed.jammer_subband_start_norm < 0.50
    assert 0.50 < jammed.jammer_subband_end_norm < 0.65
    assert jammed.spectral_excess_fraction > 0.05
