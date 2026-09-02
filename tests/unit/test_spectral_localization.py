import numpy as np

from src.detector.localization import estimate_spectral_localization
from src.jammer.waveforms import JammerConfig, JammerType, inject_jammer


def test_measured_iq_estimator_localizes_partial_band_without_truth_input() -> None:
    generator = np.random.default_rng(17)
    legitimate = (generator.normal(size=4096) + 1j * generator.normal(size=4096)).astype(np.complex64)
    observed = inject_jammer(
        legitimate, JammerConfig(JammerType.PARTIAL_BAND, jsr_db=20.0, seed=19),
    ).output_iq
    result = estimate_spectral_localization(observed)
    assert result.localized_jammer_region_available
    assert 0.0 <= result.jammer_subband_start_norm < result.jammer_subband_end_norm <= 1.0
    assert result.jammer_occupancy_fraction > 0.0


def test_estimator_is_conservative_when_no_localized_region_is_observed() -> None:
    generator = np.random.default_rng(23)
    observed = (generator.normal(size=4096) + 1j * generator.normal(size=4096)).astype(np.complex64)
    result = estimate_spectral_localization(observed)
    assert not result.localized_jammer_region_available
    assert result.jammer_subband_start_norm == result.jammer_subband_end_norm == 0.0
