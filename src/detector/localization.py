"""Measured IQ spectral localization for the live controller.

This estimator operates only on the altered downlink IQ received by the UE
side of the virtual radio.  It deliberately has no access to the jammer
configuration or truth log.  Its conservative availability flag prevents a
PRB-localized mitigation action when the observed spectrum is not localized.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from collections import deque

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class SpectralLocalization:
    clean_prb_fraction: float
    jammer_occupancy_fraction: float
    jammer_subband_start_norm: float
    jammer_subband_end_norm: float
    localized_jammer_region_available: bool
    sampled_bins: int

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class BaselineRelativeSpectralEstimate:
    """Prospective estimate relative to an observed clean-service spectrum."""

    clean_prb_fraction: float
    jammer_occupancy_fraction: float
    jammer_subband_start_norm: float
    jammer_subband_end_norm: float
    localized_jammer_region_available: bool
    sampled_bins: int
    spectral_reference_ready: bool
    spectral_reference_observations: int
    spectral_divergence: float
    spectral_excess_fraction: float

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


class BaselineRelativeSpectralMonitor:
    """Compare measured IQ with a pre-action clean-service spectral baseline.

    Absolute spectral occupancy is not a jammer detector because legitimate
    OFDM traffic itself occupies the band.  The prospective CEE controller
    therefore observes a mandatory action-free calibration period and freezes
    its normalized mean spectrum before mitigation is permitted.  No jammer
    configuration or truth label is available to this class.
    """

    def __init__(
        self, *, bins: int = 1024, minimum_reference_observations: int = 32,
        minimum_comparison_observations: int = 16, excess_ratio: float = 4.0,
    ) -> None:
        if (
            bins < 16 or minimum_reference_observations < 2
            or minimum_comparison_observations < 2 or excess_ratio <= 1.0
        ):
            raise ValueError("invalid baseline-relative spectral monitor settings")
        self.bins = bins
        self.minimum_reference_observations = minimum_reference_observations
        self.minimum_comparison_observations = minimum_comparison_observations
        self.excess_ratio = excess_ratio
        self._reference: NDArray[np.float64] | None = None
        self._observations = 0
        self._comparison: deque[NDArray[np.float64]] = deque(maxlen=minimum_comparison_observations)
        self._comparison_sum: NDArray[np.float64] | None = None

    @property
    def reference_ready(self) -> bool:
        return self._observations >= self.minimum_reference_observations

    def observe(
        self, iq: NDArray[np.complexfloating], *, calibration: bool,
    ) -> BaselineRelativeSpectralEstimate:
        spectrum = _normalized_spectrum(iq, self.bins)
        if calibration:
            self._comparison.clear()
            self._comparison_sum = None
            self._observations += 1
            if self._reference is None:
                self._reference = spectrum
            else:
                # Exact online mean; the reference is frozen when calibration
                # ends and is never adapted to a suspected interference event.
                self._reference += (spectrum - self._reference) / self._observations
            return self._neutral(spectrum.size)
        if not self.reference_ready or self._reference is None:
            return self._neutral(spectrum.size)

        if self._comparison_sum is None:
            self._comparison_sum = np.zeros_like(spectrum)
        if len(self._comparison) == self._comparison.maxlen:
            self._comparison_sum -= self._comparison[0]
        self._comparison.append(spectrum)
        self._comparison_sum += spectrum
        if len(self._comparison) < self.minimum_comparison_observations:
            return self._neutral(spectrum.size)
        spectrum = self._comparison_sum / len(self._comparison)

        ratio = spectrum / np.maximum(self._reference, np.finfo(np.float64).tiny)
        # A short circular moving average joins adjacent interfered resource
        # bins without turning isolated OFDM symbol variation into a region.
        smoothing_bins = min(9, max(1, spectrum.size // 64))
        kernel = np.full(smoothing_bins, 1.0 / smoothing_bins)
        padded = np.pad(ratio, (smoothing_bins // 2,), mode="wrap")
        smooth_ratio = np.convolve(padded, kernel, mode="valid")[: spectrum.size]
        excess = smooth_ratio >= self.excess_ratio
        occupancy = float(excess.mean())
        log_ratio = np.log(np.maximum(ratio, np.finfo(np.float64).tiny))
        divergence = float(np.sqrt(np.mean(log_ratio**2)))
        starts, ends = _contiguous_true_regions(excess)
        if not starts:
            return self._result(spectrum.size, occupancy, divergence, False, 0.0, 0.0)
        best = int(np.argmax(np.asarray(ends) - np.asarray(starts)))
        start, end = starts[best], ends[best]
        width = (end - start) / float(spectrum.size)
        localized = 0.03 <= width <= 0.75 and occupancy <= 0.80
        return self._result(
            spectrum.size, occupancy, divergence, localized,
            float(start / spectrum.size) if localized else 0.0,
            float(end / spectrum.size) if localized else 0.0,
        )

    def _neutral(self, sampled_bins: int) -> BaselineRelativeSpectralEstimate:
        return BaselineRelativeSpectralEstimate(
            1.0, 0.0, 0.0, 0.0, False, sampled_bins,
            self.reference_ready, self._observations, 0.0, 0.0,
        )

    def _result(
        self, sampled_bins: int, occupancy: float, divergence: float,
        localized: bool, start: float, end: float,
    ) -> BaselineRelativeSpectralEstimate:
        return BaselineRelativeSpectralEstimate(
            max(0.0, 1.0 - occupancy), occupancy, start, end, localized,
            sampled_bins, True, self._observations, divergence, occupancy,
        )


def _normalized_spectrum(
    iq: NDArray[np.complexfloating], bins: int,
) -> NDArray[np.float64]:
    values = np.asarray(iq, dtype=np.complex64).reshape(-1)
    if values.size < 16:
        raise ValueError("at least sixteen measured IQ samples are required")
    length = min(values.size, bins)
    # Hann taper reduces frame-boundary leakage.  Unit-sum normalization makes
    # the comparison insensitive to a uniform receiver-gain change.
    window = np.hanning(length).astype(np.float32)
    power = np.abs(np.fft.fftshift(np.fft.fft(values[:length] * window))) ** 2
    total = float(power.sum())
    if not np.isfinite(total) or total <= 0.0:
        return np.full(length, 1.0 / length, dtype=np.float64)
    return np.asarray(power / total, dtype=np.float64)


def estimate_spectral_localization(iq: NDArray[np.complexfloating], *, bins: int = 1024) -> SpectralLocalization:
    """Estimate a contiguous interference-dominated region from measured IQ.

    The result is a *controller estimate*, not ground truth.  If its simple
    robust spectral test cannot identify a bounded region, it reports
    ``localized_jammer_region_available=False`` and disables localization-
    dependent actions through the action mask.
    """
    values = np.asarray(iq, dtype=np.complex64).reshape(-1)
    if values.size < 16:
        raise ValueError("at least sixteen measured IQ samples are required")
    length = min(values.size, bins)
    # A contiguous observed prefix retains the frame's spectral structure;
    # evenly spaced decimation would alias a narrow-band component across the
    # full band.  The bounded prefix keeps forwarding-loop work constant.
    spectrum = np.abs(np.fft.fftshift(np.fft.fft(values[:length]))) ** 2
    floor = float(np.median(spectrum))
    if not np.isfinite(floor) or floor <= 0.0:
        return SpectralLocalization(1.0, 0.0, 0.0, 0.0, False, int(length))
    # A 6 dB robust excess test.  It is intentionally conservative: a broad
    # occupied OFDM spectrum is not falsely treated as a localizable jammer.
    occupied = spectrum >= floor * 4.0
    occupancy = float(occupied.mean())
    starts, ends = _contiguous_true_regions(occupied)
    if not starts:
        return SpectralLocalization(1.0, 0.0, 0.0, 0.0, False, int(length))
    best = int(np.argmax(np.asarray(ends) - np.asarray(starts)))
    start, end = starts[best], ends[best]
    width = (end - start) / float(length)
    localized = 0.03 <= width <= 0.75 and occupancy <= 0.80
    if not localized:
        return SpectralLocalization(max(0.0, 1.0 - occupancy), occupancy, 0.0, 0.0, False, int(length))
    return SpectralLocalization(
        max(0.0, 1.0 - occupancy), occupancy,
        float(start / length), float(end / length), True, int(length),
    )


def _contiguous_true_regions(mask: NDArray[np.bool_]) -> tuple[list[int], list[int]]:
    """Return half-open observed FFT-bin runs.

    A region that crosses the FFT wrap boundary is deliberately left split;
    reporting it as non-localized is safer than inventing a single invalid
    interval for a PRB control command.
    """
    values = np.asarray(mask, dtype=bool)
    if not values.any():
        return [], []
    transitions = np.diff(values.astype(np.int8), prepend=0, append=0)
    starts = np.flatnonzero(transitions == 1).tolist()
    ends = np.flatnonzero(transitions == -1).tolist()
    return starts, ends
