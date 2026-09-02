from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
from numpy.typing import NDArray


ComplexArray = NDArray[np.complex64]
FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


class JammerType(str, Enum):
    NONE = "none"
    BARRAGE = "barrage"
    PARTIAL_BAND = "partial_band"
    SWEEP = "sweep"
    INTERMITTENT = "intermittent"
    REACTIVE = "reactive"


@dataclass(frozen=True)
class JammerConfig:
    jammer_type: JammerType
    jsr_db: float = 0.0
    seed: int = 0
    subband_start_norm: float = 0.25
    subband_end_norm: float = 0.50
    sweep_start_norm: float = -0.45
    sweep_end_norm: float = 0.45
    sweep_period_samples: int = 23_040_000
    duty_cycle: float = 0.50
    intermittent_period_samples: int = 4096
    reactive_energy_threshold: float = 0.05
    reactive_min_delay_samples: int = 256

    def validate(self) -> None:
        if not 0.0 <= self.subband_start_norm < self.subband_end_norm <= 1.0:
            raise ValueError("subband bounds must satisfy 0 <= start < end <= 1")
        if not -0.5 <= self.sweep_start_norm < self.sweep_end_norm <= 0.5:
            raise ValueError("sweep bounds must lie within normalized Nyquist band")
        if self.sweep_period_samples <= 1:
            raise ValueError("sweep_period_samples must be greater than one")
        if not 0.0 <= self.duty_cycle <= 1.0:
            raise ValueError("duty_cycle must lie in [0, 1]")
        if self.intermittent_period_samples <= 0:
            raise ValueError("intermittent_period_samples must be positive")
        if self.reactive_min_delay_samples < 0:
            raise ValueError("reactive_min_delay_samples must be non-negative")


@dataclass(frozen=True)
class JammerResult:
    output_iq: ComplexArray
    jammer_iq: ComplexArray
    active_mask: BoolArray
    legitimate_rms: float
    jammer_rms_active: float
    realized_jsr_db: float | None
    instantaneous_center_norm: FloatArray


def _complex_gaussian(rng: np.random.Generator, size: int) -> ComplexArray:
    samples = rng.normal(size=size) + 1j * rng.normal(size=size)
    return (samples / np.sqrt(2.0)).astype(np.complex64)


def _partial_band_noise(
    rng: np.random.Generator,
    size: int,
    start_norm: float,
    end_norm: float,
) -> ComplexArray:
    spectrum = np.zeros(size, dtype=np.complex128)
    frequencies = np.fft.fftfreq(size)
    lower = start_norm - 0.5
    upper = end_norm - 0.5
    selected = (frequencies >= lower) & (frequencies < upper)
    count = int(selected.sum())
    if count == 0:
        raise ValueError("partial-band selection contains no FFT bins")
    spectrum[selected] = rng.normal(size=count) + 1j * rng.normal(size=count)
    waveform = np.fft.ifft(spectrum)
    return waveform.astype(np.complex64)


def _sweep_waveform(size: int, config: JammerConfig, sample_offset: int) -> tuple[ComplexArray, FloatArray]:
    period = config.sweep_period_samples
    positions = (np.arange(size, dtype=np.int64) + sample_offset + config.seed) % period
    fraction = positions.astype(np.float64) / period
    center = config.sweep_start_norm + fraction * (config.sweep_end_norm - config.sweep_start_norm)
    phase = 2.0 * np.pi * np.cumsum(center)
    waveform = np.exp(1j * phase).astype(np.complex64)
    return waveform, center


def _intermittent_mask(size: int, config: JammerConfig, sample_offset: int) -> BoolArray:
    on_samples = int(round(config.duty_cycle * config.intermittent_period_samples))
    phase = config.seed % config.intermittent_period_samples
    indexes = (np.arange(size, dtype=np.int64) + sample_offset + phase) % config.intermittent_period_samples
    return indexes < on_samples


def _reactive_mask(legitimate: ComplexArray, config: JammerConfig) -> BoolArray:
    active_energy = np.abs(legitimate) >= config.reactive_energy_threshold
    trigger_indexes = np.flatnonzero(active_energy)
    mask = np.zeros(len(legitimate), dtype=np.bool_)
    if trigger_indexes.size == 0:
        return mask
    start = int(trigger_indexes[0]) + config.reactive_min_delay_samples
    if start < len(mask):
        mask[start:] = True
    return mask


def _rms(values: ComplexArray, mask: BoolArray | None = None) -> float:
    selected = values if mask is None else values[mask]
    if selected.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.abs(selected.astype(np.complex128)) ** 2)))


def inject_jammer(
    legitimate_iq: NDArray[np.complexfloating],
    config: JammerConfig,
    sample_offset: int = 0,
) -> JammerResult:
    """Add a deterministic signal-domain jammer to complex baseband samples."""
    config.validate()
    legitimate = np.asarray(legitimate_iq, dtype=np.complex64)
    if legitimate.ndim != 1 or legitimate.size == 0:
        raise ValueError("legitimate_iq must be a non-empty one-dimensional complex array")

    size = legitimate.size
    legitimate_rms = _rms(legitimate)
    if config.jammer_type is not JammerType.NONE and legitimate_rms <= 0.0:
        raise ValueError("non-zero legitimate RMS is required to define JSR")

    if sample_offset < 0:
        raise ValueError("sample_offset must be non-negative")
    # A frame-specific child stream prevents the same noise vector from being
    # replayed in every ZMQ frame while remaining exactly reproducible.
    rng = np.random.default_rng(np.random.SeedSequence([config.seed, sample_offset]))
    active = np.ones(size, dtype=np.bool_)
    center = np.full(size, np.nan, dtype=np.float64)

    if config.jammer_type is JammerType.NONE:
        raw = np.zeros(size, dtype=np.complex64)
        active[:] = False
    elif config.jammer_type is JammerType.BARRAGE:
        raw = _complex_gaussian(rng, size)
    elif config.jammer_type is JammerType.PARTIAL_BAND:
        raw = _partial_band_noise(
            rng,
            size,
            config.subband_start_norm,
            config.subband_end_norm,
        )
        center[:] = (config.subband_start_norm + config.subband_end_norm) / 2.0 - 0.5
    elif config.jammer_type is JammerType.SWEEP:
        raw, center = _sweep_waveform(size, config, sample_offset)
    elif config.jammer_type is JammerType.INTERMITTENT:
        raw = _complex_gaussian(rng, size)
        active = _intermittent_mask(size, config, sample_offset)
    elif config.jammer_type is JammerType.REACTIVE:
        raw = _complex_gaussian(rng, size)
        active = _reactive_mask(legitimate, config)
    else:  # pragma: no cover - Enum makes this defensive branch unreachable.
        raise ValueError(f"Unsupported jammer type: {config.jammer_type}")

    jammer = np.zeros(size, dtype=np.complex64)
    realized_jsr: float | None = None
    active_rms = 0.0
    if active.any():
        raw_active_rms = _rms(raw, active)
        if raw_active_rms <= 0.0:
            raise ValueError("generated active jammer has zero RMS")
        target_rms = legitimate_rms * (10.0 ** (config.jsr_db / 20.0))
        jammer[active] = raw[active] * np.float32(target_rms / raw_active_rms)
        active_rms = _rms(jammer, active)
        realized_jsr = float(20.0 * np.log10(active_rms / legitimate_rms))

    return JammerResult(
        output_iq=(legitimate + jammer).astype(np.complex64),
        jammer_iq=jammer,
        active_mask=active,
        legitimate_rms=legitimate_rms,
        jammer_rms_active=active_rms,
        realized_jsr_db=realized_jsr,
        instantaneous_center_norm=center,
    )
