from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy import stats


def top_k_jaccard(left: Sequence[float], right: Sequence[float], k: int) -> float:
    left_values = np.abs(np.asarray(left, dtype=np.float64))
    right_values = np.abs(np.asarray(right, dtype=np.float64))
    if left_values.shape != right_values.shape or left_values.ndim != 1:
        raise ValueError("attribution vectors must have the same one-dimensional shape")
    if not 0 < k <= left_values.size:
        raise ValueError("k must lie between one and the feature count")
    left_top = set(np.argsort(left_values)[-k:].tolist())
    right_top = set(np.argsort(right_values)[-k:].tolist())
    return len(left_top & right_top) / len(left_top | right_top)


def rank_stability(left: Sequence[float], right: Sequence[float]) -> float:
    left_values = np.abs(np.asarray(left, dtype=np.float64))
    right_values = np.abs(np.asarray(right, dtype=np.float64))
    if left_values.shape != right_values.shape or left_values.ndim != 1:
        raise ValueError("attribution vectors must have the same one-dimensional shape")
    result = stats.spearmanr(left_values, right_values)
    return float(result.statistic)


def features_for_attribution_mass(attribution: Sequence[float], mass: float = 0.80) -> int:
    values = np.abs(np.asarray(attribution, dtype=np.float64))
    if values.ndim != 1 or values.size == 0 or not 0.0 < mass <= 1.0:
        raise ValueError("invalid attribution vector or mass threshold")
    total = float(values.sum())
    if total == 0.0:
        return 0
    cumulative = np.cumsum(np.sort(values)[::-1]) / total
    return int(np.searchsorted(cumulative, mass, side="left") + 1)


def faithfulness_ratio(top_k_gap_effect: float, low_rank_gap_effect: float) -> float:
    if top_k_gap_effect < 0.0 or low_rank_gap_effect < 0.0:
        raise ValueError("Q-gap effects must be non-negative")
    return float(top_k_gap_effect / max(low_rank_gap_effect, 1e-12))

