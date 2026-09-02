from __future__ import annotations

import pytest

from analysis.xai_metrics import (
    faithfulness_ratio,
    features_for_attribution_mass,
    rank_stability,
    top_k_jaccard,
)


def test_stability_and_sparsity_metrics() -> None:
    assert top_k_jaccard([3, 2, 1], [4, 2, 0], k=2) == pytest.approx(1.0)
    assert rank_stability([3, 2, 1], [4, 2, 0]) == pytest.approx(1.0)
    assert features_for_attribution_mass([8, 1, 1], mass=0.8) == 1


def test_faithfulness_ratio_handles_zero_low_rank_effect() -> None:
    assert faithfulness_ratio(1.0, 0.0) == pytest.approx(1e12)

