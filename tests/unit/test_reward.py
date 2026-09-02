from __future__ import annotations

import pytest

from src.controller.reward import RewardInputs, RewardWeights, compute_reward


def test_reward_matches_locked_equation_and_decomposition() -> None:
    inputs = RewardInputs(
        throughput_norm=0.8,
        latency_norm=0.3,
        drop_norm=0.1,
        sla_violation=1.0,
        action_cost=0.2,
        action_switched=True,
    )
    weights = RewardWeights(
        throughput=1.0,
        latency=2.0,
        drop=3.0,
        sla_violation=4.0,
        action_cost=5.0,
        action_switch=6.0,
    )
    result = compute_reward(inputs, weights)
    assert result["reward_throughput"] == pytest.approx(0.8)
    assert result["reward_latency"] == pytest.approx(-0.6)
    assert result["reward_reliability"] == pytest.approx(-0.3)
    assert result["reward_service"] == pytest.approx(-4.0)
    assert result["reward_action_cost"] == pytest.approx(-1.0)
    assert result["reward_switching_cost"] == pytest.approx(-6.0)
    assert result["total_reward"] == pytest.approx(-11.1)


def test_negative_reward_weight_is_rejected() -> None:
    with pytest.raises(ValueError):
        compute_reward(
            RewardInputs(1.0, 0.0, 0.0, 0.0, 0.0, False),
            RewardWeights(throughput=-1.0),
        )

