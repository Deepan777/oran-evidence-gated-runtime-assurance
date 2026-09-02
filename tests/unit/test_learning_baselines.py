from __future__ import annotations

import numpy as np
import pytest
import torch

from src.baselines.learning import MaskedPPOPolicy, StandardQNetwork, ppo_loss, train_masked_ppo, train_standard_dqn
from src.baselines.policies import choose_baseline
from src.controller.replay import PrioritizedReplayBuffer
from src.environment.iq_calibrated_env import CalibrationScenario, IqCalibratedAntiJammingEnv
from src.jammer.waveforms import JammerType


def test_standard_dqn_baseline_trains_without_per_updates() -> None:
    buffer = PrioritizedReplayBuffer(128, 3, 5, seed=3)
    for index in range(80):
        buffer.add(np.full(3, index, dtype=np.float32), index % 5, 1.0, np.ones(3, dtype=np.float32), False, np.ones(5, dtype=bool))
    model, report = train_standard_dqn(buffer, 3, 3, steps=3, batch_size=16, target_update_steps=2)
    assert isinstance(model, StandardQNetwork)
    assert report.gradient_steps == 3


def test_learned_baselines_never_reintroduce_random_action_selection() -> None:
    state, mask = np.zeros(3, dtype=np.float32), np.asarray([True, False, True, False, False])
    with pytest.raises(ValueError, match="trained"):
        choose_baseline("B3", state, mask)
    ppo = MaskedPPOPolicy(3)
    assert choose_baseline("B4", state, mask, learned_policy=ppo).value in {0, 2}
    with pytest.raises(ValueError, match="PRB-only"):
        choose_baseline("B5", state, mask)
    prb_only = StandardQNetwork(3)
    assert choose_baseline("B5", state, mask, learned_policy=prb_only).value == 0


def test_masked_ppo_loss_is_finite() -> None:
    policy = MaskedPPOPolicy(3)
    states = torch.zeros((4, 3)); masks = torch.tensor([[True, True, False, False, False]] * 4)
    distribution = policy.distribution(states, masks)
    actions = torch.zeros(4, dtype=torch.int64)
    loss = ppo_loss(policy, states, masks, actions, distribution.log_prob(actions).detach(), torch.ones(4), torch.zeros(4))
    assert torch.isfinite(loss)


def test_masked_ppo_baseline_trains_on_explicit_offline_calibration_env() -> None:
    scenario = CalibrationScenario("ppo-test", JammerType.BARRAGE, 0.0, "low", "protected_only", "nominal", 8, steps=4)
    policy, report = train_masked_ppo(lambda: IqCalibratedAntiJammingEnv(scenario), 27, 8, rollout_steps=8, updates=1, epochs_per_update=1)
    assert isinstance(policy, MaskedPPOPolicy)
    assert report.collected_steps == 8
