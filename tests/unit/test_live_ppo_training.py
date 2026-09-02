from __future__ import annotations

import numpy as np

from experiments.training.train_live_ppo_controller import _advantages


def test_ppo_advantages_have_terminal_bootstrap_zero() -> None:
    advantages, returns = _advantages([1.0, 2.0], [0.5, 0.75], [False, True])
    assert advantages.shape == returns.shape == (2,)
    assert np.isclose(returns[-1], 2.0)
