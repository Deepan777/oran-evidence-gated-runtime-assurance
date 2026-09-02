from __future__ import annotations

import numpy as np
import pytest

from src.controller.replay import PrioritizedReplayBuffer


def populated_buffer() -> PrioritizedReplayBuffer:
    buffer = PrioritizedReplayBuffer(capacity=16, state_dim=3, action_count=5, seed=19)
    for index in range(10):
        state = np.full(3, index, dtype=np.float32)
        buffer.add(
            state,
            index % 5,
            float(index),
            state + 1,
            index == 9,
            np.ones(5, dtype=bool),
        )
    return buffer


def test_replay_sample_shapes_and_normalized_importance_weights() -> None:
    batch = populated_buffer().sample(batch_size=4, beta=0.4)
    assert batch.states.shape == (4, 3)
    assert batch.next_masks.shape == (4, 5)
    assert batch.importance_weights.max() == pytest.approx(1.0)
    assert np.all(batch.importance_weights > 0.0)


def test_priority_update_uses_absolute_td_error() -> None:
    buffer = populated_buffer()
    indexes = np.asarray([0, 1, 2])
    buffer.update_priorities(indexes, np.asarray([-3.0, 0.5, 2.0]))
    assert buffer.priorities[0] > buffer.priorities[2] > buffer.priorities[1]


def test_sampling_too_early_fails() -> None:
    buffer = PrioritizedReplayBuffer(capacity=4, state_dim=2, action_count=5)
    with pytest.raises(ValueError, match="not enough"):
        buffer.sample(batch_size=1, beta=0.4)

