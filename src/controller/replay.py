from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class ReplayBatch:
    states: NDArray[np.float32]
    actions: NDArray[np.int64]
    rewards: NDArray[np.float32]
    next_states: NDArray[np.float32]
    terminals: NDArray[np.bool_]
    next_masks: NDArray[np.bool_]
    indexes: NDArray[np.int64]
    importance_weights: NDArray[np.float32]


class PrioritizedReplayBuffer:
    def __init__(
        self,
        capacity: int,
        state_dim: int,
        action_count: int,
        alpha: float = 0.6,
        priority_epsilon: float = 1e-6,
        seed: int = 0,
    ) -> None:
        if capacity <= 0 or state_dim <= 0 or action_count <= 0:
            raise ValueError("capacity, state_dim, and action_count must be positive")
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must lie in [0, 1]")
        self.capacity = capacity
        self.alpha = alpha
        self.priority_epsilon = priority_epsilon
        self.rng = np.random.default_rng(seed)
        self.states = np.empty((capacity, state_dim), dtype=np.float32)
        self.actions = np.empty(capacity, dtype=np.int64)
        self.rewards = np.empty(capacity, dtype=np.float32)
        self.next_states = np.empty((capacity, state_dim), dtype=np.float32)
        self.terminals = np.empty(capacity, dtype=np.bool_)
        self.next_masks = np.empty((capacity, action_count), dtype=np.bool_)
        self.priorities = np.zeros(capacity, dtype=np.float64)
        self.position = 0
        self.size = 0

    def add(
        self,
        state: NDArray[np.floating],
        action: int,
        reward: float,
        next_state: NDArray[np.floating],
        terminal: bool,
        next_mask: NDArray[np.bool_],
    ) -> None:
        index = self.position
        self.states[index] = state
        self.actions[index] = action
        self.rewards[index] = reward
        self.next_states[index] = next_state
        self.terminals[index] = terminal
        self.next_masks[index] = next_mask
        current_max = float(self.priorities[: self.size].max()) if self.size else 1.0
        self.priorities[index] = max(current_max, self.priority_epsilon)
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, beta: float) -> ReplayBatch:
        if self.size < batch_size:
            raise ValueError("not enough transitions to sample the requested batch")
        if not 0.0 <= beta <= 1.0:
            raise ValueError("beta must lie in [0, 1]")
        scaled = np.power(self.priorities[: self.size], self.alpha)
        probabilities = scaled / scaled.sum()
        indexes = self.rng.choice(self.size, size=batch_size, replace=False, p=probabilities)
        weights = np.power(self.size * probabilities[indexes], -beta)
        weights /= weights.max()
        return ReplayBatch(
            self.states[indexes].copy(),
            self.actions[indexes].copy(),
            self.rewards[indexes].copy(),
            self.next_states[indexes].copy(),
            self.terminals[indexes].copy(),
            self.next_masks[indexes].copy(),
            indexes.astype(np.int64),
            weights.astype(np.float32),
        )

    def update_priorities(
        self,
        indexes: NDArray[np.integer],
        td_errors: NDArray[np.floating],
    ) -> None:
        index_values = np.asarray(indexes, dtype=np.int64)
        error_values = np.asarray(td_errors, dtype=np.float64)
        if index_values.shape != error_values.shape:
            raise ValueError("indexes and td_errors must have the same shape")
        if np.any(index_values < 0) or np.any(index_values >= self.size):
            raise IndexError("priority update index outside populated buffer")
        self.priorities[index_values] = np.abs(error_values) + self.priority_epsilon

