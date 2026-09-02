import numpy as np

from src.controller.replay import PrioritizedReplayBuffer
from src.controller.training import train_md3qn_per


def test_md3qn_per_training_runs() -> None:
    buffer = PrioritizedReplayBuffer(100, 3, 5, seed=2)
    for i in range(80): buffer.add(np.ones(3, dtype=np.float32)*i, i % 5, 1.0, np.ones(3, dtype=np.float32), False, np.ones(5, dtype=bool))
    _, report = train_md3qn_per(buffer, 3, 2, steps=2, batch_size=16)
    assert report.gradient_steps == 2
