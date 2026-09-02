from __future__ import annotations

import numpy as np
import pytest

from src.controller.state import StateBuilder, StateSchema


def test_state_builder_preserves_exact_order_and_standardizes() -> None:
    schema = StateSchema(
        feature_names=("b", "a"),
        means=np.asarray([10.0, 2.0]),
        scales=np.asarray([2.0, 4.0]),
    )
    state = StateBuilder(schema).build({"a": 6.0, "b": 12.0})
    np.testing.assert_allclose(state, np.asarray([1.0, 1.0], dtype=np.float32))


def test_state_builder_rejects_missing_or_extra_features() -> None:
    schema = StateSchema(("a",), np.zeros(1), np.ones(1))
    with pytest.raises(ValueError, match="mismatch"):
        StateBuilder(schema).build({"b": 1.0})

