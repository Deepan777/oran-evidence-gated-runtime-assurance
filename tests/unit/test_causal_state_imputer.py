import numpy as np

from src.controller.state import CausalStateImputer, StateSchema


def test_causal_imputer_uses_training_mean_then_previous_measured_value() -> None:
    schema = StateSchema(("x", "y"), np.asarray([2.0, 3.0]), np.asarray([1.0, 1.0]))
    imputer = CausalStateImputer(schema)
    first, first_source = imputer.fill({"x": 4.0}, ("x", "y"))
    assert first == {"x": 4.0, "y": 3.0}
    assert first_source == {"x": "measured", "y": "training_mean"}
    second, second_source = imputer.fill({}, ("x", "y"))
    assert second == {"x": 4.0, "y": 3.0}
    assert second_source == {"x": "previous_measured", "y": "training_mean"}
