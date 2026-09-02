import numpy as np

from src.detector.training import select_simplest_detector


def test_detector_selection_is_telemetry_only() -> None:
    rng = np.random.default_rng(4)
    x = rng.normal(size=(80, 4))
    y = (x[:, 0] > 0).astype(int)
    result = select_simplest_detector(x[:60], y[:60], x[60:], y[60:], 4)
    assert result.model_name in {"logistic_regression", "random_forest", "gradient_boosted_trees", "small_mlp"}
    assert 0.0 <= result.macro_f1 <= 1.0
