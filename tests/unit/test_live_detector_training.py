from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier

from experiments.live.run_xapp_loop import DETECTOR_FEATURES, JAMMER_CLASSES, _load_detector
from experiments.training.train_live_detector import FEATURES, causal_impute


def test_live_detector_causal_imputer_never_uses_a_future_window_value() -> None:
    frame = pd.DataFrame({
        "run_id": ["run-a", "run-a", "run-b"],
        "window_index": [0, 1, 0],
        "sinr_db": [np.nan, 7.0, np.nan],
        "cqi": [4.0, np.nan, np.nan],
    })
    result = causal_impute(frame, ("sinr_db", "cqi"), pd.Series({"sinr_db": 2.0, "cqi": 3.0}))
    assert result.sinr_db.tolist() == [2.0, 7.0, 2.0]
    assert result.cqi.tolist() == [4.0, 4.0, 3.0]


def test_live_xapp_unwraps_and_validates_detector_artifact(tmp_path: Path) -> None:
    detector = DummyClassifier(strategy="prior").fit(
        np.arange(72, dtype=float).reshape(6, 12), np.arange(6)
    )
    artifact = tmp_path / "detector.joblib"
    joblib.dump({"model": detector, "feature_names": list(DETECTOR_FEATURES),
                 "class_names": list(JAMMER_CLASSES)}, artifact)
    loaded = _load_detector(artifact)
    assert loaded is not None
    assert tuple(loaded.classes_) == tuple(range(6))


def test_live_detector_feature_order_matches_xapp_contract() -> None:
    assert tuple(FEATURES) == tuple(DETECTOR_FEATURES)
