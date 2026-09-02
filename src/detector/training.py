from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter_ns
from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits


@dataclass(frozen=True)
class DetectorSelection:
    model_name: str
    model: Any
    macro_f1: float
    balanced_accuracy: float
    latency_ms: float


def select_simplest_detector(
    x_train: np.ndarray, y_train: np.ndarray, x_validation: np.ndarray, y_validation: np.ndarray, seed: int
) -> DetectorSelection:
    """Choose the simplest candidate within one F1 percentage point of the best."""
    candidates: list[tuple[str, Any]] = [
        ("logistic_regression", make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=seed))),
        ("random_forest", RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=1)),
        ("gradient_boosted_trees", HistGradientBoostingClassifier(random_state=seed)),
        ("small_mlp", make_pipeline(
            StandardScaler(),
            MLPClassifier(hidden_layer_sizes=(32,), max_iter=500, early_stopping=True, random_state=seed),
        )),
    ]
    evaluated: list[DetectorSelection] = []
    for name, candidate in candidates:
        # Candidate comparison must be reproducible and must not depend on
        # host multiprocessing permissions. A single OpenMP thread also keeps
        # latency measurements comparable across the cloud and local audit.
        with threadpool_limits(limits=1):
            candidate.fit(x_train, y_train)
            start = perf_counter_ns(); prediction = candidate.predict(x_validation)
        latency_ms = (perf_counter_ns() - start) / 1_000_000 / max(1, len(x_validation))
        evaluated.append(DetectorSelection(name, candidate, float(f1_score(y_validation, prediction, average="macro")),
                                           float(balanced_accuracy_score(y_validation, prediction)), latency_ms))
    best = max(item.macro_f1 for item in evaluated)
    return next(item for item in evaluated if item.macro_f1 >= best - 0.01)
