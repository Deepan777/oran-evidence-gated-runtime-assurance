from __future__ import annotations

import numpy as np
import pytest

from analysis.statistics import (
    build_statistics_report,
    bootstrap_ci,
    friedman_test,
    holm_bonferroni,
    planned_wilcoxon,
    rank_biserial_paired,
)


def test_bootstrap_ci_is_deterministic_and_contains_mean() -> None:
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    first = bootstrap_ci(values, iterations=2000, seed=8)
    second = bootstrap_ci(values, iterations=2000, seed=8)
    assert first == second
    assert first[0] <= np.mean(values) <= first[1]


def test_holm_bonferroni_preserves_original_order() -> None:
    result = holm_bonferroni([0.04, 0.001, 0.02])
    assert [row["index"] for row in result] == [0, 1, 2]
    assert result[1]["significant"] is True


def test_paired_effect_and_wilcoxon_report_sample_count() -> None:
    left = [2.0, 3.0, 4.0, 5.0]
    right = [1.0, 2.0, 3.0, 4.0]
    assert rank_biserial_paired(left, right) == pytest.approx(1.0)
    assert planned_wilcoxon(left, right)["n_pairs"] == 4


def test_friedman_requires_matched_methods_and_returns_probability() -> None:
    matrix = np.asarray([[1.0, 2.0, 3.0], [1.5, 2.5, 4.0], [1.2, 2.1, 3.2]])
    result = friedman_test(matrix)
    assert result["statistic"] >= 0.0
    assert 0.0 <= result["p_value"] <= 1.0


def test_statistics_report_computes_declared_outcome_and_holm_adjustment() -> None:
    report = build_statistics_report({"seed": 3, "bootstrap_iterations": 50, "outcomes": {
        "throughput": {"methods": ["M", "B0", "B1"], "matched_blocks": [[3.0, 2.0, 1.0], [4.0, 2.5, 1.5]], "planned_pairs": [["M", "B0"], ["M", "B1"]]}
    }})
    comparisons = report["outcomes"]["throughput"]["planned_comparisons"]
    assert len(comparisons) == 2
    assert "holm" in comparisons[0]
    summary = report["outcomes"]["throughput"]["summaries"]["M"]
    assert summary["median"] == pytest.approx(3.5)
    assert summary["iqr"] == pytest.approx([3.25, 3.75])
    relative = report["outcomes"]["throughput"]["relative_improvements_vs_first_method"]
    assert relative[0]["mean_relative_improvement"] > 0.0
