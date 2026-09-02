from __future__ import annotations

import pytest

from src.xapp.telemetry.windowing import JammerTruthInterval, MetricSample, WindowAggregator


def test_nonoverlapping_window_statistics_and_truth_occupancy() -> None:
    aggregator = WindowAggregator(("cqi", "bler"), origin_ns=1_000_000_000)
    aggregator.add_sample(MetricSample(1_100_000_000, {"cqi": 10.0, "bler": 0.1}))
    aggregator.add_sample(MetricSample(1_900_000_000, {"cqi": 6.0, "bler": 0.5}))
    aggregator.add_truth(JammerTruthInterval(
        1_500_000_000, 2_000_000_000, "partial_band", -5.0, -4.9, True, 0.2, 0.4
    ))

    windows = aggregator.close_through(2_000_000_000)

    assert len(windows) == 1
    window = windows[0]
    assert window.start_ns == 1_000_000_000
    assert window.end_ns == 2_000_000_000
    assert window.metrics["cqi"]["mean"] == pytest.approx(8.0)
    assert window.metrics["cqi"]["last"] == pytest.approx(6.0)
    assert window.metrics["bler"]["max"] == pytest.approx(0.5)
    assert window.missing_metrics == ()
    assert window.jammer_occupancy_fraction == pytest.approx(0.5)
    assert window.jammer_type == "partial_band"
    assert window.realized_jsr_db == pytest.approx(-4.9)


def test_missing_live_metric_is_declared_not_imputed() -> None:
    aggregator = WindowAggregator(("cqi", "rsrp"))
    aggregator.add_sample(MetricSample(100, {"cqi": 7.0}))
    window = aggregator.close_through(1_000_000_000)[0]
    assert window.missing_metrics == ("rsrp",)
    assert "rsrp" not in window.metrics


def test_rejects_late_measurement_after_window_close() -> None:
    aggregator = WindowAggregator(("cqi",))
    aggregator.close_through(1_000_000_000)
    with pytest.raises(ValueError, match="already closed"):
        aggregator.add_sample(MetricSample(999_999_999, {"cqi": 1.0}))

