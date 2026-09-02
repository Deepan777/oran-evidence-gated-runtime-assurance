from analysis.cee_v10_safety_diagnostic import _phase, _quantiles


def test_frozen_phase_boundaries() -> None:
    assert _phase(0) == "pre_jam"
    assert _phase(19) == "pre_jam"
    assert _phase(20) == "jammer_on"
    assert _phase(99) == "jammer_on"
    assert _phase(100) == "recovery"
    assert _phase(119) == "recovery"


def test_quantiles_are_finite_and_auditable() -> None:
    summary = _quantiles([0.0, 0.5, 1.0])
    assert summary["n"] == 3
    assert summary["p50"] == 0.5
