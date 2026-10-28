import pytest
from analysis.recovery_censoring_correction import paired_binary
from analysis.actuator_recovery_censoring_correction import derive


def test_exact_mcnemar_uses_only_discordant_pairs():
    result=paired_binary([0]*90,[1]*23+[0]*67)
    assert result["paired_counts"]=={"both_recover":0,"left_only":0,"right_only":23,"neither_recovers":67}
    assert result["p_value"]==pytest.approx(2/(2**23))


def test_actuator_non_event_is_censored_at_five_not_six_seconds():
    rows=[{"protected_throughput_mbps":1 if i<10 else .1} for i in range(30)]
    outcome=derive(rows)
    assert outcome["event_time_seconds"] is None
    assert outcome["censor_time_seconds"]==5
    assert outcome["restricted_time_to_recovery_5s"]==5
    rows[29]["protected_throughput_mbps"]=.9
    outcome=derive(rows)
    assert outcome["event_time_seconds"]==5
    assert outcome["censor_time_seconds"] is None
