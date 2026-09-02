from __future__ import annotations

from experiments.generate_live_calibration_recovery_manifest import make_rows as recovery_rows
from experiments.generate_live_calibration_recovery_v3_manifest import make_combined_rows, make_retry_rows
from experiments.generate_live_calibration_recovery_v4_manifest import make_combined_rows as make_v4_combined_rows, make_retry_rows as make_v4_retry_rows
from experiments.generate_live_calibration_recovery_v6_manifest import make_combined_rows as make_v6_combined_rows, make_retry_rows as make_v6_retry_rows
from experiments.generate_live_calibration_recovery_v7_manifest import make_combined_rows as make_v7_combined_rows, make_retry_rows as make_v7_retry_rows
from experiments.generate_live_calibration_recovery_v8_manifest import make_combined_rows as make_v8_combined_rows, make_retry_rows as make_v8_retry_rows
from experiments.generate_live_calibration_recovery_v9_manifest import make_combined_rows as make_v9_combined_rows, make_retry_rows as make_v9_retry_rows
from experiments.generate_live_calibration_recovery_v10_manifest import make_combined_rows as make_v10_combined_rows, make_retry_rows as make_v10_retry_rows
from experiments.generate_live_exploration_manifest import make_rows as exploration_rows
from experiments.generate_live_ppo_exploration_manifest import make_rows as ppo_exploration_rows


def test_recovery_calibration_is_separate_from_failed_v1_and_balanced() -> None:
    rows = recovery_rows()
    assert len(rows) == 30
    assert {row["evidence_class"] for row in rows} == {"development_live_calibration"}
    assert all(str(row["run_id"]).startswith("livecalv2-") for row in rows)
    for jammer in {str(row["jammer_type"]) for row in rows}:
        family = [row for row in rows if row["jammer_type"] == jammer]
        assert [row["partition"] for row in family] == ["train", "train", "train", "validation", "locked_test"]
        if jammer != "none":
            assert {row["jsr_db"] for row in family} == {0, 5}


def test_exploration_training_partition_covers_all_locked_actions() -> None:
    rows = exploration_rows()
    assert len(rows) == 33
    assert {row["evidence_class"] for row in rows} == {"development_live_exploration"}
    train_actions = {int(row["LIVE_XAPP_FORCED_ACTION_ID"]) for row in rows if row["partition"] == "train"}
    assert train_actions == {0, 1, 2, 3, 4}
    localized = [row for row in rows if int(row["LIVE_XAPP_FORCED_ACTION_ID"]) in {1, 4}]
    assert {row["jammer_type"] for row in localized} <= {"partial_band", "sweep"}
    assert {row["traffic_load"] for row in rows} == {"low"}
    assert {row["service_mix"] for row in rows} == {"protected_only"}
    assert {row["attenuation"] for row in rows} == {"nominal"}
    assert {row["jsr_db"] for row in rows if row["jammer_type"] != "none"} == {0, 5}
    prb_only = [row for row in rows if row.get("baseline_training_role") == "B5_PRB_ONLY_A1_COVERAGE"]
    assert len(prb_only) == 3
    assert all(row["partition"] == "train" and int(row["LIVE_XAPP_FORCED_ACTION_ID"]) == 1 for row in prb_only)


def test_v3_recovery_preserves_failed_v2_and_restores_balanced_training_manifest() -> None:
    retry, combined = make_retry_rows(), make_combined_rows()
    assert len(retry) == 20
    assert len(combined) == 30
    assert all(str(row["run_id"]).startswith("livecalv3-") for row in retry)
    assert "livecalv2-partial_bandmjsrp0mlowmprotected_onlymnominal-s1301" not in {
        row["run_id"] for row in combined
    }
    for jammer in {str(row["jammer_type"]) for row in combined}:
        family = [row for row in combined if row["jammer_type"] == jammer]
        assert [row["partition"] for row in family] == ["train", "train", "train", "validation", "locked_test"]


def test_ppo_exploration_is_separate_and_on_policy_only() -> None:
    rows = ppo_exploration_rows()
    assert len(rows) == 24
    assert {row["evidence_class"] for row in rows} == {"development_live_ppo_exploration"}
    assert {row["LIVE_XAPP_SELECTION_MODE"] for row in rows} == {"ppo_sampling"}
    assert {row["CONTROLLER_KIND"] for row in rows} == {"masked_ppo"}
    for jammer in {str(row["jammer_type"]) for row in rows}:
        family = [row for row in rows if row["jammer_type"] == jammer]
        assert [row["partition"] for row in family] == ["train", "train", "train", "validation"]


def test_v4_recovery_preserves_complete_v3_rows_and_replaces_the_failed_sequence() -> None:
    retry, combined = make_v4_retry_rows(), make_v4_combined_rows()
    assert len(retry) == 18
    assert len(combined) == 30
    assert all(str(row["run_id"]).startswith("livecalv4-") for row in retry)
    failed = [row for row in retry if row.get("recovery_of_run_id")]
    assert len(failed) == 1
    assert failed[0]["recovery_of_run_id"] == "livecalv3-partial_bandmjsrp0mlowmprotected_onlymnominal-s1303"


def test_v6_recovery_replaces_uncompleted_v5_rows_with_fresh_ids() -> None:
    retry, combined = make_v6_retry_rows(), make_v6_combined_rows()
    assert len(retry) == 16
    assert len(combined) == 30
    assert all(str(row["run_id"]).startswith("livecalv6-") for row in retry)
    assert all(row["recovery_protocol"] == "v6_bounded_uplink_deadline_ue_timeout_500ms" for row in retry)
    assert all(row.get("recovery_of_or_supersedes_v5_run_id") for row in retry)


def test_v7_recovery_records_the_cpu_capacity_correction() -> None:
    retry, combined = make_v7_retry_rows(), make_v7_combined_rows()
    assert len(retry) == 16
    assert len(combined) == 30
    assert all(str(row["run_id"]).startswith("livecalv7-") for row in retry)
    assert all(row["recovery_protocol"] == "v7_e2_standard_8_cpu_capacity" for row in retry)
    assert all(row.get("recovery_of_or_supersedes_v6_run_id") for row in retry)


def test_v8_recovery_records_ordered_core_restart() -> None:
    retry, combined = make_v8_retry_rows(), make_v8_combined_rows()
    assert len(retry) == 16
    assert len(combined) == 30
    assert all(str(row["run_id"]).startswith("livecalv8-") for row in retry)
    assert all(row["recovery_protocol"] == "v8_ordered_mongodb_open5gs_restart" for row in retry)
    assert all(row.get("recovery_of_or_supersedes_v7_run_id") for row in retry)


def test_v9_recovery_keeps_six_v8_rows_and_replaces_only_the_tail() -> None:
    retry, combined = make_v9_retry_rows(), make_v9_combined_rows()
    assert len(retry) == 10
    assert len(combined) == 30
    assert all(str(row["run_id"]).startswith("livecalv9-") for row in retry)
    assert all(row["recovery_protocol"] == "v9_ue_timeout_1000ms_intermittent_tail" for row in retry)


def test_v10_recovery_retries_only_unaccepted_v9_tail_with_bounded_timeout() -> None:
    retry, combined = make_v10_retry_rows(), make_v10_combined_rows()
    assert len(retry) == 9
    assert len(combined) == 30
    assert all(str(row["run_id"]).startswith("livecalv10-") for row in retry)
    assert all(row["recovery_protocol"] == "v10_ue_timeout_5000ms_intermittent_tail" for row in retry)
    assert all(row["UE_TIMEOUT_MS"] == "5000" for row in retry)
    assert "livecalv9-intermittentmjsrp0mlowmprotected_onlymnominal-s1301" in {str(row["run_id"]) for row in combined}
    assert all(row.get("recovery_of_or_supersedes_v8_run_id") for row in retry)
