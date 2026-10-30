"""UNIT-ONLY synthetic fixtures; none are experimental observations or results."""
from __future__ import annotations

import json

import pytest

from analysis.corrected_guard_integration_analysis import (
    ARMS, build_report, paired_estimate, service_outcomes, sha256,
    summarize_decisions, summarize_pairs,
)

UNIT_ROLE = "UNIT_TEST_ONLY_NOT_RESEARCH_DATA"


def execution(time_ns: int, action: str="NO_OP", *, uncertain: bool=False) -> dict:
    """Artificial clock values test sequencing only; no timing result is generated."""
    return {
        "fixture_role":UNIT_ROLE,"requested_action":action,
        "requested_parameters":{} if action=="NO_OP" else {"mcs_cap":6},
        "command_sent":True,"command_outcome":"UNKNOWN" if uncertain else "ACKNOWLEDGED",
        "acknowledged":not uncertain,"action_match":None if uncertain else True,
        "parameter_match":None if uncertain else True,
        "configuration_state_status":"UNKNOWN" if uncertain else "VERIFIED",
        "configuration_state_action":None if uncertain else action,
        "configuration_state_parameters":{} if action=="NO_OP" or uncertain else {"mcs_cap":6},
        "restoration_requested":action=="NO_OP","restoration_acknowledged":action=="NO_OP" and not uncertain,
        "restoration_verified":action=="NO_OP" and not uncertain,
        "started_ns":time_ns,"completed_ns":time_ns+10,
        "readback":None if uncertain else {"successful":True,"verification_scope":"scheduler_software_configuration"},
    }


def record(index: int, *, allowed: bool, event: dict) -> dict:
    return {"fixture_role":UNIT_ROLE,"window_index":index,"status":"DECISION",
            "safety_guard":{"allow_policy":allowed,"observation_eligible":True,"reason":"unit_predicate"},
            "decision":{"policy_selected_action":"MCS_BACKOFF","selected_action":"MCS_BACKOFF" if allowed else "NO_OP",
                        "execution":event,"latency_ms":{"controller_inference_latency":.1}},
            "full_loop_processing_ms":.2}


def test_distinct_command_states_restoration_and_resumption_are_counted():
    rows=[record(0,allowed=False,event=execution(20)),
          record(1,allowed=True,event=execution(100,"MCS_BACKOFF",uncertain=True)),
          record(2,allowed=False,event=execution(220)),
          record(3,allowed=True,event=execution(300,"MCS_BACKOFF"))]
    rows[0]["baseline_reconciliation"]=execution(0)
    rows[2]["baseline_reconciliation"]=execution(200)
    result=summarize_decisions(rows,expected_windows=4)
    counts=result["counts"]
    assert counts["policy_proposals"]==4
    assert counts["commands_sent"]==6  # Two separate baseline commands retained.
    assert counts["operationally_supported_nontrivial_proposals"]==2
    assert counts["intercepted_nontrivial_proposals"]==2
    assert counts["commands_with_unknown_outcome"]==1
    assert counts["post_baseline_unknown_entries"]==1
    assert counts["verified_nontrivial_resumptions_after_restoration"]==1
    assert result["unknown_intervals"][0]["observed_duration_ms"]==pytest.approx(.0001)
    assert result["final_configuration_state_status"]=="VERIFIED"


def test_no_inference_record_is_unavailable_not_a0_proposal():
    rows=[{"fixture_role":UNIT_ROLE,"window_index":0,"status":"STATE_UNVERIFIED_NO_INFERENCE",
           "previous_action_state_status":"UNKNOWN","policy_proposal":None}]
    result=summarize_decisions(rows,expected_windows=1)
    assert result["counts"]["proposal_unavailable_windows"]==1
    assert result["counts"].get("policy_proposals",0)==0
    assert result["policy_proposal_action_counts"]=={}
    assert result["counts"]["windows_with_observed_unknown_state"]==1


def test_duplicate_delivery_is_an_evidence_error_not_more_replication():
    row=record(0,allowed=False,event=execution(0))
    with pytest.raises(ValueError,match="duplicate decision"):
        summarize_decisions([row,row],expected_windows=2)


def test_parameter_mismatch_cannot_be_hidden_by_verified_status():
    event=execution(0,"MCS_BACKOFF")
    event["parameter_match"]=False
    event["configuration_state_parameters"]={"mcs_cap":5}
    result=summarize_decisions([record(0,allowed=True,event=event)],expected_windows=1)
    assert result["counts"]["parameter_mismatch_executions"]==1
    assert result["counts"]["verified_request_parameter_disagreements"]==1


def test_service_uses_frozen_radio_windows_and_retains_observed_outage():
    phases={"pre":1,"interference":2,"recovery":1}
    rows=[{"fixture_role":UNIT_ROLE,"window_index":i,"protected_throughput_mbps":v,
           "protected_lost_percent":None,"protected_jitter_ms":None} for i,v in enumerate([2.,1.,0.,1.])]
    result=service_outcomes(rows,phases)
    assert result["exposure_goodput_mbps"]==.5
    assert result["exposure_goodput_retention"]==.25
    assert result["exposure_zero_goodput_windows"]==1
    assert result["exposure_loss_percent"]["n_missing_or_nonfinite"]==2
    rows[2]["protected_throughput_mbps"]=float("nan")
    with pytest.raises(ValueError,match="cannot be silently"):
        service_outcomes(rows,phases)


def test_paired_bootstrap_uses_blocks_and_exposes_insufficient_replication():
    one=paired_estimate([2.],[1.],iterations=10)
    assert one["n_matched_blocks"]==1 and one["paired_block_bootstrap_95ci"] is None
    two=paired_estimate([2.,4.],[1.,3.],iterations=100)
    assert two["n_matched_blocks"]==2
    assert two["paired_block_bootstrap_95ci"]==[1.,1.]
    runs=[]
    for block in ("unit-b1","unit-b2"):
        for arm in ARMS:
            runs.append({"fixture_role":UNIT_ROLE,"block":block,"arm":arm,"measurement_and_identity_verified":True,
                         "service":{"exposure_goodput_mbps":2. if arm=="guarded_policy" else 1.,"exposure_goodput_retention":1.}})
    paired=summarize_pairs(list(reversed(runs)),["unit-b1","unit-b2"],iterations=100)
    assert paired["complete_four_arm_blocks"]==["unit-b1","unit-b2"]
    assert paired["comparisons"][0]["mean_difference"]==1.
    runs.pop()  # Missing one arm means the entire block is unavailable to four-arm inference.
    paired=summarize_pairs(runs,["unit-b1","unit-b2"],iterations=100)
    assert len(paired["complete_four_arm_blocks"])==1
    assert paired["comparisons"][0]["paired_block_bootstrap_95ci"] is None


def test_unexecuted_frozen_rows_remain_missing_in_report(tmp_path):
    study=tmp_path/"unit_only_study"; study.mkdir()
    freeze={"fixture_role":UNIT_ROLE,"source_sha256":{},"phases_seconds":{"pre":1,"interference":1,"recovery":1},
            "rows":[["unit-b1","none",0,"low",0.,1,"guarded_policy"]]}
    (study/"evaluation_freeze.json").write_text(json.dumps(freeze))
    report=build_report(study,tmp_path/"unit_only_raw",iterations=100)
    assert report["completeness"]=="INCOMPLETE_PRESERVED"
    assert report["population"]["recorded_runs"]==0
    assert report["runs"][0]["status"]=="NOT_RECORDED"


def test_changed_attestation_is_not_accepted_even_when_service_is_complete(tmp_path):
    study=tmp_path/"unit_only_study"; study.mkdir()
    raw=tmp_path/"unit_only_raw"; evidence=raw/"iq-broker-live-unit-only"; evidence.mkdir(parents=True)
    freeze={"fixture_role":UNIT_ROLE,"source_sha256":{"unit.py":"not_a_research_hash"},
            "phases_seconds":{"pre":1,"interference":1,"recovery":1},
            "rows":[["unit-b1","none",0,"low",0.,1,"no_action"]]}
    freeze_path=study/"evaluation_freeze.json"; freeze_path.write_text(json.dumps(freeze))
    row={"fixture_role":UNIT_ROLE,"block":"unit-b1","seed":1,"arm":"no_action","run_id":"unit-only",
         "exit_code":0,"measurements_complete":True,"source_sha256":{},"freeze_sha256":sha256(freeze_path)}
    (study/"evaluation_progress.json").write_text(json.dumps({"fixture_role":UNIT_ROLE,"completed":[row]}))
    (evidence/"corrected_guard_attestation.json").write_text(json.dumps(row))
    windows=[{"fixture_role":UNIT_ROLE,"window_index":i,"protected_throughput_mbps":1.} for i in range(3)]
    (evidence/"protected-traffic-status.json").write_text(json.dumps({"fixture_role":UNIT_ROLE,"complete_radio_windows":3,"radio_windows":windows}))
    report=build_report(study,raw,iterations=100)
    assert report["runs"][0]["source_identity_verified"] is False
    assert report["population"]["measurement_identity_verified_runs"]==0
    assert report["runs"][0]["service"]["exposure_goodput_mbps"]==1.  # Retained, not accepted.
