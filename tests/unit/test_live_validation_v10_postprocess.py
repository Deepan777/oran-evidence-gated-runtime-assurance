from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

import analysis.live_validation_v10_postprocess as postprocess
from analysis.live_validation_v10_postprocess import (
    METHODS,
    derive_run_outcomes,
    statistics_input,
    traceability_audit,
)
from analysis.live_validation_v10_publication import _manuscript


def _radio_run() -> pd.DataFrame:
    throughput = np.concatenate((np.full(20, 10.0), np.full(80, 8.0), np.full(20, 9.5)))
    delay = np.concatenate(
        (np.full(20, 5.0), np.r_[np.full(40, 25.0), np.full(40, 10.0)], np.full(20, 10.0))
    )
    return pd.DataFrame({
        "window_index": range(120),
        "phase": ["pre_jam"] * 20 + ["jammer_on"] * 80 + ["recovery"] * 20,
        "dl_throughput_mbps": throughput,
        "rlc_sdu_delay_ms": delay,
        "rlc_drop_rate": np.concatenate((np.zeros(20), np.full(80, 0.1), np.zeros(20))),
        "telemetry_observability": ["fully_measured"] * 120,
    })


def test_v10_frozen_run_outcome_formulas() -> None:
    outcomes = derive_run_outcomes(_radio_run())
    assert outcomes["protected_flow_latency_violation_rate"] == 0.5
    assert outcomes["protected_flow_throughput_retention"] == 0.8
    assert outcomes["protected_flow_packet_or_rlc_drop_rate"] == pytest.approx(0.1)
    assert outcomes["recovery_time"] == 0.0


def test_v10_detached_outage_is_violation_even_with_measured_gnb_delay() -> None:
    frame = _radio_run()
    outage = frame.index[(frame.phase == "jammer_on")][-8:]
    frame.loc[outage, "rlc_sdu_delay_ms"] = 2.0
    frame.loc[outage, "dl_throughput_mbps"] = 0.0
    frame.loc[outage, "rlc_drop_rate"] = 1.0
    frame.loc[outage, "telemetry_observability"] = "receiver_detached_outage"

    outcomes = derive_run_outcomes(frame)

    assert outcomes["protected_flow_latency_violation_rate"] == 0.60
    assert (frame.loc[outage, "rlc_sdu_delay_ms"] == 2.0).all()


def test_v10_statistics_input_uses_matched_blocks_not_windows() -> None:
    rows = []
    for block in range(90):
        for method_index, method in enumerate(METHODS):
            rows.append({
                "source_run_id": f"block-{block:03d}",
                "method": method,
                "protected_flow_latency_violation_rate": 0.1 + method_index,
                "protected_flow_throughput_retention": 1.0 - method_index * 0.1,
                "protected_flow_packet_or_rlc_drop_rate": 0.01 + method_index * 0.01,
                "recovery_time": float(method_index),
            })

    payload = statistics_input(pd.DataFrame(rows))

    assert payload["replication_unit"] == "source_run_id"
    for outcome in payload["outcomes"].values():
        assert outcome["methods"] == list(METHODS)
        assert len(outcome["matched_blocks"]) == 90
        assert len(outcome["matched_blocks"][0]) == 4


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _lower_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _traceability_fixture(root: Path) -> tuple[Path, dict[str, object]]:
    versioned_artifacts = {
        "b0_v10_manifest": "manifests/B0.csv",
        "b2_v10_manifest": "manifests/B2.csv",
        "b5_v10_manifest": "manifests/B5.csv",
        "p_v10_manifest": "manifests/P.csv",
        "v10_analysis_plan": "LIVE_VALIDATION_V10_ANALYSIS_PLAN.yaml",
    }
    artifacts: dict[str, object] = {}
    for key, relative in versioned_artifacts.items():
        path = root / relative
        if key == "v10_analysis_plan":
            _write(path, yaml.safe_dump({
                "status": "FROZEN_BEFORE_ANY_V10_OUTCOME",
                "campaign": "LIVE_VALIDATION_V10",
                "methods": ["P", "B0", "B2", "B5"],
            }))
        else:
            _write(path, f"{key}\n")
        artifacts[key] = {"path": relative, "sha256": _lower_sha256(path)}

    implementation_relative = "experiments/collection/build_live_run_records.py"
    implementation = root / implementation_relative
    _write(implementation, "# frozen detached-outage implementation\n")
    artifacts["v10_detached_outage_implementation"] = {
        "paths": [implementation_relative],
        "sha256": {implementation_relative: _lower_sha256(implementation)},
    }
    freeze: dict[str, object] = {
        "schema_version": 10,
        "status": "LIVE_VALIDATION_FROZEN_V10",
        "execution_revision": "V10_DETACHED_OUTAGE_EVIDENCE_R1",
        "total_method_executions": 360,
        "scenario_count": 90,
        "runs_per_method": 90,
        "run_length_seconds": 120,
        "methods": ["B0", "B2", "B5", "P"],
        "artifacts": artifacts,
    }
    _write(root / "LIVE_VALIDATION_FREEZE_MANIFEST_V10.yaml", yaml.safe_dump(freeze))
    return implementation, freeze


def test_v10_traceability_accepts_lowercase_frozen_hashes(tmp_path: Path) -> None:
    _traceability_fixture(tmp_path)

    audit = traceability_audit(tmp_path)

    assert audit["schema_version"] == 1
    assert audit["campaign"] == "LIVE_VALIDATION_V10"
    assert audit["status"] == "PASS"
    assert audit["audited_files"] == 6
    assert audit["critical_findings"] == []


def test_v10_traceability_enforces_detached_outage_implementation_hash(tmp_path: Path) -> None:
    implementation, _ = _traceability_fixture(tmp_path)
    implementation.write_text("# mutated after freeze\n", encoding="utf-8")

    audit = traceability_audit(tmp_path)

    assert audit["status"] == "FAIL"
    assert any(
        "V10 implementation hash mismatch" in finding
        and "build_live_run_records.py" in finding
        for finding in audit["critical_findings"]
    )


def test_v10_traceability_rejects_missing_strict_implementation_key(tmp_path: Path) -> None:
    _, freeze = _traceability_fixture(tmp_path)
    artifacts = freeze["artifacts"]
    assert isinstance(artifacts, dict)
    implementation = artifacts.pop("v10_detached_outage_implementation")
    artifacts["v10_native_receiver_attachment_implementation"] = implementation
    _write(tmp_path / "LIVE_VALIDATION_FREEZE_MANIFEST_V10.yaml", yaml.safe_dump(freeze))

    audit = traceability_audit(tmp_path)

    assert audit["status"] == "FAIL"
    assert any(
        "detached-outage implementation traceability" in finding
        for finding in audit["critical_findings"]
    )


def _run_stubbed_main(monkeypatch: pytest.MonkeyPatch, root: Path, gate_status: str) -> None:
    combined = pd.DataFrame({"window_index": [0]})
    per_run = pd.DataFrame([{
        "run_id": "livev10-p-test",
        "method": "P",
        "seed": 1001,
        "start_time_ns": 1,
        "end_time_ns": 2,
        "raw_result_path": "results/raw/test",
        "scenario_definition": "{}",
        "frozen_provenance_id": "LIVE_VALIDATION_V10_CAMPAIGN_PROVENANCE",
    }])
    monkeypatch.setattr(
        postprocess,
        "load_strict_campaign",
        lambda _: (combined, per_run, [], []),
    )
    monkeypatch.setattr(postprocess, "statistics_input", lambda _: {"schema_version": 1})
    monkeypatch.setattr(postprocess, "build_statistics_report", lambda _: {"outcomes": {}})
    monkeypatch.setattr(postprocess, "supported_outcomes", lambda _: ([], {}))
    monkeypatch.setattr(postprocess, "xai_summary", lambda *_: {
        "median_faithfulness_ratio": 2.0,
        "median_stability_spearman": 0.9,
        "median_stability_top_k_jaccard": 0.7,
        "counterfactual_valid_fraction": 0.9,
        "p95_online_control_loop_latency_ms": 10.0,
    })
    monkeypatch.setattr(postprocess, "robustness_summary", lambda *_: {
        "no_major_unexplained_no_jam_regression": True,
        "benefit_across_multiple_jammer_conditions": True,
        "conclusion_reproduces_across_seeds": True,
    })
    monkeypatch.setattr(
        postprocess,
        "traceability_audit",
        lambda _: {"schema_version": 1, "status": "PASS", "critical_findings": []},
    )
    monkeypatch.setattr(postprocess, "evaluate_evidence", lambda _: {
        "status": gate_status,
        "reasons": [] if gate_status == "PASS" else ["fixture failure"],
        "supported_primary_resilience_outcomes": [],
        "manuscript_allowed": gate_status == "PASS",
    })
    monkeypatch.setattr(postprocess, "render_results", lambda *_: "# V10 fixture results\n")
    monkeypatch.setattr(postprocess, "generate_publication_artifacts", lambda _: None)

    def fake_parquet(frame: pd.DataFrame, path: Path, **_: object) -> None:
        del frame
        _write(Path(path), "parquet fixture")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", fake_parquet)
    monkeypatch.setattr(sys, "argv", ["live_validation_v10_postprocess.py", "--root", str(root)])
    assert postprocess.main() == 0


def test_v10_postprocess_fail_outputs_are_fully_versioned(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _run_stubbed_main(monkeypatch, tmp_path, "FAIL")

    expected = (
        "results/aggregate/live_validation_v10_statistics_input.json",
        "results/aggregate/live_validation_v10_evidence_summary.json",
        "results/evidence_gate/live_validation_v10_evidence_gate.json",
        "results/evidence_gate/LIVE_VALIDATION_V10_EVIDENCE_GATE_REPORT.md",
        "results/LIVE_VALIDATION_V10_FAILURE_ANALYSIS.md",
    )
    for relative in expected:
        assert (tmp_path / relative).is_file(), relative
    for legacy in (
        "results/aggregate/statistics_input.json",
        "results/aggregate/evidence_summary.json",
        "results/evidence_gate.json",
        "results/evidence_gate/EVIDENCE_GATE_REPORT.md",
        "results/failure_analysis.md",
    ):
        assert not (tmp_path / legacy).exists(), legacy


def test_v10_postprocess_pass_writes_versioned_manuscript_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _run_stubbed_main(monkeypatch, tmp_path, "PASS")

    assert (tmp_path / "manuscript/LIVE_VALIDATION_V10_RESULTS.md").is_file()
    assert not (tmp_path / "manuscript/LIVE_VALIDATION_V10_FAILURE_ANALYSIS.md").exists()


def test_v10_publication_writes_versioned_manuscript_and_claim_map(tmp_path: Path) -> None:
    evidence = {
        "supported_primary_resilience_outcomes": [
            "protected_flow_latency_violation_rate",
            "protected_flow_throughput_retention",
        ]
    }

    _manuscript(
        tmp_path,
        {"status": "PASS", "manuscript_allowed": True},
        evidence,
    )

    manuscript = tmp_path / "manuscript/LIVE_VALIDATION_V10_MANUSCRIPT.md"
    claim_map = tmp_path / "manuscript/LIVE_VALIDATION_V10_CLAIM_EVIDENCE_MAP.csv"
    assert manuscript.is_file()
    assert claim_map.is_file()
    assert "frozen V10 protocol" in manuscript.read_text(encoding="utf-8")
    assert "live_validation_v10_stats_report.json" in claim_map.read_text(encoding="utf-8")
    producer = manuscript.with_suffix(manuscript.suffix + ".producer.json")
    assert json.loads(producer.read_text(encoding="utf-8"))["producer"] == (
        "analysis/live_validation_v10_publication.py"
    )
    assert not (tmp_path / "manuscript/manuscript.md").exists()
    assert not (tmp_path / "manuscript/claim_evidence_map.csv").exists()
