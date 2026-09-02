from __future__ import annotations

import pytest

from src.xapp.telemetry.srsue_csv import parse_srsue_csv


HEADER = "timestamp_epoch_ms;rsrp;rsrq;rssi;dl_snr;dl_brate;dl_bler;ul_buff"


def test_parse_srsue_receiver_metrics_and_units() -> None:
    sample = parse_srsue_csv(HEADER, "1786630000123.456;-63.2;-11.4;-52.0;18.5;2500000;3.5;4096")
    assert sample.timestamp_ns == 1_786_630_000_123_456_000
    assert sample.values["sinr_db"] == pytest.approx(18.5)
    assert sample.values["rsrp_dbfs"] == pytest.approx(-63.2)
    assert sample.values["rsrq_db"] == pytest.approx(-11.4)
    assert sample.values["ue_dl_throughput_mbps"] == pytest.approx(2.5)
    assert sample.values["ue_dl_bler"] == pytest.approx(0.035)
    assert sample.values["receiver_attached"] == 1.0


def test_parse_native_receiver_attachment_flag() -> None:
    header = HEADER + ";is_attached"
    sample = parse_srsue_csv(
        header,
        "1786630000123.456;23;0;0;113;0;0;4096;0",
    )
    assert sample.values["receiver_attached"] == 0.0
    # The strict parser retains the native envelope.  Consumers, rather than
    # the parser, decide that finite detached placeholders are unavailable.
    assert sample.values["sinr_db"] == 113.0


def test_reject_non_numeric_receiver_measurement() -> None:
    with pytest.raises(ValueError, match="rsrq"):
        parse_srsue_csv(HEADER, "1786630000123.456;-63.2;n/a;-52;18;0;0;0")
