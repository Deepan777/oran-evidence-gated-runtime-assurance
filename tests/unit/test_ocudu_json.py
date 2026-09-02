from __future__ import annotations

import pytest

from src.xapp.telemetry.ocudu_json import flatten_numeric, parse_ocudu_json_line


def test_parse_metric_envelope_and_epoch_milliseconds() -> None:
    envelope = parse_ocudu_json_line(
        '{"ts":1786625000000,"type":"scheduler","metrics":{"ue":{"cqi":8,"active":true}}}'
    )
    assert envelope is not None
    assert envelope.timestamp_ns == 1_786_625_000_000_000_000
    assert envelope.metric_type == "scheduler"
    assert flatten_numeric(envelope.payload) == {"ue.cqi": 8.0}


def test_non_json_log_is_ignored() -> None:
    assert parse_ocudu_json_line("2026-08-13 INFO Cell activated") is None


def test_json_without_required_envelope_is_rejected() -> None:
    with pytest.raises(ValueError, match="lacks"):
        parse_ocudu_json_line('{"hello":"world"}')


def test_native_ocudu_metric_envelope_is_parsed() -> None:
    envelope = parse_ocudu_json_line(
        '{"cells":[{"ue_list":[{"cqi":15,"dl_brate":16000.0}]}],'
        '"timestamp":"2026-08-13T13:28:52.909"}'
    )
    assert envelope is not None
    assert envelope.metric_type == "cells"
    assert envelope.timestamp_ns == 1_786_627_732_909_000_000
    assert envelope.payload["cells"][0]["ue_list"][0]["cqi"] == 15
