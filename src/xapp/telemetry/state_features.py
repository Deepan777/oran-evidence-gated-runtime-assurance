"""Truthful derivation of locked controller features from live radio data."""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any, Mapping


@dataclass(frozen=True)
class OcuduStateObservation:
    timestamp_ns: int
    values: Mapping[str, float]
    ue_id: int
    rnti: int


def state_observation_from_ocudu(
    timestamp_ns: int,
    payload: Mapping[str, Any],
    *,
    total_prbs: int = 106,
) -> OcuduStateObservation | None:
    """Derive only fields directly supported by one native OCUDU JSON record.

    RLC and cell metrics arrive as separate records, so they are returned by
    :func:`rlc_observation_from_ocudu` and must be joined by timestamp/window.
    """
    cells = payload.get("cells")
    if not isinstance(cells, list) or not cells:
        return None
    cell = cells[0]
    ue_list = cell.get("ue_list", []) if isinstance(cell, dict) else []
    if not ue_list:
        return None
    ue = ue_list[0]
    if not isinstance(ue, dict):
        raise ValueError("OCUDU UE metric is not an object")

    dl_ok = _number(ue, "dl_nof_ok")
    dl_nok = _number(ue, "dl_nof_nok")
    harq_total = dl_ok + dl_nok
    harq_nack = dl_nok / harq_total if harq_total > 0 else 0.0
    cell_metrics = cell.get("cell_metrics", {})
    pucch_rbs = _number(cell_metrics, "pucch_tot_rb_usage_avg") if isinstance(cell_metrics, dict) else 0.0
    values = {
        "cqi": _number(ue, "cqi"),
        "harq_nack_ratio": harq_nack,
        "gnb_dl_throughput_mbps": _number(ue, "dl_brate") / 1_000_000.0,
        "scheduler_queue_backlog_bytes": _number(ue, "bsr") + _number(ue, "dl_bs"),
        "prb_utilization": max(0.0, min(1.0, pucch_rbs / float(total_prbs))),
        "current_dl_mcs": _number(ue, "dl_mcs"),
    }
    return OcuduStateObservation(timestamp_ns, values, int(_number(ue, "ue")), int(_number(ue, "rnti")))


def rlc_observation_from_ocudu(payload: Mapping[str, Any]) -> dict[str, float] | None:
    rlc = payload.get("rlc_metrics")
    if not isinstance(rlc, dict):
        return None
    tx = rlc.get("tx", {})
    if not isinstance(tx, dict):
        raise ValueError("OCUDU RLC tx metrics are not an object")
    sent = _number(tx, "num_sdus")
    dropped = _number(tx, "num_dropped_sdus") + _number(tx, "num_discarded_sdus")
    attempted = sent + dropped
    values = {
        "rlc_drop_rate": dropped / attempted if attempted > 0 else 0.0,
    }
    # OCUDU emits an interval-local ``num_sdus`` but its latency accumulator
    # may retain a value while no SDU was transmitted in that interval.  The
    # ratio would then fabricate an arbitrarily large delay.  Leave the field
    # unavailable; the collection/runtime layer records the missing source and
    # applies only its frozen causal imputation policy, if one is validated.
    if sent > 0:
        values["rlc_sdu_delay_ms"] = _number(tx, "sum_sdu_latency_us") / (sent * 1000.0)
    return values


def _number(mapping: Mapping[str, Any], name: str) -> float:
    value = mapping.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(float(value)):
        raise ValueError(f"OCUDU metric {name} is missing or non-finite")
    return float(value)
