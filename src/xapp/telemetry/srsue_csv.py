"""Parser for the pinned srsUE receiver-side 1 s metric stream."""
from __future__ import annotations

import csv
from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class SrsUeMetricSample:
    timestamp_ns: int
    values: dict[str, float]


REQUIRED_COLUMNS = (
    "timestamp_epoch_ms",
    "rsrp",
    "rsrq",
    "rssi",
    "dl_snr",
    "dl_brate",
    "dl_bler",
    "ul_buff",
)


def parse_srsue_csv(header: str, row: str, *, allow_unbounded_sinr: bool = False) -> SrsUeMetricSample:
    """Parse one data row without silently coercing ``n/a`` measurements."""
    columns = next(csv.reader([header], delimiter=";"))
    values = next(csv.reader([row], delimiter=";"))
    if len(values) != len(columns):
        raise ValueError("srsUE CSV row width does not match its header")
    document = dict(zip(columns, values, strict=True))
    missing = tuple(column for column in REQUIRED_COLUMNS if column not in document)
    if missing:
        raise ValueError(f"srsUE CSV lacks required columns: {missing}")

    def finite(name: str) -> float:
        try:
            value = float(document[name])
        except ValueError as error:
            raise ValueError(f"srsUE metric {name} is not numeric") from error
        if not isfinite(value):
            raise ValueError(f"srsUE metric {name} is not finite")
        return value

    timestamp_ns = round(finite("timestamp_epoch_ms") * 1_000_000.0)
    if timestamp_ns < 0:
        raise ValueError("srsUE timestamp must be non-negative")
    attached = finite("is_attached") if "is_attached" in document else 1.0
    native_sinr = float(document["dl_snr"])
    unbounded_sinr = allow_unbounded_sinr and native_sinr == float("inf") and attached == 1.0
    return SrsUeMetricSample(
        timestamp_ns=timestamp_ns,
        values={
            # Positive infinity is an explicitly observed, unbounded native
            # estimate, not a finite SNR value. Never cap it or call a policy
            # substitution measured. This option is prospective and opt-in.
            **({"sinr_positive_infinity_observed": 1.0} if unbounded_sinr else {"sinr_db": finite("dl_snr")}),
            # The ZMQ software receiver has no calibrated RF front-end. Its
            # native signal level is therefore dBFS and must not be called dBm.
            "rsrp_dbfs": finite("rsrp"),
            "rsrq_db": finite("rsrq"),
            "rssi_dbfs": finite("rssi"),
            "ue_dl_throughput_mbps": finite("dl_brate") / 1_000_000.0,
            "ue_dl_bler": finite("dl_bler") / 100.0,
            "ue_ul_backlog_bytes": finite("ul_buff"),
            # The pinned stream exposes the native attachment state.  Keep it
            # alongside a finite row so consumers can distinguish a valid RF
            # observation from finite placeholder values emitted after the
            # receiver has detached.  Legacy fixture streams without this
            # optional column are attached by construction.
            "receiver_attached": attached,
        },
    )
