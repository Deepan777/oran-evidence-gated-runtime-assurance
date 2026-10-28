from __future__ import annotations

import json
from io import BytesIO

from src.controller.action_mask import Action
from src.xapp.adapters.http_backend import HttpHookConfig, HttpResearchHookBackend


class FakeResponse:
    status = 200

    def __init__(self, payload: object) -> None:
        self.payload = payload

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


def test_http_backend_preserves_reported_action_and_parameters(monkeypatch: object) -> None:
    captured: dict[str, object] = {}

    def fake_open(request: object, timeout: float) -> FakeResponse:
        captured["request"] = request
        captured["timeout"] = timeout
        return FakeResponse({
            "status": "applied",
            "action": "MCS_BACKOFF",
            "message": "ok",
            "applied_parameters": {"backoff_levels": 4},
        })

    monkeypatch.setattr("src.xapp.adapters.http_backend.urlopen", fake_open)  # type: ignore[attr-defined]
    backend = HttpResearchHookBackend(HttpHookConfig("http://127.0.0.1:9060", "secret"))
    result = backend.apply(Action.MCS_BACKOFF, {"backoff_levels": 4})
    request = captured["request"]
    assert result.accepted
    assert result.reported_action is Action.MCS_BACKOFF
    assert result.reported_parameters == {"backoff_levels": 4}
    assert request.get_header("Authorization") == "Bearer secret"  # type: ignore[union-attr]


def test_http_backend_exposes_mismatched_ack_for_adapter_integrity_check(monkeypatch: object) -> None:
    monkeypatch.setattr(  # type: ignore[attr-defined]
        "src.xapp.adapters.http_backend.urlopen",
        lambda request, timeout: FakeResponse({"status": "applied", "action": "NO_OP"}),
    )
    backend = HttpResearchHookBackend(HttpHookConfig("http://127.0.0.1:9060", "secret"))
    result = backend.apply(Action.PRB_AVOID_SHIFT, {})
    assert result.accepted
    assert result.reported_action is Action.NO_OP
    assert result.reported_parameters == {}

