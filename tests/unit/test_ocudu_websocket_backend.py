from __future__ import annotations

from src.controller.action_mask import Action
from src.xapp.adapters.ocudu_websocket import OcuduWebSocketResearchBackend


def test_websocket_backend_builds_atomic_a4_document(monkeypatch: object) -> None:
    captured: dict[str, object] = {}

    def fake_send(host: str, port: int, document: object, timeout: float) -> object:
        captured.update(host=host, port=port, document=document, timeout=timeout)
        return {"status": "success", "cmd": "research_action_set"}

    monkeypatch.setattr("src.xapp.adapters.ocudu_websocket.send", fake_send)  # type: ignore[attr-defined]
    backend = OcuduWebSocketResearchBackend()
    result = backend.apply(Action.COMPOSITE_LOCAL_DEFENSE, {"prb_start": 20, "prb_stop": 40, "mcs_cap": 8})
    assert result.accepted
    assert captured["document"] == {
        "cmd": "research_action_set", "action": "A4", "prb_start": 20, "prb_stop": 40, "mcs_cap": 8
    }


def test_websocket_backend_rejects_error(monkeypatch: object) -> None:
    monkeypatch.setattr(
        "src.xapp.adapters.ocudu_websocket.send",
        lambda *args: {"status": "error", "message": "invalid"},  # type: ignore[attr-defined]
    )
    result = OcuduWebSocketResearchBackend().apply(Action.MCS_BACKOFF, {"mcs_cap": 5})
    assert not result.accepted
    assert result.reported_parameters == {}


def test_websocket_backend_accepts_native_ocudu_ack(monkeypatch: object) -> None:
    monkeypatch.setattr(
        "src.xapp.adapters.ocudu_websocket.send",
        lambda *args: {"cmd": "research_action_set", "timestamp": "2026-08-13T18:07:47.761"},  # type: ignore[attr-defined]
    )
    result = OcuduWebSocketResearchBackend().apply(Action.NO_OP, {})
    assert result.accepted
    assert result.reported_action is None
    assert result.reported_parameters == {}
