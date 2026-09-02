from __future__ import annotations

from pathlib import Path

from src.common.config import load_methodology_lock, state_feature_names


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_methodology_lock_has_exact_actions_and_contiguous_state() -> None:
    lock = load_methodology_lock(PROJECT_ROOT)
    actions = lock["actions"]["definitions"]
    assert [action["code"] for action in actions] == ["A0", "A1", "A2", "A3", "A4"]
    assert [action["id"] for action in actions] == list(range(5))
    names = state_feature_names(lock)
    assert len(names) == 27
    assert names[0] == "p_jam"
    assert names[-1] == "previous_action_a4"


def test_authoritative_hashes_are_locked() -> None:
    lock = load_methodology_lock(PROJECT_ROOT)
    hashes = {item["path"]: item["sha256"] for item in lock["authoritative_sources"]}
    assert hashes["CODEX_IMPLEMENTATION_INSTRUCTIONS.md"] == (
        "C517187C9911F388B0DED27C42B5CE768C09F99CC299BF686BCCD13272169186"
    )
    assert hashes["Proposed_Methodology_Action_Explainable_ORAN_Anti_Jamming.docx"] == (
        "F5AC544D0E415FBD87B4C0790A5A2AA9043E3AC4752F9CADB642F1A4D0933957"
    )

