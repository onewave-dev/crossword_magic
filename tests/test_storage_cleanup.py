"""Tests for storage cleanup helpers."""

from __future__ import annotations

from pathlib import Path

from _pytest.monkeypatch import MonkeyPatch

from utils import storage


def test_delete_state_preserves_base_game_when_removing_test(
    tmp_path, monkeypatch: MonkeyPatch
) -> None:
    """Deleting a test game must not remove the base state file."""

    tmp_states: Path = tmp_path
    admin_game_id = "admin:42"
    base_game_id = "42"

    admin_file = tmp_states / f"{admin_game_id}{storage.STATE_FILE_SUFFIX}"
    base_file = tmp_states / f"{base_game_id}{storage.STATE_FILE_SUFFIX}"
    admin_file.write_text("{}", encoding="utf-8")
    base_file.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(storage, "STATES_DIR", tmp_states)
    storage.delete_state(admin_game_id)

    assert not admin_file.exists(), "Test game state should be removed"
    assert base_file.exists(), "Base game state must persist after test cleanup"


def test_delete_state_removes_legacy_chat_file(
    tmp_path, monkeypatch: MonkeyPatch
) -> None:
    """Deleting a normal game should also remove its legacy chat-id file."""

    tmp_states: Path = tmp_path
    game_id = "0000012345"
    legacy_chat_file = tmp_states / f"{int(game_id)}{storage.STATE_FILE_SUFFIX}"
    legacy_chat_file.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(storage, "STATES_DIR", tmp_states)
    storage.delete_state(game_id)

    assert not legacy_chat_file.exists(), "Legacy chat state should be removed"
