"""Tests covering startup restoration of multiplayer lobbies."""

from __future__ import annotations

import pytest

from app import _select_preferred_restored_states, _update_dm_mappings, state
from utils.storage import GameState, Player


@pytest.fixture
def reset_state():
    """Ensure the global bot state is reset before and after the test."""

    original_settings = state.settings
    state.active_games.clear()
    state.chat_to_game.clear()
    state.player_chats.clear()
    state.dm_chat_to_game.clear()
    state.join_codes.clear()
    state.lobby_messages.clear()
    state.lobby_host_invites.clear()
    state.lobby_invite_requests.clear()
    state.generating_chats.clear()
    state.lobby_generation_tasks.clear()
    state.scheduled_jobs.clear()
    state.chat_threads.clear()
    yield
    state.active_games.clear()
    state.chat_to_game.clear()
    state.player_chats.clear()
    state.dm_chat_to_game.clear()
    state.join_codes.clear()
    state.lobby_messages.clear()
    state.lobby_host_invites.clear()
    state.lobby_invite_requests.clear()
    state.generating_chats.clear()
    state.lobby_generation_tasks.clear()
    state.scheduled_jobs.clear()
    state.chat_threads.clear()
    state.settings = original_settings


def _make_state(
    *,
    chat_id: int,
    game_id: str,
    mode: str,
    last_update: float,
    players: dict[int, Player] | None = None,
    join_codes: dict[str, str] | None = None,
) -> GameState:
    state_obj = GameState(
        chat_id=chat_id,
        puzzle_id="p",
        mode=mode,
        status="lobby" if mode != "single" else "running",
        game_id=game_id,
        started_at=0.0,
        last_update=last_update,
    )
    if players:
        state_obj.players = players
    if join_codes:
        state_obj.join_codes = join_codes
    return state_obj


def test_select_preferred_restored_states_prefers_multiplayer() -> None:
    """A turn-based lobby should win over a newer single helper."""

    chat_id = 111
    helper = _make_state(
        chat_id=chat_id,
        game_id=str(chat_id),
        mode="single",
        last_update=200.0,
    )
    lobby = _make_state(
        chat_id=chat_id,
        game_id="turn:abc",
        mode="turn_based",
        last_update=150.0,
    )

    preferred, helpers = _select_preferred_restored_states(
        {helper.game_id: helper, lobby.game_id: lobby}
    )

    assert preferred[chat_id] is lobby
    assert helpers == [helper]


def test_startup_restoration_populates_private_lobby_mappings(reset_state) -> None:
    """Startup should map private chats and DMs to the restored lobby."""

    chat_id = 555
    host_id = 42
    guest_id = 99
    guest_dm = 777
    helper = _make_state(
        chat_id=chat_id,
        game_id=str(chat_id),
        mode="single",
        last_update=10.0,
    )
    lobby_players = {
        host_id: Player(user_id=host_id, name="Host", dm_chat_id=chat_id),
        guest_id: Player(user_id=guest_id, name="Guest", dm_chat_id=guest_dm),
    }
    lobby = _make_state(
        chat_id=chat_id,
        game_id="turn:lobby",
        mode="turn_based",
        last_update=20.0,
        players=lobby_players,
        join_codes={"JOIN": "turn:lobby"},
    )

    restored = {helper.game_id: helper, lobby.game_id: lobby}
    preferred, helpers = _select_preferred_restored_states(restored)
    helper_ids = {state.game_id for state in helpers}

    state.active_games = {
        game_id: game_state
        for game_id, game_state in restored.items()
        if game_id not in helper_ids
    }
    state.chat_to_game = {
        chat: game_state.game_id for chat, game_state in preferred.items()
    }
    state.dm_chat_to_game = {}
    state.player_chats = {}
    state.join_codes = {}
    for game_state in state.active_games.values():
        for code, target in game_state.join_codes.items():
            state.join_codes[code] = target
        _update_dm_mappings(game_state)

    assert helpers == [helper]
    assert helper.game_id not in state.active_games
    assert state.chat_to_game[chat_id] == lobby.game_id
    assert state.join_codes == {"JOIN": lobby.game_id}
    assert state.dm_chat_to_game[guest_dm] == lobby.game_id
    assert state.player_chats[guest_id] == guest_dm
