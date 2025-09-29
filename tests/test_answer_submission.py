"""Tests for handling answer submissions."""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from telegram.constants import ChatType

import app
from app import _handle_answer_submission, state
from utils.crossword import Direction, Puzzle, Slot
from utils.storage import GameState


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
def puzzle_with_shared_number() -> Puzzle:
    """Create a simple puzzle with across and down slots sharing the same number."""

    puzzle = Puzzle.from_size("test-puzzle", "Тест", "ru", 5, 5)
    puzzle.slots = [
        Slot(
            slot_id="A1",
            direction=Direction.ACROSS,
            number=1,
            start_row=0,
            start_col=0,
            length=5,
            clue="Популярная порода",
            answer="шпиц",
        ),
        Slot(
            slot_id="D1",
            direction=Direction.DOWN,
            number=1,
            start_row=0,
            start_col=0,
            length=3,
            clue="Собака",
            answer="дог",
        ),
    ]
    return puzzle


def _build_game_state(puzzle: Puzzle) -> GameState:
    now = time.time()
    return GameState(
        chat_id=777,
        puzzle_id=puzzle.id,
        filled_cells={},
        solved_slots=set(),
        score=0,
        hints_used=0,
        started_at=now,
        last_update=now,
    )


def _prepare_context(
    tmp_path: Path,
) -> tuple[
    SimpleNamespace,
    SimpleNamespace,
    SimpleNamespace,
    Callable[[Puzzle, GameState], str],
]:
    reply_text = AsyncMock()
    reply_photo = AsyncMock()
    message = SimpleNamespace(
        reply_text=reply_text,
        reply_photo=reply_photo,
        message_thread_id=None,
        message_id=1,
    )
    chat = SimpleNamespace(id=777, type=ChatType.PRIVATE)
    bot = SimpleNamespace(send_chat_action=AsyncMock())
    context = SimpleNamespace(bot=bot)

    image_path = tmp_path / "grid.png"
    image_path.write_bytes(b"fake-image")

    render_mock = lambda _puzzle, _state: str(image_path)

    return message, chat, context, render_mock


@pytest.mark.anyio
async def test_numeric_slot_answer_matches_across(monkeypatch, tmp_path, puzzle_with_shared_number):
    game_state = _build_game_state(puzzle_with_shared_number)
    message, chat, context, render_mock = _prepare_context(tmp_path)

    clue_mock = AsyncMock()

    monkeypatch.setattr(state, "active_states", {})
    monkeypatch.setattr(app, "_load_state_for_chat", lambda chat_id: game_state)
    monkeypatch.setattr(app, "_load_puzzle_for_state", lambda _: puzzle_with_shared_number)
    monkeypatch.setattr(app, "_store_state", lambda _state: None)
    monkeypatch.setattr(app, "_send_clues_update", clue_mock)
    monkeypatch.setattr(app, "render_puzzle", render_mock)

    await _handle_answer_submission(context, chat, message, "1", "шпиц")

    assert game_state.score == puzzle_with_shared_number.slots[0].length
    assert game_state.solved_slots == {"A1"}
    assert message.reply_photo.await_count == 1
    assert "A1" in message.reply_photo.await_args.kwargs.get("caption", "")
    assert clue_mock.await_count == 1
    assert message.reply_text.await_count == 0


@pytest.mark.anyio
async def test_numeric_slot_answer_matches_down(monkeypatch, tmp_path, puzzle_with_shared_number):
    game_state = _build_game_state(puzzle_with_shared_number)
    message, chat, context, render_mock = _prepare_context(tmp_path)

    clue_mock = AsyncMock()

    monkeypatch.setattr(state, "active_states", {})
    monkeypatch.setattr(app, "_load_state_for_chat", lambda chat_id: game_state)
    monkeypatch.setattr(app, "_load_puzzle_for_state", lambda _: puzzle_with_shared_number)
    monkeypatch.setattr(app, "_store_state", lambda _state: None)
    monkeypatch.setattr(app, "_send_clues_update", clue_mock)
    monkeypatch.setattr(app, "render_puzzle", render_mock)

    await _handle_answer_submission(context, chat, message, "1", "дог")

    assert game_state.score == puzzle_with_shared_number.slots[1].length
    assert game_state.solved_slots == {"D1"}
    assert message.reply_photo.await_count == 1
    assert "D1" in message.reply_photo.await_args.kwargs.get("caption", "")
    assert clue_mock.await_count == 1
    assert message.reply_text.await_count == 0
