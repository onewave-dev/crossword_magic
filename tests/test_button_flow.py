"""Tests covering the button-driven new game flow."""

from __future__ import annotations

import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from telegram.constants import ChatType

import app
from app import (
    BUTTON_LANGUAGE_KEY,
    BUTTON_NEW_GAME_KEY,
    BUTTON_STEP_KEY,
    BUTTON_STEP_THEME,
    GENERATION_NOTICE_KEY,
    MODE_AWAIT_THEME,
    button_language_handler,
    button_theme_handler,
    state,
)
from utils.crossword import Puzzle
from utils.storage import GameState, Player


@pytest.fixture
def anyio_backend():
    return "asyncio"


class DummyJob:
    def __init__(self, chat_id: int, name: str) -> None:
        self.chat_id = chat_id
        self.name = name
        self.cancelled = False

    def schedule_removal(self) -> None:
        self.cancelled = True


class DummyJobQueue:
    def __init__(self) -> None:
        self.submitted: list[tuple] = []

    def run_once(self, callback, when, *, chat_id: int, name: str):  # noqa: ANN001 - signature mimics library
        job = DummyJob(chat_id, name)
        self.submitted.append((callback, when, chat_id, name))
        return job


@pytest.mark.anyio
async def test_button_language_handler_initialises_flow_when_missing():
    chat = SimpleNamespace(id=123, type=ChatType.PRIVATE)
    message = SimpleNamespace(text=" Ru ", message_thread_id=None, reply_text=AsyncMock())
    context = SimpleNamespace(chat_data={}, user_data={})
    update = SimpleNamespace(effective_chat=chat, effective_message=message)

    app.set_chat_mode(context, app.MODE_AWAIT_LANGUAGE)

    await button_language_handler(update, context)

    flow_state = context.chat_data[BUTTON_NEW_GAME_KEY]
    assert flow_state[BUTTON_LANGUAGE_KEY] == "ru"
    assert flow_state[BUTTON_STEP_KEY] == BUTTON_STEP_THEME
    assert app.get_chat_mode(context) == MODE_AWAIT_THEME
    assert context.user_data["new_game_language"] == "ru"
    message.reply_text.assert_awaited_once_with("Отлично! Теперь укажите тему кроссворда.")


@pytest.mark.anyio
async def test_button_theme_handler_generates_puzzle_via_completion_menu(monkeypatch):
    chat_id = 789
    puzzle = Puzzle.from_size("test-puzzle", "История", "ru", 3, 3)
    now = time.time()
    game_state = GameState(
        chat_id=chat_id,
        puzzle_id=puzzle.id,
        filled_cells={},
        solved_slots=set(),
        score=0,
        started_at=now,
        last_update=now,
        scoreboard={chat_id: 0},
        players={chat_id: Player(user_id=chat_id, name="Player", dm_chat_id=chat_id)},
        host_id=chat_id,
        mode="single",
        status="running",
    )

    generate_calls: list[tuple[int, str, str]] = []

    def fake_generate(request_chat_id: int, language: str, theme: str):
        generate_calls.append((request_chat_id, language, theme))
        return puzzle, game_state

    monkeypatch.setattr(app, "_generate_puzzle", fake_generate)
    deliver_mock = AsyncMock(return_value=True)
    monkeypatch.setattr(app, "_deliver_puzzle_via_bot", deliver_mock)

    message = SimpleNamespace(text="  Древний мир  ", message_thread_id=None, reply_text=AsyncMock())
    chat = SimpleNamespace(id=chat_id, type=ChatType.PRIVATE)

    previous_job = DummyJob(chat_id, "old-job")
    job_queue = DummyJobQueue()
    context = SimpleNamespace(
        bot=SimpleNamespace(),
        chat_data={
            BUTTON_NEW_GAME_KEY: {
                BUTTON_STEP_KEY: BUTTON_STEP_THEME,
                BUTTON_LANGUAGE_KEY: "ru",
            },
            "reminder_job": previous_job,
        },
        job_queue=job_queue,
    )
    update = SimpleNamespace(effective_chat=chat, effective_message=message)

    state.generating_chats.clear()

    await button_theme_handler(update, context)

    assert generate_calls == [(chat_id, "ru", "Древний мир")]
    deliver_mock.assert_awaited_once_with(context, chat_id, puzzle, game_state)
    assert BUTTON_NEW_GAME_KEY not in context.chat_data
    assert GENERATION_NOTICE_KEY not in context.chat_data
    assert chat_id not in state.generating_chats
    assert previous_job.cancelled is True
    assert len(job_queue.submitted) == 1
    assert context.chat_data["reminder_job"].name.startswith("hint-reminder-")
    message.reply_text.assert_awaited()
