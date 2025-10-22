"""Tests covering the button-driven new game flow."""

from __future__ import annotations

import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from telegram import InlineKeyboardMarkup
from telegram.constants import ChatType

import app
from app import (
    BUTTON_LANGUAGE_KEY,
    BUTTON_NEW_GAME_KEY,
    BUTTON_STEP_LANGUAGE,
    BUTTON_STEP_KEY,
    BUTTON_STEP_THEME,
    LOBBY_START_BUTTON_TEXT,
    MODE_AWAIT_LANGUAGE,
    GENERATION_NOTICE_KEY,
    MODE_AWAIT_THEME,
    NEW_PUZZLE_CALLBACK_PREFIX,
    button_language_handler,
    button_theme_handler,
    completion_callback_handler,
    language_callback_handler,
    LANGUAGE_BUTTONS,
    LANGUAGE_CALLBACK_PREFIX,
    LANGUAGE_PROMPT_TEXT,
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

    flow_state = context.user_data[BUTTON_NEW_GAME_KEY]
    assert flow_state[BUTTON_LANGUAGE_KEY] == "ru"
    assert flow_state[BUTTON_STEP_KEY] == BUTTON_STEP_THEME
    assert app.get_chat_mode(context) == MODE_AWAIT_THEME
    assert context.user_data["new_game_language"] == "ru"
    message.reply_text.assert_awaited_once_with("Отлично! Теперь укажите тему кроссворда.")


@pytest.mark.anyio
async def test_language_callback_handler_processes_selection():
    chat = SimpleNamespace(id=321, type=ChatType.PRIVATE)
    message = SimpleNamespace(
        message_thread_id=None,
        reply_text=AsyncMock(),
        edit_reply_markup=AsyncMock(),
    )
    query = SimpleNamespace(
        data=f"{LANGUAGE_CALLBACK_PREFIX}de",
        message=message,
        answer=AsyncMock(),
    )
    update = SimpleNamespace(
        effective_chat=chat,
        callback_query=query,
        effective_message=message,
    )
    context = SimpleNamespace(chat_data={}, user_data={})

    app.set_chat_mode(context, app.MODE_AWAIT_LANGUAGE)

    await language_callback_handler(update, context)

    flow_state = context.user_data[BUTTON_NEW_GAME_KEY]
    assert flow_state[BUTTON_LANGUAGE_KEY] == "de"
    assert flow_state[BUTTON_STEP_KEY] == BUTTON_STEP_THEME
    assert context.user_data["new_game_language"] == "de"
    message.reply_text.assert_awaited_once_with("Отлично! Теперь укажите тему кроссворда.")
    message.edit_reply_markup.assert_awaited_once_with(reply_markup=None)
    query.answer.assert_awaited_once()


@pytest.mark.anyio
async def test_completion_callback_followed_by_language_message_uses_private_storage(monkeypatch):
    chat = SimpleNamespace(id=555, type=ChatType.PRIVATE)
    callback_message = SimpleNamespace(message_thread_id=None, edit_reply_markup=AsyncMock())
    query = SimpleNamespace(
        data=f"{NEW_PUZZLE_CALLBACK_PREFIX}prev",
        answer=AsyncMock(),
        message=callback_message,
    )
    update_callback = SimpleNamespace(
        effective_chat=chat,
        effective_message=callback_message,
        callback_query=query,
    )
    context = SimpleNamespace(
        chat_data={},
        user_data={},
        bot=SimpleNamespace(send_message=AsyncMock()),
    )

    monkeypatch.setattr(app, "_load_state_for_chat", lambda _chat_id: None)

    await completion_callback_handler(update_callback, context)

    assert app.get_chat_mode(context) == MODE_AWAIT_LANGUAGE
    assert BUTTON_NEW_GAME_KEY in context.user_data
    flow_state = context.user_data[BUTTON_NEW_GAME_KEY]
    assert flow_state[BUTTON_STEP_KEY] == BUTTON_STEP_LANGUAGE
    query.answer.assert_awaited_once()
    callback_message.edit_reply_markup.assert_awaited_once()
    send_call = context.bot.send_message.await_args
    assert send_call.kwargs["chat_id"] == chat.id
    assert send_call.kwargs["text"] == LANGUAGE_PROMPT_TEXT
    reply_markup = send_call.kwargs["reply_markup"]
    assert isinstance(reply_markup, InlineKeyboardMarkup)
    flat_buttons = [
        button
        for row in reply_markup.inline_keyboard
        for button in row
    ]
    assert [button.text for button in flat_buttons] == [label for _, label in LANGUAGE_BUTTONS]
    assert all(
        button.callback_data.startswith(LANGUAGE_CALLBACK_PREFIX)
        for button in flat_buttons
    )

    language_message = SimpleNamespace(text=" En ", message_thread_id=None, reply_text=AsyncMock())
    update_language = SimpleNamespace(effective_chat=chat, effective_message=language_message)

    await button_language_handler(update_language, context)

    updated_state = context.user_data[BUTTON_NEW_GAME_KEY]
    assert updated_state[BUTTON_LANGUAGE_KEY] == "en"
    assert updated_state[BUTTON_STEP_KEY] == BUTTON_STEP_THEME
    assert context.user_data["new_game_language"] == "en"
    assert app.get_chat_mode(context) == MODE_AWAIT_THEME
    language_message.reply_text.assert_awaited_once()


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
        chat_data={"reminder_job": previous_job},
        user_data={
            BUTTON_NEW_GAME_KEY: {
                BUTTON_STEP_KEY: BUTTON_STEP_THEME,
                BUTTON_LANGUAGE_KEY: "ru",
            }
        },
        job_queue=job_queue,
    )
    update = SimpleNamespace(effective_chat=chat, effective_message=message)

    state.generating_chats.clear()

    await button_theme_handler(update, context)

    assert generate_calls == [(chat_id, "ru", "Древний мир")]
    deliver_mock.assert_awaited_once_with(context, chat_id, puzzle, game_state)
    assert BUTTON_NEW_GAME_KEY not in context.user_data
    assert GENERATION_NOTICE_KEY not in context.chat_data
    assert chat_id not in state.generating_chats
    assert previous_job.cancelled is True
    assert len(job_queue.submitted) == 1
    assert context.chat_data["reminder_job"].name.startswith("hint-reminder-")
    message.reply_text.assert_awaited()


@pytest.mark.anyio
async def test_button_language_handler_ignores_lobby_caption():
    chat = SimpleNamespace(id=999, type=ChatType.PRIVATE)
    message = SimpleNamespace(
        text=f"  {LOBBY_START_BUTTON_TEXT}  ",
        message_thread_id=None,
        reply_text=AsyncMock(),
    )
    context = SimpleNamespace(
        chat_data={},
        user_data={
            BUTTON_NEW_GAME_KEY: {
                BUTTON_STEP_KEY: BUTTON_STEP_LANGUAGE,
                BUTTON_LANGUAGE_KEY: "existing",
            }
        },
    )
    update = SimpleNamespace(effective_chat=chat, effective_message=message)

    app.set_chat_mode(context, app.MODE_AWAIT_LANGUAGE)

    await button_language_handler(update, context)

    flow_state = context.user_data[BUTTON_NEW_GAME_KEY]
    assert flow_state[BUTTON_LANGUAGE_KEY] == "existing"
    assert flow_state[BUTTON_STEP_KEY] == BUTTON_STEP_LANGUAGE
    assert app.get_chat_mode(context) == MODE_AWAIT_LANGUAGE
    message.reply_text.assert_not_awaited()


@pytest.mark.anyio
async def test_button_theme_handler_ignores_lobby_caption(monkeypatch):
    chat_id = 1001
    chat = SimpleNamespace(id=chat_id, type=ChatType.PRIVATE)
    message = SimpleNamespace(
        text=f" {LOBBY_START_BUTTON_TEXT} ",
        message_thread_id=None,
        reply_text=AsyncMock(),
    )
    context = SimpleNamespace(
        bot=SimpleNamespace(),
        chat_data={},
        user_data={
            BUTTON_NEW_GAME_KEY: {
                BUTTON_STEP_KEY: BUTTON_STEP_THEME,
                BUTTON_LANGUAGE_KEY: "ru",
            }
        },
        job_queue=None,
    )
    update = SimpleNamespace(effective_chat=chat, effective_message=message)

    app.set_chat_mode(context, app.MODE_AWAIT_THEME)

    run_generate_mock = AsyncMock()
    monkeypatch.setattr(app, "_run_generate_puzzle", run_generate_mock)

    state.generating_chats.clear()

    await button_theme_handler(update, context)

    flow_state = context.user_data[BUTTON_NEW_GAME_KEY]
    assert flow_state[BUTTON_STEP_KEY] == BUTTON_STEP_THEME
    assert flow_state[BUTTON_LANGUAGE_KEY] == "ru"
    assert app.get_chat_mode(context) == MODE_AWAIT_THEME
    run_generate_mock.assert_not_awaited()
    assert chat_id not in state.generating_chats
    message.reply_text.assert_not_awaited()
