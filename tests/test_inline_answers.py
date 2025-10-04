"""Tests for inline answer parsing."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import app
import pytest
from telegram.constants import ChatType

from telegram.ext import ConversationHandler

from app import (
    GENERATION_NOTICE_KEY,
    GENERATION_TOKEN_KEY,
    LOBBY_INVITE_BUTTON_TEXT,
    LOBBY_LINK_BUTTON_TEXT,
    LOBBY_SHARE_CONTACT_BUTTON_TEXT,
    LOBBY_START_BUTTON_TEXT,
    _parse_inline_answer,
    handle_theme,
    inline_answer_handler,
    quit_command,
    state,
)
from utils.crossword import Puzzle
from utils.storage import GameState


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("A1 - Tokyo", ("A1", "Tokyo")),
        (" b12 –  Kyoto ", ("B12", "Kyoto")),
        ("c3:Rio", ("C3", "Rio")),
        ("А1 - ответ", ("A1", "ответ")),
        ("д7 - слово", ("D7", "слово")),
        ("я5 - Ответ", ("Я5", "Ответ")),
        ("β12-3:Αθήνα", ("Β12-3", "Αθήνα")),
        (" z9-2 :  respuesta ", ("Z9-2", "respuesta")),
        ("A2 — мопс", ("A2", "мопс")),
        ("F5 ‑ Oslo", ("F5", "Oslo")),
        ("A1 шпиц", ("A1", "шпиц")),
        ("A1  шпиц", ("A1", "шпиц")),
        ("A1- шпиц", ("A1", "шпиц")),
        ("А1:шпиц", ("A1", "шпиц")),
        ("А1:  шпиц", ("A1", "шпиц")),
        ("1 шпиц", ("1", "шпиц")),
        ("1 - шпиц", ("1", "шпиц")),
        ("1:шпиц", ("1", "шпиц")),
        ("1: шпиц", ("1", "шпиц")),
        ("  15   ответ  ", ("15", "ответ")),
    ],
)
def test_parse_inline_answer_valid(text, expected):
    assert _parse_inline_answer(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "",
        "  ",
        "A1",
        "A1-",
        "- Tokyo",
        "A - foo",
        "A1 - ",
        "1",
        "1-",
        "слово без слота",
        "A_- foo",
    ],
)
def test_parse_inline_answer_invalid(text):
    assert _parse_inline_answer(text) is None


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_inline_handler_passes_parsed_values_to_submission_handler():
    chat = SimpleNamespace(id=123, type=ChatType.PRIVATE)
    message = SimpleNamespace(
        text="β12-3:Αθήνα",
        message_thread_id=None,
        reply_text=AsyncMock(),
        message_id=1,
    )
    update = SimpleNamespace(effective_chat=chat, effective_message=message)
    context = SimpleNamespace(user_data={})

    with patch("app._handle_answer_submission", new_callable=AsyncMock) as handler_mock:
        await inline_answer_handler(update, context)

    handler_mock.assert_awaited_once_with(context, chat, message, "Β12-3", "Αθήνα")


@pytest.mark.anyio
async def test_inline_handler_replies_when_parse_fails_with_active_game():
    chat = SimpleNamespace(id=124, type=ChatType.PRIVATE)
    message = SimpleNamespace(
        text="неверный формат",
        message_thread_id=None,
        reply_text=AsyncMock(),
        message_id=2,
    )
    update = SimpleNamespace(effective_chat=chat, effective_message=message)
    context = SimpleNamespace(user_data={})

    with patch("app._load_state_for_chat", return_value=SimpleNamespace()):
        with patch("app._handle_answer_submission", new_callable=AsyncMock) as handler_mock:
            await inline_answer_handler(update, context)

    handler_mock.assert_not_awaited()
    message.reply_text.assert_awaited_once()
    reply_call = message.reply_text.await_args
    assert reply_call.args
    assert app.ANSWER_INSTRUCTIONS_TEXT in reply_call.args[0]


@pytest.mark.anyio
@pytest.mark.parametrize(
    "caption",
    [
        LOBBY_INVITE_BUTTON_TEXT,
        LOBBY_LINK_BUTTON_TEXT,
        LOBBY_SHARE_CONTACT_BUTTON_TEXT,
        LOBBY_START_BUTTON_TEXT,
    ],
)
async def test_inline_handler_ignores_lobby_control_captions(caption, monkeypatch):
    chat = SimpleNamespace(id=777, type=ChatType.PRIVATE)
    message = SimpleNamespace(
        text=caption,
        message_thread_id=None,
        reply_text=AsyncMock(),
        message_id=8,
    )
    update = SimpleNamespace(effective_chat=chat, effective_message=message)
    context = SimpleNamespace(user_data={}, chat_data={})

    monkeypatch.setattr(app, "_load_state_for_chat", lambda _: SimpleNamespace())
    submission_mock = AsyncMock()
    monkeypatch.setattr(app, "_handle_answer_submission", submission_mock)

    await inline_answer_handler(update, context)

    submission_mock.assert_not_awaited()
    message.reply_text.assert_not_awaited()


@pytest.mark.anyio
async def test_inline_answer_allowed_after_extra_theme_during_generation():
    chat = SimpleNamespace(id=777, type=ChatType.PRIVATE)
    theme_message = SimpleNamespace(
        text="Космос",
        message_thread_id=None,
        reply_text=AsyncMock(),
    )
    update = SimpleNamespace(effective_chat=chat, effective_message=theme_message)
    context = SimpleNamespace(user_data={"new_game_language": "ru"}, chat_data={}, job_queue=None)

    state.generating_chats.add(chat.id)
    try:
        with patch("app._reject_group_chat", new=AsyncMock(return_value=True)):
            result = await handle_theme(update, context)

        assert result == ConversationHandler.END
        theme_message.reply_text.assert_awaited_once()
        assert "new_game_language" not in context.user_data

        # Simulate puzzle generation finishing and player sending an inline answer.
        state.generating_chats.discard(chat.id)
        answer_message = SimpleNamespace(
            text="A1 - ответ",
            message_thread_id=None,
            reply_text=AsyncMock(),
        )
        answer_update = SimpleNamespace(effective_chat=chat, effective_message=answer_message)

        with patch("app._reject_group_chat", new=AsyncMock(return_value=True)):
            with patch("app._handle_answer_submission", new_callable=AsyncMock) as handler_mock:
                await inline_answer_handler(answer_update, context)

        handler_mock.assert_awaited_once_with(context, chat, answer_message, "A1", "ответ")
    finally:
        state.generating_chats.discard(chat.id)


@pytest.mark.anyio
async def test_inline_handler_uses_caption_when_text_missing():
    chat = SimpleNamespace(id=555, type=ChatType.PRIVATE)
    message = SimpleNamespace(
        text=None,
        caption="D1 - такса",
        message_thread_id=None,
        reply_text=AsyncMock(),
        message_id=3,
    )
    update = SimpleNamespace(effective_chat=chat, effective_message=message)
    context = SimpleNamespace(user_data={})

    with patch("app._handle_answer_submission", new_callable=AsyncMock) as handler_mock:
        await inline_answer_handler(update, context)

    handler_mock.assert_awaited_once_with(context, chat, message, "D1", "такса")


@pytest.mark.anyio
async def test_inline_handler_silent_when_no_game_and_parse_fails():
    chat = SimpleNamespace(id=556, type=ChatType.PRIVATE)
    message = SimpleNamespace(
        text="какой-то текст",
        message_thread_id=None,
        reply_text=AsyncMock(),
        message_id=4,
    )
    update = SimpleNamespace(effective_chat=chat, effective_message=message)
    context = SimpleNamespace(user_data={})

    with patch("app._load_state_for_chat", return_value=None):
        with patch("app._handle_answer_submission", new_callable=AsyncMock) as handler_mock:
            await inline_answer_handler(update, context)

    handler_mock.assert_not_awaited()
    message.reply_text.assert_not_awaited()


@pytest.mark.anyio
async def test_handle_theme_skips_delivery_after_quit_during_generation(monkeypatch):
    chat_id = 901
    chat = SimpleNamespace(id=chat_id, type=ChatType.PRIVATE)
    theme_message = SimpleNamespace(
        text="История",
        message_thread_id=None,
        reply_text=AsyncMock(),
    )
    update = SimpleNamespace(effective_chat=chat, effective_message=theme_message)
    quit_message = SimpleNamespace(
        text="/quit",
        message_thread_id=None,
        reply_text=AsyncMock(),
    )
    quit_update = SimpleNamespace(effective_chat=chat, effective_message=quit_message)
    context = SimpleNamespace(
        user_data={"new_game_language": "ru"},
        chat_data={},
        bot=SimpleNamespace(send_message=AsyncMock()),
        job_queue=None,
    )

    state.generating_chats.clear()

    puzzle = Puzzle.from_size("test-puzzle", "История", "ru", 3, 3)
    game_state = GameState(chat_id=chat_id, puzzle_id=puzzle.id)

    def fake_generate(request_chat_id: int, language: str, theme: str):
        assert request_chat_id == chat_id
        return puzzle, game_state

    class DummyLoop:
        def __init__(self) -> None:
            self.started = asyncio.Event()
            self.resume = asyncio.Event()

        async def run_in_executor(self, executor, func, *args):  # noqa: ANN001 - signature parity
            self.started.set()
            await self.resume.wait()
            return func(*args)

    dummy_loop = DummyLoop()

    monkeypatch.setattr(app, "_generate_puzzle", fake_generate)
    deliver_mock = AsyncMock()
    monkeypatch.setattr(app, "_deliver_puzzle_via_bot", deliver_mock)
    cleanup_mock = MagicMock()
    monkeypatch.setattr(app, "_cleanup_game_state", cleanup_mock)
    cleanup_chat_mock = MagicMock()
    monkeypatch.setattr(app, "_cleanup_chat_resources", cleanup_chat_mock)
    cancel_mock = MagicMock()
    monkeypatch.setattr(app, "_cancel_reminder", cancel_mock)
    load_state_mock = MagicMock(return_value=None)
    monkeypatch.setattr(app, "_load_state_for_chat", load_state_mock)
    reject_mock = AsyncMock(return_value=True)
    monkeypatch.setattr(app, "_reject_group_chat", reject_mock)
    monkeypatch.setattr(app.asyncio, "get_running_loop", lambda: dummy_loop)

    generation_task = asyncio.create_task(handle_theme(update, context))
    await dummy_loop.started.wait()

    assert context.chat_data.get(GENERATION_NOTICE_KEY)
    assert context.chat_data.get(GENERATION_TOKEN_KEY)

    await quit_command(quit_update, context)

    cleanup_chat_mock.assert_called_once_with(chat_id)
    quit_message.reply_text.assert_awaited_once()
    assert context.chat_data.get(GENERATION_TOKEN_KEY) is None
    assert context.chat_data.get(GENERATION_NOTICE_KEY) is None

    dummy_loop.resume.set()
    result = await generation_task

    assert result == ConversationHandler.END
    deliver_mock.assert_not_awaited()
    cleanup_mock.assert_called_once_with(game_state)
    assert chat_id not in state.generating_chats
    assert context.chat_data.get(GENERATION_TOKEN_KEY) is None
    assert context.chat_data.get(GENERATION_NOTICE_KEY) is None
    assert context.chat_data.get("chat_mode") is None
    assert context.user_data.get("new_game_language") is None
