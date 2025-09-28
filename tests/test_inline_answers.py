"""Tests for inline answer parsing."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from telegram.constants import ChatType

from telegram.ext import ConversationHandler

from app import _parse_inline_answer, handle_theme, inline_answer_handler, state


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
        "123 - foo",
        "A - foo",
        "A1 - ",
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
async def test_inline_handler_replies_when_parse_fails():
    chat = SimpleNamespace(id=124, type=ChatType.PRIVATE)
    message = SimpleNamespace(
        text="неверный формат",
        message_thread_id=None,
        reply_text=AsyncMock(),
        message_id=2,
    )
    update = SimpleNamespace(effective_chat=chat, effective_message=message)
    context = SimpleNamespace(user_data={})

    with patch("app._handle_answer_submission", new_callable=AsyncMock) as handler_mock:
        await inline_answer_handler(update, context)

    handler_mock.assert_not_awaited()
    message.reply_text.assert_awaited_once()
    reply_call = message.reply_text.await_args
    assert reply_call.args
    assert "A1 - слово" in reply_call.args[0]


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
