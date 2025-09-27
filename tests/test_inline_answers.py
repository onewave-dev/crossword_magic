"""Tests for inline answer parsing."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from telegram.constants import ChatType

from app import _parse_inline_answer, inline_answer_handler


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
    message = SimpleNamespace(text="β12-3:Αθήνα", message_thread_id=None, reply_text=AsyncMock())
    update = SimpleNamespace(effective_chat=chat, effective_message=message)
    context = SimpleNamespace(user_data={})

    with patch("app._handle_answer_submission", new_callable=AsyncMock) as handler_mock:
        await inline_answer_handler(update, context)

    handler_mock.assert_awaited_once_with(context, chat, message, "Β12-3", "Αθήνα")
