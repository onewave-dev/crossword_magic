"""Tests for Telegram handler configuration."""

from unittest.mock import MagicMock

from telegram.ext import CommandHandler

from app import configure_telegram_handlers, quit_command


def test_configure_handlers_includes_quit_command() -> None:
    """The dispatcher should register the /quit command."""

    telegram_application = MagicMock()

    configure_telegram_handlers(telegram_application)

    quit_handlers = [
        call.args[0]
        for call in telegram_application.add_handler.call_args_list
        if isinstance(call.args[0], CommandHandler) and "quit" in call.args[0].commands
    ]

    assert any(handler.callback is quit_command for handler in quit_handlers)
