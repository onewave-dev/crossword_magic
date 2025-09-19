"""Centralised logging helpers for the crossword bot project."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from contextvars import ContextVar, Token
from logging.config import dictConfig
from typing import Iterator

BASE_LOGGER_NAME = "crossword"

_chat_id_var: ContextVar[str] = ContextVar("chat_id", default="-")
_puzzle_id_var: ContextVar[str] = ContextVar("puzzle_id", default="-")


class ChatPuzzleContextFilter(logging.Filter):
    """Ensure that log records always contain chat and puzzle identifiers."""

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: D401 - logging API
        record.chat_id = getattr(record, "chat_id", _chat_id_var.get("-"))
        record.puzzle_id = getattr(record, "puzzle_id", _puzzle_id_var.get("-"))
        return True


def configure_logging(level: int | str = "INFO") -> None:
    """Configure project-wide logging using :func:`logging.config.dictConfig`."""

    dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "filters": {
                "chat_puzzle": {
                    "()": "utils.logging_config.ChatPuzzleContextFilter",
                }
            },
            "formatters": {
                "default": {
                    "format": "%(asctime)s [%(levelname)s] %(name)s [chat=%(chat_id)s puzzle=%(puzzle_id)s] %(message)s",
                }
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "formatter": "default",
                    "filters": ["chat_puzzle"],
                    "level": level,
                }
            },
            "root": {
                "handlers": ["console"],
                "level": level,
            },
        }
    )


def get_logger(name: str) -> logging.Logger:
    """Return a logger within the project namespace."""

    if name.startswith(BASE_LOGGER_NAME):
        return logging.getLogger(name)
    return logging.getLogger(f"{BASE_LOGGER_NAME}.{name}")


@contextmanager
def logging_context(*, chat_id: int | str | None = None, puzzle_id: str | None = None) -> Iterator[None]:
    """Temporarily bind chat and puzzle identifiers to log records."""

    tokens: list[tuple[ContextVar[str], Token[str]]] = []
    if chat_id is not None:
        tokens.append((_chat_id_var, _chat_id_var.set(str(chat_id))))
    if puzzle_id is not None:
        tokens.append((_puzzle_id_var, _puzzle_id_var.set(str(puzzle_id))))
    try:
        yield
    finally:
        for var, token in reversed(tokens):
            var.reset(token)


def current_chat_id() -> str:
    """Return the chat identifier stored in the current logging context."""

    return _chat_id_var.get("-")


def current_puzzle_id() -> str:
    """Return the puzzle identifier stored in the current logging context."""

    return _puzzle_id_var.get("-")

