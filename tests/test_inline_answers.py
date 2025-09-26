"""Tests for inline answer parsing."""

import pytest

from app import _parse_inline_answer


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("A1 - Tokyo", ("A1", "Tokyo")),
        (" b12 –  Kyoto ", ("B12", "Kyoto")),
        ("c3:Rio", ("C3", "Rio")),
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
    ],
)
def test_parse_inline_answer_invalid(text):
    assert _parse_inline_answer(text) is None
