"""Tests for the dynamic fill-in crossword generator."""

from __future__ import annotations

import pytest

from utils.crossword import Direction
from utils.fill_in_generator import (
    DisconnectedWordError,
    FillInGenerationError,
    generate_fill_in_puzzle,
)


def _canonical(word: str, language: str) -> str:
    transformed = word.strip().upper()
    if language.lower() == "ru":
        transformed = transformed.replace("Ё", "Е")
    return transformed


def test_generate_fill_in_puzzle_english_words() -> None:
    words = ["asteroid", "saturn", "radar", "star", "nova"]
    puzzle = generate_fill_in_puzzle(
        puzzle_id="test",
        theme="Space",
        language="en",
        words=words,
    )

    assert puzzle.size_rows <= 15
    assert puzzle.size_cols <= 15

    answers = {slot.answer for slot in puzzle.slots}
    for word in words:
        assert _canonical(word, "en") in answers

    assert any(slot.direction is Direction.ACROSS for slot in puzzle.slots)
    assert any(slot.direction is Direction.DOWN for slot in puzzle.slots)

    for row in puzzle.grid:
        for cell in row:
            assert cell.is_block == (cell.letter == "")

    for slot in puzzle.slots:
        intersects = False
        for row, col in slot.coordinates():
            if len(puzzle.grid[row][col].source_slots) > 1:
                intersects = True
                break
        assert intersects, f"Slot {slot.slot_id} must intersect with another word"


def test_generate_fill_in_puzzle_russian_words() -> None:
    words = ["парус", "палуба", "баркас", "якорь"]
    puzzle = generate_fill_in_puzzle(
        puzzle_id="test_ru",
        theme="Море",
        language="ru",
        words=words,
    )

    answers = {slot.answer for slot in puzzle.slots}
    for word in words:
        assert _canonical(word, "ru") in answers

    assert puzzle.size_rows <= 15
    assert puzzle.size_cols <= 15


def test_generate_fill_in_puzzle_requires_intersections() -> None:
    with pytest.raises(DisconnectedWordError):
        generate_fill_in_puzzle(
            puzzle_id="disconnected",
            theme="Test",
            language="en",
            words=["aaa", "bbb", "ccc"],
        )


def test_generate_fill_in_puzzle_reorders_words_when_needed() -> None:
    words = ["OMEGA", "TURF", "FOAM"]
    puzzle = generate_fill_in_puzzle(
        puzzle_id="reordered",
        theme="Test",
        language="en",
        words=words,
    )

    answers = {slot.answer for slot in puzzle.slots}
    for word in words:
        assert _canonical(word, "en") in answers

    assert puzzle.size_rows <= 15
    assert puzzle.size_cols <= 15


def test_generate_fill_in_puzzle_backtracks_when_initial_choice_fails() -> None:
    words = ["PEAL", "SEAL", "TAPE", "TEAL", "EEL"]

    puzzle = generate_fill_in_puzzle(
        puzzle_id="backtrack",
        theme="Test",
        language="en",
        words=words,
    )

    answers = {slot.answer for slot in puzzle.slots}
    assert {_canonical(word, "en") for word in words} <= answers

    for slot in puzzle.slots:
        intersects = any(
            len(puzzle.grid[row][col].source_slots) > 1 for row, col in slot.coordinates()
        )
        assert intersects, f"Slot {slot.slot_id} must intersect with another word"


def test_generate_fill_in_puzzle_is_order_invariant() -> None:
    base_words = ["PLANET", "NEAT", "LEAP", "TONE"]
    variants = [
        base_words,
        list(reversed(base_words)),
        base_words[1:] + base_words[:1],
    ]

    expected = {_canonical(word, "en") for word in base_words}

    for words in variants:
        puzzle = generate_fill_in_puzzle(
            puzzle_id="permutation",
            theme="Permutation",
            language="en",
            words=words,
        )

        answers = {slot.answer for slot in puzzle.slots}
        assert expected <= answers


def test_generate_fill_in_puzzle_respects_max_size() -> None:
    with pytest.raises(FillInGenerationError):
        generate_fill_in_puzzle(
            puzzle_id="tiny",
            theme="Test",
            language="en",
            words=["alphabet", "letters", "symbols"],
            max_size=4,
        )
