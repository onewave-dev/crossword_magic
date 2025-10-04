"""Tests for clue formatting helpers in :mod:`app`."""

from utils.crossword import Cell, Direction, Puzzle, Slot, SlotRef

from app import _format_clue_section, _format_clues_message


def _build_puzzle_with_slots() -> Puzzle:
    grid = [[Cell(row=0, col=0), Cell(row=0, col=1)]]
    across_slot = Slot(
        slot_id="A1",
        direction=Direction.ACROSS,
        number=1,
        start_row=0,
        start_col=0,
        length=2,
        clue="Across clue",
        answer="AA",
    )
    down_slot = Slot(
        slot_id="D2",
        direction=Direction.DOWN,
        number=2,
        start_row=0,
        start_col=1,
        length=2,
        clue="Down clue",
        answer="DD",
    )
    return Puzzle(
        id="p1",
        theme="test",
        language="en",
        size_rows=1,
        size_cols=2,
        grid=grid,
        slots=[across_slot, down_slot],
    )


def test_format_clue_section_omits_length_hint() -> None:
    slot = Slot(
        slot_id="A1",
        direction=Direction.ACROSS,
        number=1,
        start_row=0,
        start_col=0,
        length=5,
        clue="Example clue",
        answer="ABCDE",
    )
    ref = SlotRef(slot=slot)

    formatted = _format_clue_section([ref])

    assert formatted == "A1: Example clue"


def test_format_clues_message_without_length_hint() -> None:
    puzzle = _build_puzzle_with_slots()

    message = _format_clues_message(puzzle)

    assert "(2)" not in message
    assert message == "Across:\nA1: Across clue\n\nDown:\nD2: Down clue"
