"""Utility helpers for crossword puzzle generation and storage.

This module provides dataclasses that mirror the structure described in
``AGENTS.md`` together with helper functions for numbering crossword slots,
serialising puzzles and a simple backtracking based filling algorithm.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json
import logging
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple


logger = logging.getLogger(__name__)


class Direction(str, Enum):
    """Enumeration of crossword slot directions."""

    ACROSS = "across"
    DOWN = "down"


@dataclass(slots=True)
class Cell:
    """Represents a single cell of a crossword grid."""

    row: int
    col: int
    is_block: bool = False
    letter: str = ""
    source_slots: Set[str] = field(default_factory=set, repr=False)

    def clear_if_unused(self) -> None:
        """Reset the letter if no slots reference the cell."""

        if not self.source_slots:
            self.letter = ""


@dataclass(slots=True)
class Slot:
    """Crossword slot metadata following the structure from ``AGENTS.md``."""

    slot_id: str
    direction: Direction
    number: int
    start_row: int
    start_col: int
    length: int
    clue: str = ""
    answer: Optional[str] = None

    def coordinates(self) -> Iterator[Tuple[int, int]]:
        """Yield coordinates for every cell covered by the slot."""

        for offset in range(self.length):
            if self.direction is Direction.ACROSS:
                yield self.start_row, self.start_col + offset
            else:
                yield self.start_row + offset, self.start_col


@dataclass
class Puzzle:
    """Dataclass mirroring the crossword puzzle structure."""

    id: str
    theme: str
    language: str
    size_rows: int
    size_cols: int
    grid: List[List[Cell]]
    slots: List[Slot] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)

    @classmethod
    def from_size(
        cls,
        puzzle_id: str,
        theme: str,
        language: str,
        rows: int,
        cols: int,
        block_positions: Optional[Iterable[Tuple[int, int]]] = None,
    ) -> "Puzzle":
        """Create an empty puzzle with the specified size and block positions."""

        block_set = {(r, c) for r, c in block_positions or ()}
        grid: List[List[Cell]] = []
        for row in range(rows):
            row_cells = []
            for col in range(cols):
                row_cells.append(Cell(row=row, col=col, is_block=(row, col) in block_set))
            grid.append(row_cells)
        puzzle = cls(
            id=puzzle_id,
            theme=theme,
            language=language,
            size_rows=rows,
            size_cols=cols,
            grid=grid,
        )
        puzzle.slots = calculate_slots(puzzle)
        return puzzle

    def cell(self, row: int, col: int) -> Cell:
        """Return the cell at the provided coordinates."""

        return self.grid[row][col]


DEFAULT_FILLER_WORDS: Dict[str, Sequence[str]] = {
    "en": (
        "AREA",
        "ABLE",
        "NOTE",
        "TREE",
        "STAR",
        "RING",
        "SHARE",
        "ALARM",
        "PLAN",
        "SILK",
        "IDEA",
        "RIVER",
        "STONE",
        "PHOTO",
        "LIGHT",
        "VALUE",
    ),
    "ru": (
        "ЛЕС",
        "ГОРОД",
        "КОД",
        "ПЛАН",
        "СЦЕНА",
        "КАФЕ",
        "КЛУБ",
        "ОЗЕРО",
        "СИГНАЛ",
        "СФЕРА",
    ),
}


def calculate_slots(puzzle: Puzzle) -> List[Slot]:
    """Calculate crossword slots and assign numbers for across and down directions."""

    across_number = 1
    down_number = 1
    slots: List[Slot] = []

    for row in range(puzzle.size_rows):
        col = 0
        while col < puzzle.size_cols:
            cell = puzzle.grid[row][col]
            if cell.is_block:
                col += 1
                continue
            is_start = col == 0 or puzzle.grid[row][col - 1].is_block
            if is_start:
                length = 0
                while col + length < puzzle.size_cols and not puzzle.grid[row][col + length].is_block:
                    length += 1
                if length > 1:
                    slot = Slot(
                        slot_id=f"A{across_number}",
                        direction=Direction.ACROSS,
                        number=across_number,
                        start_row=row,
                        start_col=col,
                        length=length,
                    )
                    slots.append(slot)
                    across_number += 1
                col += length
            else:
                col += 1

    for col in range(puzzle.size_cols):
        row = 0
        while row < puzzle.size_rows:
            cell = puzzle.grid[row][col]
            if cell.is_block:
                row += 1
                continue
            is_start = row == 0 or puzzle.grid[row - 1][col].is_block
            if is_start:
                length = 0
                while row + length < puzzle.size_rows and not puzzle.grid[row + length][col].is_block:
                    length += 1
                if length > 1:
                    slot = Slot(
                        slot_id=f"D{down_number}",
                        direction=Direction.DOWN,
                        number=down_number,
                        start_row=row,
                        start_col=col,
                        length=length,
                    )
                    slots.append(slot)
                    down_number += 1
                row += length
            else:
                row += 1

    slots.sort(key=lambda s: (s.direction.value, s.number))
    logger.debug("Calculated %d slots for puzzle %s", len(slots), puzzle.id)
    return slots


def _normalise_word(word: str, language: str) -> str:
    """Normalise a candidate word for placement."""

    normalised = word.strip().replace(" ", "").upper()
    if language.lower() == "ru":
        normalised = normalised.replace("Ё", "Е")
    return normalised


def _candidate_pool(
    words: Iterable[str],
    language: str,
) -> Dict[int, List[str]]:
    pool: Dict[int, List[str]] = {}
    for word in words:
        normalised = _normalise_word(word, language)
        if not normalised:
            continue
        pool.setdefault(len(normalised), []).append(normalised)
    return pool


def fill_puzzle_with_words(
    puzzle: Puzzle,
    thematic_words: Sequence[str],
    filler_words: Optional[Sequence[str]] = None,
) -> bool:
    """Fill the crossword puzzle using a backtracking search.

    The algorithm prioritises the provided thematic words and falls back to a
    language specific filler pool supplemented by ``filler_words``. The
    function mutates the puzzle grid and slots with the found answers.
    """

    filler_source = list(filler_words or [])
    default_pool = DEFAULT_FILLER_WORDS.get(puzzle.language.lower(), ())
    filler_source.extend(default_pool)

    thematic_pool = _candidate_pool(thematic_words, puzzle.language)
    filler_pool = _candidate_pool(filler_source, puzzle.language)

    puzzle.slots = calculate_slots(puzzle)
    slots = sorted(puzzle.slots, key=lambda s: (-s.length, s.direction.value, s.number))
    assignment: Dict[str, str] = {}

    def fits(slot: Slot, word: str) -> bool:
        for idx, (row, col) in enumerate(slot.coordinates()):
            cell = puzzle.cell(row, col)
            if cell.is_block:
                logger.debug(
                    "Encountered block when checking slot %s at %s", slot.slot_id, (row, col)
                )
                return False
            if cell.letter and cell.letter != word[idx]:
                logger.debug(
                    "Conflict for slot %s with word %s at %s: %s != %s",
                    slot.slot_id,
                    word,
                    (row, col),
                    cell.letter,
                    word[idx],
                )
                return False
        return True

    def place(slot: Slot, word: str) -> None:
        logger.debug("Placing word %s into slot %s", word, slot.slot_id)
        assignment[slot.slot_id] = word
        slot.answer = word
        for idx, (row, col) in enumerate(slot.coordinates()):
            cell = puzzle.cell(row, col)
            cell.letter = word[idx]
            cell.source_slots.add(slot.slot_id)

    def remove(slot: Slot) -> None:
        logger.debug("Removing word from slot %s", slot.slot_id)
        assignment.pop(slot.slot_id, None)
        slot.answer = None
        for row, col in slot.coordinates():
            cell = puzzle.cell(row, col)
            cell.source_slots.discard(slot.slot_id)
            if not cell.source_slots:
                cell.letter = ""

    def candidates(slot: Slot) -> List[str]:
        thematic_candidates = [
            word
            for word in thematic_pool.get(slot.length, [])
            if word not in assignment.values()
        ]
        filler_candidates = [
            word
            for word in filler_pool.get(slot.length, [])
            if word not in assignment.values()
        ]
        return thematic_candidates + filler_candidates

    def backtrack(index: int) -> bool:
        if index >= len(slots):
            return True
        slot = slots[index]
        for word in candidates(slot):
            if not fits(slot, word):
                continue
            place(slot, word)
            if backtrack(index + 1):
                return True
            logger.debug("Backtracking from slot %s with word %s", slot.slot_id, word)
            remove(slot)
        return False

    success = backtrack(0)
    if not success:
        logger.warning("Failed to fill puzzle %s with provided words", puzzle.id)
    else:
        logger.info("Successfully filled puzzle %s", puzzle.id)
    return success


def puzzle_to_dict(puzzle: Puzzle) -> Dict[str, object]:
    """Convert a ``Puzzle`` instance into a serialisable dictionary."""

    grid_payload: List[List[Dict[str, object]]] = []
    for row in puzzle.grid:
        row_payload = []
        for cell in row:
            row_payload.append(
                {
                    "row": cell.row,
                    "col": cell.col,
                    "is_block": cell.is_block,
                    "letter": cell.letter,
                }
            )
        grid_payload.append(row_payload)

    slots_payload = []
    for slot in puzzle.slots:
        slots_payload.append(
            {
                "slot_id": slot.slot_id,
                "direction": slot.direction.value,
                "number": slot.number,
                "start_row": slot.start_row,
                "start_col": slot.start_col,
                "length": slot.length,
                "clue": slot.clue,
                "answer": slot.answer,
            }
        )

    return {
        "id": puzzle.id,
        "theme": puzzle.theme,
        "language": puzzle.language,
        "size_rows": puzzle.size_rows,
        "size_cols": puzzle.size_cols,
        "grid": grid_payload,
        "slots": slots_payload,
        "created_at": puzzle.created_at.isoformat(),
    }


def puzzle_from_dict(payload: Dict[str, object]) -> Puzzle:
    """Reconstruct a ``Puzzle`` instance from a dictionary."""

    rows = payload["size_rows"]
    cols = payload["size_cols"]
    grid_payload = payload["grid"]
    grid: List[List[Cell]] = []
    for row in range(rows):
        row_cells = []
        for col in range(cols):
            cell_data = grid_payload[row][col]
            row_cells.append(
                Cell(
                    row=cell_data["row"],
                    col=cell_data["col"],
                    is_block=cell_data["is_block"],
                    letter=cell_data.get("letter", ""),
                )
            )
        grid.append(row_cells)

    slots_payload = payload.get("slots", [])
    slots: List[Slot] = []
    for slot_data in slots_payload:
        slot = Slot(
            slot_id=slot_data["slot_id"],
            direction=Direction(slot_data["direction"]),
            number=slot_data["number"],
            start_row=slot_data["start_row"],
            start_col=slot_data["start_col"],
            length=slot_data["length"],
            clue=slot_data.get("clue", ""),
            answer=slot_data.get("answer"),
        )
        if slot.answer:
            for idx, (row, col) in enumerate(slot.coordinates()):
                cell = grid[row][col]
                cell.letter = slot.answer[idx]
                cell.source_slots.add(slot.slot_id)
        slots.append(slot)

    created_at_raw = payload.get("created_at")
    created_at = (
        datetime.fromisoformat(created_at_raw)
        if isinstance(created_at_raw, str)
        else datetime.utcnow()
    )

    puzzle = Puzzle(
        id=payload["id"],
        theme=payload["theme"],
        language=payload["language"],
        size_rows=rows,
        size_cols=cols,
        grid=grid,
        slots=slots,
        created_at=created_at,
    )
    return puzzle


def puzzle_to_json(puzzle: Puzzle, *, ensure_ascii: bool = False) -> str:
    """Serialise the puzzle to a JSON string."""

    payload = puzzle_to_dict(puzzle)
    return json.dumps(payload, ensure_ascii=ensure_ascii, separators=(",", ":"))


def puzzle_from_json(data: str) -> Puzzle:
    """Deserialize a puzzle from a JSON string."""

    payload = json.loads(data)
    return puzzle_from_dict(payload)


__all__ = [
    "Cell",
    "Direction",
    "Puzzle",
    "Slot",
    "calculate_slots",
    "fill_puzzle_with_words",
    "puzzle_from_dict",
    "puzzle_from_json",
    "puzzle_to_dict",
    "puzzle_to_json",
]

