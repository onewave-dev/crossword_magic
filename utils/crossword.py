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
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple

from utils.logging_config import get_logger

logger = get_logger("crossword")


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


@dataclass(slots=True)
class CompositeComponent:
    """Single connected component within a composite crossword."""

    index: int
    puzzle: Puzzle
    row_offset: int = 0
    col_offset: int = 0


@dataclass
class CompositePuzzle:
    """Container holding several disjoint crossword components."""

    id: str
    theme: str
    language: str
    components: List[CompositeComponent]
    gap_cells: int = 1
    created_at: datetime = field(default_factory=datetime.utcnow)

    @property
    def size_rows(self) -> int:
        if not self.components:
            return 0
        max_offset = 0
        for component in self.components:
            bottom = component.row_offset + component.puzzle.size_rows
            if bottom > max_offset:
                max_offset = bottom
        if self.components:
            max_offset += self.gap_cells
        return max_offset

    @property
    def size_cols(self) -> int:
        if not self.components:
            return 0
        max_offset = 0
        for component in self.components:
            right = component.col_offset + component.puzzle.size_cols
            if right > max_offset:
                max_offset = right
        if self.components:
            max_offset += self.gap_cells
        return max_offset


@dataclass(slots=True)
class SlotRef:
    """Reference to a slot optionally bound to a composite component."""

    slot: Slot
    component_index: Optional[int] = None

    @property
    def public_id(self) -> str:
        if self.component_index is None:
            return self.slot.slot_id
        return f"{self.slot.slot_id}-{self.component_index}"

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


def _apply_directional_numbering(puzzle: Puzzle) -> Dict[str, str]:
    """Assign sequential slot numbers based on row-major traversal."""

    across_slots = sorted(
        (slot for slot in puzzle.slots if slot.direction is Direction.ACROSS),
        key=lambda slot: (slot.start_row, slot.start_col),
    )
    down_slots = sorted(
        (slot for slot in puzzle.slots if slot.direction is Direction.DOWN),
        key=lambda slot: (slot.start_row, slot.start_col),
    )

    mapping: Dict[str, str] = {}

    for index, slot in enumerate(across_slots, start=1):
        new_id = f"A{index}"
        old_id = slot.slot_id
        slot.number = index
        if old_id != new_id:
            if old_id:
                mapping[old_id.upper()] = new_id.upper()
            slot.slot_id = new_id
        else:
            slot.slot_id = new_id.upper()

    for index, slot in enumerate(down_slots, start=1):
        new_id = f"D{index}"
        old_id = slot.slot_id
        slot.number = index
        if old_id != new_id:
            if old_id:
                mapping[old_id.upper()] = new_id.upper()
            slot.slot_id = new_id
        else:
            slot.slot_id = new_id.upper()

    if mapping:
        for row_cells in puzzle.grid:
            for cell in row_cells:
                if not cell.source_slots:
                    continue
                updated = {mapping.get(name.upper(), name.upper()) for name in cell.source_slots}
                cell.source_slots = {name.upper() for name in updated}

    puzzle.slots = [*across_slots, *down_slots]
    return mapping


def calculate_slots(puzzle: Puzzle) -> List[Slot]:
    """Calculate crossword slots and assign numbers for across and down directions."""

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
                    slots.append(
                        Slot(
                            slot_id="",
                            direction=Direction.ACROSS,
                            number=0,
                            start_row=row,
                            start_col=col,
                            length=length,
                        )
                    )
                col += length
            else:
                col += 1

    for row in range(puzzle.size_rows):
        for col in range(puzzle.size_cols):
            cell = puzzle.grid[row][col]
            if cell.is_block:
                continue
            is_start = row == 0 or puzzle.grid[row - 1][col].is_block
            if not is_start:
                continue
            length = 0
            while row + length < puzzle.size_rows and not puzzle.grid[row + length][col].is_block:
                length += 1
            if length > 1:
                slots.append(
                    Slot(
                        slot_id="",
                        direction=Direction.DOWN,
                        number=0,
                        start_row=row,
                        start_col=col,
                        length=length,
                    )
                )

    puzzle.slots = slots
    _apply_directional_numbering(puzzle)
    logger.debug("Calculated %d slots for puzzle %s", len(puzzle.slots), puzzle.id)
    return puzzle.slots


def renumber_slots(puzzle: Puzzle | CompositePuzzle) -> Dict[str, str]:
    """Reassign slot numbers according to the row-major rule.

    Returns a mapping from previous public identifiers (upper-case) to the
    updated identifiers. The function also refreshes ``cell.source_slots`` to
    reflect the new identifiers.
    """

    mapping: Dict[str, str] = {}

    if isinstance(puzzle, CompositePuzzle):
        for component in puzzle.components:
            component_mapping = _apply_directional_numbering(component.puzzle)
            if not component_mapping:
                continue
            for old_id, new_id in component_mapping.items():
                mapping[f"{old_id}-{component.index}".upper()] = (
                    f"{new_id}-{component.index}".upper()
                )
    else:
        mapping = _apply_directional_numbering(puzzle)

    return mapping


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


def is_composite_puzzle(obj: object) -> bool:
    """Return ``True`` if the provided object is a ``CompositePuzzle``."""

    return isinstance(obj, CompositePuzzle)


def parse_slot_public_id(slot_id: str) -> Tuple[str, Optional[int]]:
    """Split a public slot identifier into base id and component index."""

    text = slot_id.strip()
    if "-" not in text:
        return text.upper(), None
    base, suffix = text.rsplit("-", 1)
    try:
        component_index = int(suffix)
    except ValueError:
        return text.upper(), None
    return base.strip().upper(), component_index


def iter_slot_refs(puzzle: Puzzle | CompositePuzzle) -> Iterator[SlotRef]:
    """Iterate over slot references for a puzzle or composite puzzle."""

    if isinstance(puzzle, CompositePuzzle):
        for component in puzzle.components:
            for slot in component.puzzle.slots:
                yield SlotRef(slot=slot, component_index=component.index)
    else:
        for slot in puzzle.slots:
            yield SlotRef(slot=slot, component_index=None)


def find_slot_ref(puzzle: Puzzle | CompositePuzzle, slot_id: str) -> Optional[SlotRef]:
    """Locate a slot reference by its public identifier."""

    base_id, component_index = parse_slot_public_id(slot_id)
    if isinstance(puzzle, CompositePuzzle):
        for component in puzzle.components:
            if component_index is not None and component.index != component_index:
                continue
            for slot in component.puzzle.slots:
                if slot.slot_id.upper() == base_id:
                    return SlotRef(slot=slot, component_index=component.index)
    else:
        if component_index is not None:
            return None
        for slot in puzzle.slots:
            if slot.slot_id.upper() == base_id:
                return SlotRef(slot=slot, component_index=None)
    return None


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
        "kind": "puzzle",
        "id": puzzle.id,
        "theme": puzzle.theme,
        "language": puzzle.language,
        "size_rows": puzzle.size_rows,
        "size_cols": puzzle.size_cols,
        "grid": grid_payload,
        "slots": slots_payload,
        "created_at": puzzle.created_at.isoformat(),
    }


def composite_to_dict(puzzle: CompositePuzzle) -> Dict[str, object]:
    """Serialise a ``CompositePuzzle`` into JSON compatible dictionary."""

    return {
        "kind": "composite",
        "id": puzzle.id,
        "theme": puzzle.theme,
        "language": puzzle.language,
        "gap_cells": puzzle.gap_cells,
        "created_at": puzzle.created_at.isoformat(),
        "components": [
            {
                "index": component.index,
                "row_offset": component.row_offset,
                "col_offset": component.col_offset,
                "puzzle": puzzle_to_dict(component.puzzle),
            }
            for component in puzzle.components
        ],
    }


def puzzle_from_dict(payload: Dict[str, object]) -> Puzzle | CompositePuzzle:
    """Reconstruct a ``Puzzle`` or ``CompositePuzzle`` from a dictionary."""

    kind = payload.get("kind", "puzzle")
    if kind == "composite":
        return composite_from_dict(payload)

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


def composite_from_dict(payload: Dict[str, object]) -> CompositePuzzle:
    """Deserialize a ``CompositePuzzle`` from a dictionary."""

    components_payload = payload.get("components", [])
    components: List[CompositeComponent] = []
    for component_data in components_payload:
        nested_payload = dict(component_data.get("puzzle", {}))
        nested = puzzle_from_dict(nested_payload)
        if not isinstance(nested, Puzzle):
            raise ValueError("Nested composite puzzles are not supported")
        components.append(
            CompositeComponent(
                index=int(component_data["index"]),
                puzzle=nested,
                row_offset=int(component_data.get("row_offset", 0)),
                col_offset=int(component_data.get("col_offset", 0)),
            )
        )

    created_at_raw = payload.get("created_at")
    created_at = (
        datetime.fromisoformat(created_at_raw)
        if isinstance(created_at_raw, str)
        else datetime.utcnow()
    )

    return CompositePuzzle(
        id=payload["id"],
        theme=payload["theme"],
        language=payload["language"],
        components=components,
        gap_cells=int(payload.get("gap_cells", 1)),
        created_at=created_at,
    )


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
    "CompositeComponent",
    "CompositePuzzle",
    "Direction",
    "Puzzle",
    "Slot",
    "SlotRef",
    "calculate_slots",
    "composite_from_dict",
    "composite_to_dict",
    "fill_puzzle_with_words",
    "find_slot_ref",
    "is_composite_puzzle",
    "iter_slot_refs",
    "parse_slot_public_id",
    "renumber_slots",
    "puzzle_from_dict",
    "puzzle_from_json",
    "puzzle_to_dict",
    "puzzle_to_json",
]

