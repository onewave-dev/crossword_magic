"""Dynamic fill-in style crossword generator.

This module builds a crossword grid directly from a list of words without
relying on a pre-defined block template. Words are sorted by length, the
longest word is placed around the origin and the remaining ones are fitted
either through intersections or, when impossible, by expanding the grid
within the configured bounds.
"""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Deque, Dict, List, Sequence, Tuple

from utils.crossword import Cell, Direction, Puzzle, Slot, calculate_slots


class FillInGenerationError(RuntimeError):
    """Raised when the fill-in generator cannot place the provided words."""


class DisconnectedWordError(FillInGenerationError):
    """Raised when a word cannot be connected to the existing crossword graph."""

    def __init__(self, word: str) -> None:
        self.word = word
        super().__init__(
            f"Unable to place word '{word}' with an intersection; replacement candidates required"
        )


def _normalise_word(word: str, language: str) -> str:
    normalised = word.strip().upper()
    if language.lower() == "ru":
        normalised = normalised.replace("Ё", "Е")
    return normalised


def _extract_answer(slot: Slot, grid: List[List[Cell]]) -> str:
    letters: List[str] = []
    for row, col in slot.coordinates():
        letters.append(grid[row][col].letter)
    return "".join(letters)


def generate_fill_in_puzzle(
    puzzle_id: str,
    theme: str,
    language: str,
    words: Sequence[str],
    *,
    max_size: int = 15,
) -> Puzzle:
    """Create a crossword puzzle by dynamically arranging the provided words."""

    if max_size < 1:
        raise ValueError("max_size must be positive")

    seen: set[str] = set()
    normalised_words: List[str] = []
    for word in words:
        normalised = _normalise_word(word, language)
        if not normalised:
            continue
        if normalised in seen:
            continue
        seen.add(normalised)
        normalised_words.append(normalised)

    if not normalised_words:
        raise FillInGenerationError("No suitable words provided for generation")

    letter_frequency: Dict[str, int] = defaultdict(int)
    for word in normalised_words:
        for char in set(word):
            letter_frequency[char] += 1

    def starting_score(word: str) -> Tuple[int, int]:
        shared_letters = sum(letter_frequency[char] for char in set(word))
        return shared_letters, len(word)

    candidate_starts = sorted(normalised_words, key=starting_score, reverse=True)

    def attempt(first_word: str) -> Puzzle:
        remaining_words = [word for word in normalised_words if word != first_word]
        remaining_words.sort(key=len, reverse=True)
        words_to_process: Deque[str] = deque(remaining_words)

        grid_letters: Dict[Tuple[int, int], str] = {}
        cell_directions: Dict[Tuple[int, int], set[Direction]] = defaultdict(set)
        letter_positions: Dict[str, List[Tuple[int, int]]] = defaultdict(list)

        bounds_initialised = False
        min_row = max_row = min_col = max_col = 0

        def update_bounds(row: int, col: int) -> None:
            nonlocal bounds_initialised, min_row, max_row, min_col, max_col
            if not bounds_initialised:
                min_row = max_row = row
                min_col = max_col = col
                bounds_initialised = True
                return
            min_row = min(min_row, row)
            max_row = max(max_row, row)
            min_col = min(min_col, col)
            max_col = max(max_col, col)

        def would_exceed_bounds(
            start_row: int, start_col: int, direction: Direction, length: int
        ) -> bool:
            if not bounds_initialised:
                height = length if direction is Direction.DOWN else 1
                width = length if direction is Direction.ACROSS else 1
                return height > max_size or width > max_size

            end_row = start_row + (length - 1 if direction is Direction.DOWN else 0)
            end_col = start_col + (length - 1 if direction is Direction.ACROSS else 0)

            candidate_min_row = min(min_row, start_row)
            candidate_max_row = max(max_row, end_row)
            candidate_min_col = min(min_col, start_col)
            candidate_max_col = max(max_col, end_col)

            height = candidate_max_row - candidate_min_row + 1
            width = candidate_max_col - candidate_min_col + 1
            return height > max_size or width > max_size

        def can_place(
            word: str,
            start_row: int,
            start_col: int,
            direction: Direction,
            require_intersection: bool,
        ) -> bool:
            if would_exceed_bounds(start_row, start_col, direction, len(word)):
                return False

            intersection_found = False
            for idx, char in enumerate(word):
                row = start_row + (idx if direction is Direction.DOWN else 0)
                col = start_col + (idx if direction is Direction.ACROSS else 0)
                existing = grid_letters.get((row, col))
                if existing:
                    if existing != char:
                        return False
                    if direction in cell_directions[(row, col)]:
                        return False
                    intersection_found = True
                else:
                    if direction is Direction.ACROSS:
                        if grid_letters.get((row - 1, col)) or grid_letters.get((row + 1, col)):
                            return False
                    else:
                        if grid_letters.get((row, col - 1)) or grid_letters.get((row, col + 1)):
                            return False

                if direction is Direction.ACROSS:
                    if idx == 0 and grid_letters.get((row, col - 1)):
                        return False
                    if idx == len(word) - 1 and grid_letters.get((row, col + 1)):
                        return False
                else:
                    if idx == 0 and grid_letters.get((row - 1, col)):
                        return False
                    if idx == len(word) - 1 and grid_letters.get((row + 1, col)):
                        return False

            if require_intersection and not intersection_found:
                return False
            return True

        def place_word(word: str, start_row: int, start_col: int, direction: Direction) -> None:
            for idx, char in enumerate(word):
                row = start_row + (idx if direction is Direction.DOWN else 0)
                col = start_col + (idx if direction is Direction.ACROSS else 0)
                grid_letters[(row, col)] = char
                cell_directions[(row, col)].add(direction)
                letter_positions[char].append((row, col))
                update_bounds(row, col)

        def try_place_word(word: str) -> bool:
            # Prefer placements that intersect with existing words.
            for idx, char in enumerate(word):
                for row, col in letter_positions.get(char, []):
                    for direction in (Direction.ACROSS, Direction.DOWN):
                        if direction in cell_directions[(row, col)]:
                            continue
                        start_row = row if direction is Direction.ACROSS else row - idx
                        start_col = col - idx if direction is Direction.ACROSS else col
                        if not can_place(word, start_row, start_col, direction, True):
                            continue
                        place_word(word, start_row, start_col, direction)
                        return True
            return False

        horizontal_start_col = -(len(first_word) // 2)
        place_word(first_word, 0, horizontal_start_col, Direction.ACROSS)

        while words_to_process:
            progress_made = False
            waiting_words: Deque[str] = deque()

            while words_to_process:
                word = words_to_process.popleft()
                if try_place_word(word):
                    progress_made = True
                    continue
                waiting_words.append(word)

            if not waiting_words:
                break

            if not progress_made:
                raise DisconnectedWordError(waiting_words[0])

            words_to_process = waiting_words

        rows = max_row - min_row + 1
        cols = max_col - min_col + 1
        if rows > max_size or cols > max_size:
            raise FillInGenerationError("Generated grid exceeds allowed dimensions")

        grid: List[List[Cell]] = []
        for row in range(rows):
            row_cells: List[Cell] = []
            actual_row = row + min_row
            for col in range(cols):
                actual_col = col + min_col
                letter = grid_letters.get((actual_row, actual_col), "")
                row_cells.append(
                    Cell(
                        row=row,
                        col=col,
                        is_block=letter == "",
                        letter=letter,
                    )
                )
            grid.append(row_cells)

        puzzle = Puzzle(
            id=puzzle_id,
            theme=theme,
            language=language,
            size_rows=rows,
            size_cols=cols,
            grid=grid,
        )

        puzzle.slots = calculate_slots(puzzle)

        for row_cells in puzzle.grid:
            for cell in row_cells:
                cell.source_slots.clear()

        for slot in puzzle.slots:
            answer = _extract_answer(slot, puzzle.grid)
            slot.answer = answer
            for row, col in slot.coordinates():
                puzzle.grid[row][col].source_slots.add(slot.slot_id)

        return puzzle

    last_error: DisconnectedWordError | None = None
    for first_word in candidate_starts:
        try:
            return attempt(first_word)
        except DisconnectedWordError as exc:
            last_error = exc
            continue

    if last_error is not None:
        raise last_error
    raise FillInGenerationError("No valid crossword arrangement found")

