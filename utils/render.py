"""Utilities for rendering crossword puzzles into PNG images."""

from __future__ import annotations

import logging
import math
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont

from .crossword import Puzzle

# ---------------------------------------------------------------------------
# Typing helpers
# ---------------------------------------------------------------------------

try:  # pragma: no cover - optional dependency for type checking only
    from utils.state import GameState  # type: ignore
except ImportError:  # pragma: no cover - provide a runtime fallback
    GameState = Any  # type: ignore


logger = logging.getLogger(__name__)


CELL_SIZE = 64
PADDING = 24
GRID_WIDTH = 2

COLOR_BACKGROUND = "#FFFFFF"
COLOR_GRID = "#2C3E50"
COLOR_BLOCK = "#000000"
COLOR_TEXT = "#1C1C1C"
COLOR_NUMBER = "#5D6D7E"
COLOR_SOLVED = "#C7F5C4"
COLOR_HINT = "#F9E79F"


def _load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    """Return a PIL font object, falling back to the default bitmap font."""

    font_name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    try:
        return ImageFont.truetype(font_name, size)
    except OSError:
        logger.debug("Falling back to default PIL font for size %s", size)
        return ImageFont.load_default()


def _parse_coord(value: Any) -> Optional[Tuple[int, int]]:
    """Normalise a coordinate specification into an ``(row, col)`` tuple."""

    if isinstance(value, (tuple, list)) and len(value) >= 2:
        try:
            return int(value[0]), int(value[1])
        except (TypeError, ValueError):
            return None

    if isinstance(value, str):
        stripped = value.strip()
        for separator in (",", ":", "-", "|", ";", " "):
            if separator in stripped:
                parts = stripped.split(separator)
                if len(parts) < 2:
                    continue
                try:
                    return int(parts[0]), int(parts[1])
                except ValueError:
                    continue
        if stripped.startswith("(") and stripped.endswith(")"):
            return _parse_coord(stripped[1:-1])

    if isinstance(value, Mapping):
        row = value.get("row")
        col = value.get("col")
        if row is None or col is None:
            return None
        try:
            return int(row), int(col)
        except (TypeError, ValueError):
            return None

    return None


def _normalise_coord_collection(source: Any) -> Sequence[Tuple[int, int]]:
    """Extract a sequence of coordinates from arbitrary representations."""

    coords: list[Tuple[int, int]] = []
    if not source:
        return coords

    if isinstance(source, Mapping):
        iterable: Iterable[Any] = source.keys()
    elif isinstance(source, (list, tuple, set)):
        iterable = source
    else:
        iterable = [source]

    for item in iterable:
        coord = _parse_coord(item)
        if coord is not None:
            coords.append(coord)

    return coords


def _normalise_filled_cells(source: Any) -> Dict[Tuple[int, int], str]:
    """Convert ``state.filled_cells`` into a coordinate-to-letter mapping."""

    filled: Dict[Tuple[int, int], str] = {}
    if not source:
        return filled

    items: Iterable[Tuple[Any, Any]]
    if isinstance(source, Mapping):
        items = source.items()
    elif isinstance(source, (list, tuple, set)):
        candidate_items = []
        for entry in source:
            if isinstance(entry, Mapping):
                coord = _parse_coord(entry.get("coord"))
                if coord is None and "row" in entry and "col" in entry:
                    coord = _parse_coord((entry["row"], entry["col"]))
                if coord is None:
                    continue
                candidate_items.append((coord, entry.get("letter") or entry.get("value")))
        items = candidate_items
    else:
        items = []

    for key, value in items:
        coord = _parse_coord(key)
        if coord is None:
            continue
        if value is None:
            continue
        letter = str(value).strip()
        if not letter:
            continue
        filled[coord] = letter.upper()

    return filled


def _parse_datetime(value: Any) -> Optional[datetime]:
    """Parse a datetime representation into an aware ``datetime`` instance."""

    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value, tz=timezone.utc)
    if isinstance(value, str):
        text = value.strip()
        try:
            dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            try:
                dt = datetime.fromtimestamp(float(text), tz=timezone.utc)
            except ValueError:
                return None
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    return None


def _cell_slot_mapping(puzzle: Puzzle) -> Dict[Tuple[int, int], set[str]]:
    """Create a mapping from coordinates to the set of slot ids covering them."""

    mapping: Dict[Tuple[int, int], set[str]] = {}
    for slot in puzzle.slots:
        for coord in slot.coordinates():
            mapping.setdefault(coord, set()).add(slot.slot_id)
    return mapping


def _cell_numbers(puzzle: Puzzle) -> Dict[Tuple[int, int], str]:
    """Determine the numbering to display in the top-left corner of each cell."""

    numbers: Dict[Tuple[int, int], set[int]] = {}
    for slot in puzzle.slots:
        start = (slot.start_row, slot.start_col)
        numbers.setdefault(start, set()).add(slot.number)
    display: Dict[Tuple[int, int], str] = {}
    for coord, num_set in numbers.items():
        sorted_numbers = sorted(num_set)
        if not sorted_numbers:
            continue
        display[coord] = "/".join(str(number) for number in sorted_numbers)
    return display


def _should_use_cache(output_path: Path, state: GameState) -> bool:
    """Return ``True`` if an existing render is up-to-date for the given state."""

    if not output_path.exists():
        return False

    state_ts = _parse_datetime(getattr(state, "last_update", None))
    if state_ts is None:
        return False

    try:
        file_mtime = datetime.fromtimestamp(output_path.stat().st_mtime, tz=timezone.utc)
    except OSError:
        return False

    return file_mtime >= state_ts


def _draw_centered_text(
    draw: ImageDraw.ImageDraw,
    xy: Tuple[int, int, int, int],
    text: str,
    font: ImageFont.ImageFont,
    fill: str,
) -> None:
    """Draw text centered inside the provided bounding box."""

    if not text:
        return

    left, top, right, bottom = xy
    try:
        text_bbox = draw.textbbox((0, 0), text, font=font)
    except AttributeError:  # pragma: no cover - compatibility with older Pillow
        text_bbox = font.getbbox(text)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    x = left + (right - left - text_width) / 2
    y = top + (bottom - top - text_height) / 2
    draw.text((x, y), text, font=font, fill=fill)


def render_puzzle(puzzle: Puzzle, state: GameState) -> Path:
    """Render the crossword puzzle state into a PNG image and return its path."""

    start_time = perf_counter()
    output_dir = Path("/var/data/puzzles")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{puzzle.id}.png"

    if _should_use_cache(output_path, state):
        logger.debug("Using cached render for puzzle %s at %s", puzzle.id, output_path)
        return output_path

    rows = puzzle.size_rows
    cols = puzzle.size_cols
    width = cols * CELL_SIZE + PADDING * 2
    height = rows * CELL_SIZE + PADDING * 2

    solved_slots_raw = getattr(state, "solved_slots", set()) or []
    if isinstance(solved_slots_raw, (list, tuple, set)):
        solved_slots = {str(slot_id) for slot_id in solved_slots_raw}
    elif isinstance(solved_slots_raw, Mapping):
        solved_slots = {str(key) for key, value in solved_slots_raw.items() if value}
    else:
        solved_slots = {str(solved_slots_raw)}

    hinted_cells_source = (
        getattr(state, "hinted_cells", None)
        or getattr(state, "revealed_cells", None)
        or getattr(state, "revealed_letters", None)
    )
    hinted_cells = set(_normalise_coord_collection(hinted_cells_source))

    filled_cells = _normalise_filled_cells(getattr(state, "filled_cells", {}))

    cell_slots = _cell_slot_mapping(puzzle)
    cell_numbers = _cell_numbers(puzzle)

    for slot in puzzle.slots:
        if slot.slot_id not in solved_slots:
            continue
        if slot.answer:
            answer = slot.answer
        else:
            letters = [puzzle.cell(r, c).letter for r, c in slot.coordinates()]
            answer = "".join(letters)
        for index, coord in enumerate(slot.coordinates()):
            if not answer:
                continue
            if index >= len(answer):
                break
            filled_cells[coord] = answer[index].upper()

    number_font = _load_font(max(14, math.floor(CELL_SIZE * 0.28)))
    letter_font = _load_font(max(18, math.floor(CELL_SIZE * 0.6)), bold=True)

    image = Image.new("RGB", (width, height), COLOR_BACKGROUND)
    draw = ImageDraw.Draw(image)

    try:
        for row in range(rows):
            for col in range(cols):
                x0 = PADDING + col * CELL_SIZE
                y0 = PADDING + row * CELL_SIZE
                x1 = x0 + CELL_SIZE
                y1 = y0 + CELL_SIZE

                cell = puzzle.cell(row, col)
                if cell.is_block:
                    draw.rectangle((x0, y0, x1, y1), fill=COLOR_BLOCK)
                    continue

                coord = (row, col)
                cell_slot_ids = cell_slots.get(coord, set())
                is_solved = bool(cell_slot_ids & solved_slots)

                background = COLOR_SOLVED if is_solved else COLOR_BACKGROUND
                if coord in hinted_cells and not is_solved:
                    background = COLOR_HINT

                draw.rectangle((x0, y0, x1, y1), fill=background, outline=COLOR_GRID, width=GRID_WIDTH)

                number = cell_numbers.get(coord)
                if number:
                    draw.text((x0 + 4, y0 + 2), number, font=number_font, fill=COLOR_NUMBER)

                letter = filled_cells.get(coord, "")
                _draw_centered_text(draw, (x0, y0, x1, y1), letter, letter_font, COLOR_TEXT)

    except Exception as exc:  # noqa: BLE001 - log any rendering failure
        logger.exception("Failed to render puzzle %s", puzzle.id)
        raise exc

    try:
        image.save(output_path, format="PNG")
    except Exception as exc:  # noqa: BLE001 - log saving issues separately
        logger.exception("Failed to save rendered puzzle %s to %s", puzzle.id, output_path)
        raise exc

    duration = perf_counter() - start_time
    logger.info("Rendered puzzle %s in %.3f seconds -> %s", puzzle.id, duration, output_path)
    return output_path

