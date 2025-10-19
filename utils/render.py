"""Utilities for rendering crossword puzzles into PNG images."""

from __future__ import annotations

import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont

from .crossword import CompositePuzzle, Puzzle, parse_slot_public_id
from utils.logging_config import get_logger, logging_context

# ---------------------------------------------------------------------------
# Typing helpers
# ---------------------------------------------------------------------------

try:  # pragma: no cover - optional dependency for type checking only
    from utils.state import GameState  # type: ignore
except ImportError:  # pragma: no cover - provide a runtime fallback
    GameState = Any  # type: ignore


logger = get_logger("render")


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


def _parse_extended_coord(value: Any) -> Optional[Tuple[Optional[int], Tuple[int, int]]]:
    """Parse coordinates that may include a component prefix."""

    component: Optional[int] = None
    if isinstance(value, str):
        stripped = value.strip()
        if ":" in stripped:
            prefix, rest = stripped.split(":", 1)
            if prefix.strip().isdigit():
                component = int(prefix.strip())
                candidate = _parse_coord(rest)
                if candidate is not None:
                    return component, candidate
        candidate = _parse_coord(stripped)
        if candidate is not None:
            return None, candidate
        return None

    coord = _parse_coord(value)
    if coord is None:
        return None
    return component, coord


def _normalise_coord_collection(
    source: Any, *, component_filter: Optional[int] = None
) -> Sequence[Tuple[int, int]]:
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
        parsed = _parse_extended_coord(item)
        if parsed is None:
            continue
        component, coord = parsed
        if component_filter is not None and component != component_filter:
            continue
        if component_filter is None and component is not None:
            continue
        coords.append(coord)

    return coords


def _normalise_filled_cells(
    source: Any, *, component_filter: Optional[int] = None
) -> Dict[Tuple[int, int], str]:
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
        parsed = _parse_extended_coord(key)
        if parsed is None:
            continue
        component, coord = parsed
        if component_filter is not None and component != component_filter:
            continue
        if component_filter is None and component is not None:
            continue
        if value is None:
            continue
        letter = str(value).strip()
        if not letter:
            continue
        filled[coord] = letter.upper()

    return filled


def _group_solved_slots(state: GameState) -> Dict[Optional[int], set[str]]:
    """Group solved slot identifiers by component index."""

    solved_lookup: Dict[Optional[int], set[str]] = defaultdict(set)
    solved_raw = getattr(state, "solved_slots", set()) or []

    if isinstance(solved_raw, Mapping):
        entries = [str(key) for key, value in solved_raw.items() if value]
    elif isinstance(solved_raw, (list, tuple, set)):
        entries = [str(item) for item in solved_raw]
    elif solved_raw:
        entries = [str(solved_raw)]
    else:
        entries = []

    for entry in entries:
        base_id, component = parse_slot_public_id(entry)
        solved_lookup[component].add(base_id)
    return solved_lookup


def _group_hinted_cells(state: GameState) -> Dict[Optional[int], set[Tuple[int, int]]]:
    """Group hinted cell coordinates by component index."""

    hinted_lookup: Dict[Optional[int], set[Tuple[int, int]]] = defaultdict(set)
    source = (
        getattr(state, "hinted_cells", None)
        or getattr(state, "revealed_cells", None)
        or getattr(state, "revealed_letters", None)
    )
    if not source:
        return hinted_lookup

    if isinstance(source, Mapping):
        iterable: Iterable[Any] = source.keys()
    elif isinstance(source, (list, tuple, set)):
        iterable = source
    else:
        iterable = [source]

    for item in iterable:
        parsed = _parse_extended_coord(item)
        if parsed is None:
            continue
        component, coord = parsed
        hinted_lookup[component].add(coord)
    return hinted_lookup


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


def _render_single_grid_image(
    puzzle: Puzzle,
    solved_slots: set[str],
    hinted_cells: set[Tuple[int, int]],
    filled_cells: Dict[Tuple[int, int], str],
) -> Image.Image:
    """Render a single crossword grid into an image object."""

    rows = puzzle.size_rows
    cols = puzzle.size_cols
    width = cols * CELL_SIZE + PADDING * 2
    height = rows * CELL_SIZE + PADDING * 2

    number_font = _load_font(max(14, math.floor(CELL_SIZE * 0.28)) + 1)
    letter_font = _load_font(max(18, math.floor(CELL_SIZE * 0.6)), bold=True)

    cell_slots = _cell_slot_mapping(puzzle)
    cell_numbers = _cell_numbers(puzzle)

    working_filled = dict(filled_cells)
    for slot in puzzle.slots:
        if slot.slot_id not in solved_slots:
            continue
        answer = slot.answer
        if not answer:
            letters = [puzzle.cell(r, c).letter for r, c in slot.coordinates()]
            answer = "".join(letters)
        if not answer:
            continue
        for index, coord in enumerate(slot.coordinates()):
            if index >= len(answer):
                break
            working_filled[coord] = answer[index].upper()

    image = Image.new("RGB", (width, height), COLOR_BACKGROUND)
    draw = ImageDraw.Draw(image)

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
            is_solved = any(slot_id in solved_slots for slot_id in cell_slot_ids)

            background = COLOR_SOLVED if is_solved else COLOR_BACKGROUND
            if coord in hinted_cells and not is_solved:
                background = COLOR_HINT

            draw.rectangle((x0, y0, x1, y1), fill=background, outline=COLOR_GRID, width=GRID_WIDTH)

            number = cell_numbers.get(coord)
            if number:
                draw.text((x0 + 4, y0 + 2), number, font=number_font, fill=COLOR_NUMBER)

            letter = working_filled.get(coord, "")
            _draw_centered_text(draw, (x0, y0, x1, y1), letter, letter_font, COLOR_TEXT)

    return image


def render_puzzle(puzzle: Puzzle | CompositePuzzle, state: GameState) -> Path:
    """Render the crossword puzzle state into a PNG image and return its path."""

    with logging_context(puzzle_id=puzzle.id):
        start_time = perf_counter()
        output_dir = Path("/var/data/puzzles")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{puzzle.id}.png"

        if _should_use_cache(output_path, state):
            logger.debug("Using cached render for puzzle %s at %s", puzzle.id, output_path)
            return output_path

        solved_lookup = _group_solved_slots(state)
        hinted_lookup = _group_hinted_cells(state)
        filled_source = getattr(state, "filled_cells", {})

        component_keys: set[Optional[int]] = set(solved_lookup.keys()) | set(hinted_lookup.keys())
        if isinstance(puzzle, CompositePuzzle):
            component_keys.update(component.index for component in puzzle.components)
        else:
            component_keys.add(None)

        filled_lookup: Dict[Optional[int], Dict[Tuple[int, int], str]] = {
            component: _normalise_filled_cells(filled_source, component_filter=component)
            for component in component_keys
        }

        try:
            if isinstance(puzzle, CompositePuzzle):
                if not puzzle.components:
                    raise ValueError("Composite puzzle does not contain components")
                component_images: list[tuple[int, Image.Image]] = []
                ordered_components = sorted(
                    puzzle.components, key=lambda comp: (comp.row_offset, comp.index)
                )
                for component in ordered_components:
                    component_index = component.index
                    image = _render_single_grid_image(
                        component.puzzle,
                        solved_lookup.get(component_index, set()),
                        hinted_lookup.get(component_index, set()),
                        filled_lookup.get(component_index, {}),
                    )
                    component_images.append((component_index, image))

                gap = max(PADDING, CELL_SIZE // 2)
                width = max(image.width for _, image in component_images)
                height = sum(image.height for _, image in component_images)
                if len(component_images) > 1:
                    height += gap * (len(component_images) - 1)

                composite_image = Image.new("RGB", (width, height), COLOR_BACKGROUND)
                current_y = 0
                for idx, (component_index, image) in enumerate(component_images):
                    x = (width - image.width) // 2
                    composite_image.paste(image, (x, current_y))
                    logger.debug(
                        "Placed component %s at position (%s, %s) within composite image",
                        component_index,
                        x,
                        current_y,
                    )
                    current_y += image.height
                    if idx < len(component_images) - 1:
                        current_y += gap

                composite_image.save(output_path, format="PNG")
            else:
                image = _render_single_grid_image(
                    puzzle,
                    solved_lookup.get(None, set()),
                    hinted_lookup.get(None, set()),
                    filled_lookup.get(None, {}),
                )
                image.save(output_path, format="PNG")
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to render puzzle %s", puzzle.id)
            raise exc

        duration = perf_counter() - start_time
        logger.info("Rendered puzzle %s in %.3f seconds -> %s", puzzle.id, duration, output_path)
        return output_path

