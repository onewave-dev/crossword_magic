"""Utility helpers for storing puzzles and game states on disk."""

from __future__ import annotations

import logging
import os
import tempfile
import time
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

import orjson

logger = logging.getLogger(__name__)

DATA_ROOT = Path("/var/data")
PUZZLES_DIR = DATA_ROOT / "puzzles"
STATES_DIR = DATA_ROOT / "states"

STATE_FILE_SUFFIX = ".json"
STATE_EXPIRATION_SECONDS = 7 * 24 * 60 * 60  # one week
STATE_CLEANUP_INTERVAL = 60 * 60  # hourly cleanup


@dataclass(slots=True)
class GameState:
    """In-memory representation of the player progress."""

    chat_id: int
    puzzle_id: str
    filled_cells: Dict[str, str]
    solved_slots: set[str]
    score: int
    hints_used: int
    started_at: float
    last_update: float
    hinted_cells: set[str] | None = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the state into a JSON-compatible dictionary."""

        return {
            "chat_id": self.chat_id,
            "puzzle_id": self.puzzle_id,
            "filled_cells": self.filled_cells,
            "solved_slots": sorted(self.solved_slots),
            "score": self.score,
            "hints_used": self.hints_used,
            "started_at": self.started_at,
            "last_update": self.last_update,
            "hinted_cells": sorted(self.hinted_cells or ()),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "GameState":
        """Create a game state from JSON payload stored on disk."""

        solved_slots_raw = payload.get("solved_slots", [])
        solved_slots: set[str]
        if isinstance(solved_slots_raw, Iterable) and not isinstance(solved_slots_raw, (str, bytes)):
            solved_slots = {str(item) for item in solved_slots_raw}
        else:
            solved_slots = set()

        return cls(
            chat_id=int(payload["chat_id"]),
            puzzle_id=str(payload["puzzle_id"]),
            filled_cells=dict(payload.get("filled_cells", {})),
            solved_slots=solved_slots,
            score=int(payload.get("score", 0)),
            hints_used=int(payload.get("hints_used", 0)),
            started_at=float(payload.get("started_at", time.time())),
            last_update=float(payload.get("last_update", time.time())),
            hinted_cells={str(item) for item in payload.get("hinted_cells", [])},
        )


def _ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _write_atomic(path: Path, data: bytes) -> None:
    _ensure_directory(path.parent)
    with tempfile.NamedTemporaryFile("wb", delete=False, dir=path.parent) as tmp_file:
        tmp_file.write(data)
        tmp_file.flush()
        os.fsync(tmp_file.fileno())
        temp_path = Path(tmp_file.name)
    os.replace(temp_path, path)
    logger.debug("Atomic write complete for %s", path)


def _load_json(path: Path) -> Optional[Mapping[str, Any]]:
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        logger.debug("Requested JSON file %s does not exist", path)
        return None
    except OSError:
        logger.exception("Failed to read JSON file at %s", path)
        return None

    try:
        return orjson.loads(raw)
    except orjson.JSONDecodeError:
        logger.exception("Failed to decode JSON file at %s", path)
        return None


def _dump_json(payload: Mapping[str, Any]) -> bytes:
    return orjson.dumps(payload)


def load_puzzle(puzzle_id: str) -> Optional[Mapping[str, Any]]:
    """Load puzzle definition from persistent storage."""

    path = PUZZLES_DIR / f"{puzzle_id}{STATE_FILE_SUFFIX}"
    payload = _load_json(path)
    if payload is None:
        return None
    logger.debug("Loaded puzzle %s from %s", puzzle_id, path)
    return payload


def save_puzzle(puzzle_id: str, payload: Mapping[str, Any]) -> None:
    """Persist puzzle definition to disk atomically."""

    path = PUZZLES_DIR / f"{puzzle_id}{STATE_FILE_SUFFIX}"
    data = _dump_json(payload)
    _write_atomic(path, data)
    logger.debug("Saved puzzle %s to %s", puzzle_id, path)


def load_state(chat_id: int) -> Optional[GameState]:
    """Load player progress from disk."""

    path = STATES_DIR / f"{chat_id}{STATE_FILE_SUFFIX}"
    payload = _load_json(path)
    if payload is None:
        return None
    try:
        state = GameState.from_dict(payload)
    except Exception:  # noqa: BLE001 - log and treat as missing
        logger.exception("Failed to restore game state for chat %s", chat_id)
        return None
    logger.debug("Loaded state for chat %s from %s", chat_id, path)
    return state


def save_state(state: GameState) -> None:
    """Persist player progress to disk atomically."""

    path = STATES_DIR / f"{state.chat_id}{STATE_FILE_SUFFIX}"
    data = _dump_json(state.to_dict())
    _write_atomic(path, data)
    logger.debug("Saved state for chat %s to %s", state.chat_id, path)


def delete_state(chat_id: int) -> None:
    """Remove stored state for a chat if it exists."""

    path = STATES_DIR / f"{chat_id}{STATE_FILE_SUFFIX}"
    with suppress(FileNotFoundError):
        path.unlink()
        logger.debug("Deleted state file for chat %s", chat_id)


def load_all_states() -> Dict[int, GameState]:
    """Load all stored game states from disk."""

    _ensure_directory(STATES_DIR)
    results: Dict[int, GameState] = {}
    for path in STATES_DIR.glob(f"*{STATE_FILE_SUFFIX}"):
        try:
            chat_id = int(path.stem)
        except ValueError:
            logger.warning("Skipping state file with non-integer name: %s", path.name)
            continue
        state = load_state(chat_id)
        if state is not None:
            results[chat_id] = state
    logger.info("Restored %s game states from disk", len(results))
    return results


def prune_expired_states(active_states: Dict[int, GameState]) -> list[int]:
    """Delete expired states based on last update timestamps."""

    now = time.time()
    expired: list[int] = []
    for chat_id, game_state in list(active_states.items()):
        if now - game_state.last_update > STATE_EXPIRATION_SECONDS:
            expired.append(chat_id)
            delete_state(chat_id)
            del active_states[chat_id]
            logger.info("Expired state for chat %s removed", chat_id)
    return expired


__all__ = [
    "GameState",
    "STATE_CLEANUP_INTERVAL",
    "STATE_EXPIRATION_SECONDS",
    "delete_state",
    "load_all_states",
    "load_puzzle",
    "load_state",
    "prune_expired_states",
    "save_puzzle",
    "save_state",
]
