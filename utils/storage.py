"""Utility helpers for storing puzzles and game states on disk."""

from __future__ import annotations

import os
import tempfile
import time
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

import orjson

from utils.logging_config import get_logger

logger = get_logger("storage")

DATA_ROOT = Path("/var/data")
PUZZLES_DIR = DATA_ROOT / "puzzles"
STATES_DIR = DATA_ROOT / "states"

STATE_FILE_SUFFIX = ".json"
STATE_EXPIRATION_SECONDS = 7 * 24 * 60 * 60  # one week
STATE_CLEANUP_INTERVAL = 60 * 60  # hourly cleanup


@dataclass(slots=True)
class Player:
    """Description of an individual participant of the game."""

    user_id: int
    name: str
    dm_chat_id: int | None = None
    joined_at: float = field(default_factory=time.time)
    answers_ok: int = 0
    answers_fail: int = 0
    is_bot: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_id": self.user_id,
            "name": self.name,
            "dm_chat_id": self.dm_chat_id,
            "joined_at": self.joined_at,
            "answers_ok": self.answers_ok,
            "answers_fail": self.answers_fail,
            "is_bot": self.is_bot,
        }

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any], *, user_id: int | None = None
    ) -> "Player":
        resolved_id_raw = user_id if user_id is not None else payload.get("user_id")
        if resolved_id_raw is None:
            raise ValueError("Player payload missing user identifier")
        resolved_id = int(resolved_id_raw)
        name = str(payload.get("name", ""))
        dm_chat_raw = payload.get("dm_chat_id")
        dm_chat_id = int(dm_chat_raw) if dm_chat_raw not in (None, "") else None
        joined_raw = payload.get("joined_at")
        joined = time.time()
        if isinstance(joined_raw, (int, float)):
            joined = float(joined_raw)
        else:
            with suppress(TypeError, ValueError):
                joined = float(joined_raw)
        answers_ok = int(payload.get("answers_ok", 0) or 0)
        answers_fail = int(payload.get("answers_fail", 0) or 0)
        is_bot = bool(payload.get("is_bot", False))
        return cls(
            user_id=resolved_id,
            name=name,
            dm_chat_id=dm_chat_id,
            joined_at=joined,
            answers_ok=answers_ok,
            answers_fail=answers_fail,
            is_bot=is_bot,
        )


@dataclass(slots=True)
class GameState:
    """In-memory representation of the player progress."""

    chat_id: int
    puzzle_id: str
    filled_cells: Dict[str, str] = field(default_factory=dict)
    solved_slots: set[str] = field(default_factory=set)
    score: int = 0
    started_at: float = field(default_factory=time.time)
    last_update: float = field(default_factory=time.time)
    hinted_cells: set[str] | None = None
    puzzle_ids: list[str] | None = None
    game_id: str | None = None
    host_id: int | None = None
    mode: str = "single"
    status: str = "running"
    players: dict[int, Player] = field(default_factory=dict)
    turn_order: list[int] = field(default_factory=list)
    turn_index: int = 0
    scoreboard: dict[int, int] = field(default_factory=dict)
    hints_used: dict[str, dict[int, int]] = field(default_factory=dict)
    timer_job_id: str | None = None
    warn_job_id: str | None = None
    join_codes: dict[str, str] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    test_mode: bool = False
    dummy_user_id: int | None = None
    language: str | None = None
    theme: str | None = None

    def __post_init__(self) -> None:
        self.game_id = str(self.game_id or self.chat_id)
        self.puzzle_id = str(self.puzzle_id)
        self.chat_id = int(self.chat_id)
        self.score = int(self.score)
        self.mode = str(self.mode or "single")
        self.status = str(self.status or "running")
        if self.hinted_cells is not None and not isinstance(self.hinted_cells, set):
            self.hinted_cells = {str(item) for item in self.hinted_cells}
        if self.puzzle_ids is not None:
            self.puzzle_ids = [str(item) for item in self.puzzle_ids]
        self.filled_cells = {str(k): str(v) for k, v in dict(self.filled_cells).items()}
        self.solved_slots = {str(item) for item in set(self.solved_slots)}
        players_normalised: dict[int, Player] = {}
        for user_id, player in dict(self.players).items():
            try:
                key = int(user_id)
            except (TypeError, ValueError):
                continue
            if isinstance(player, Player):
                players_normalised[key] = player
            elif isinstance(player, Mapping):
                try:
                    players_normalised[key] = Player.from_dict(player, user_id=key)
                except Exception:
                    logger.exception("Failed to restore player %s", user_id)
            else:
                continue
        self.players = players_normalised
        turn_order_normalised: list[int] = []
        for user_id in list(self.turn_order):
            try:
                turn_order_normalised.append(int(user_id))
            except (TypeError, ValueError):
                continue
        self.turn_order = turn_order_normalised
        scoreboard_normalised: dict[int, int] = {}
        for user_id, score in dict(self.scoreboard).items():
            try:
                scoreboard_normalised[int(user_id)] = int(score)
            except (TypeError, ValueError):
                continue
        self.scoreboard = scoreboard_normalised
        hints_normalised: dict[str, dict[int, int]] = {}
        for slot_key, usage in dict(self.hints_used).items():
            slot_name = str(slot_key)
            if isinstance(usage, Mapping):
                inner: dict[int, int] = {}
                for user_id, count in usage.items():
                    try:
                        inner[int(user_id)] = int(count)
                    except (TypeError, ValueError):
                        continue
                hints_normalised[slot_name] = inner
            elif isinstance(usage, Iterable) and not isinstance(usage, (str, bytes)):
                # Legacy form as list of tuples [(user_id, count), ...]
                inner: dict[int, int] = {}
                for entry in usage:
                    try:
                        user_id, count = entry
                    except (TypeError, ValueError):
                        continue
                    try:
                        inner[int(user_id)] = int(count)
                    except (TypeError, ValueError):
                        continue
                hints_normalised[slot_name] = inner
        self.hints_used = hints_normalised
        self.timer_job_id = str(self.timer_job_id) if self.timer_job_id else None
        self.warn_job_id = str(self.warn_job_id) if self.warn_job_id else None
        self.join_codes = {
            str(code): str(target)
            for code, target in dict(self.join_codes).items()
        }
        self.host_id = int(self.host_id) if self.host_id is not None else None
        self.created_at = float(self.created_at or self.started_at)
        self.started_at = float(self.started_at)
        self.last_update = float(self.last_update)
        if self.dummy_user_id is not None:
            self.dummy_user_id = int(self.dummy_user_id)
        # Ensure scoreboard has an entry for singleplayer score when applicable.
        if self.host_id is not None and self.host_id not in self.scoreboard:
            self.scoreboard[self.host_id] = int(self.score)
        if self.language:
            self.language = str(self.language)
        if self.theme:
            self.theme = str(self.theme)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the state into a JSON-compatible dictionary."""

        return {
            "game_id": self.game_id,
            "chat_id": self.chat_id,
            "puzzle_id": self.puzzle_id,
            "puzzle_ids": list(self.puzzle_ids) if self.puzzle_ids else None,
            "filled_cells": self.filled_cells,
            "solved_slots": sorted(self.solved_slots),
            "score": self.score,
            "scoreboard": {str(k): v for k, v in self.scoreboard.items()},
            "hints_used": {
                slot: {str(user_id): count for user_id, count in users.items()}
                for slot, users in self.hints_used.items()
            },
            "started_at": self.started_at,
            "created_at": self.created_at,
            "last_update": self.last_update,
            "hinted_cells": sorted(self.hinted_cells or ()),
            "mode": self.mode,
            "status": self.status,
            "host_id": self.host_id,
            "players": {
                str(user_id): player.to_dict() for user_id, player in self.players.items()
            },
            "turn_order": [int(user_id) for user_id in self.turn_order],
            "turn_index": self.turn_index,
            "timer_job_id": self.timer_job_id,
            "warn_job_id": self.warn_job_id,
            "join_codes": self.join_codes,
            "test_mode": self.test_mode,
            "dummy_user_id": self.dummy_user_id,
            "language": self.language,
            "theme": self.theme,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "GameState":
        """Create a game state from JSON payload stored on disk."""

        solved_slots_raw = payload.get("solved_slots", [])
        solved_slots: set[str]
        if isinstance(solved_slots_raw, Iterable) and not isinstance(
            solved_slots_raw, (str, bytes)
        ):
            solved_slots = {str(item) for item in solved_slots_raw}
        else:
            solved_slots = set()

        puzzle_ids_raw = payload.get("puzzle_ids")
        puzzle_ids: list[str] | None
        if isinstance(puzzle_ids_raw, Iterable) and not isinstance(
            puzzle_ids_raw, (str, bytes)
        ):
            puzzle_ids = [str(item) for item in puzzle_ids_raw]
        elif puzzle_ids_raw:
            puzzle_ids = [str(puzzle_ids_raw)]
        else:
            puzzle_ids = None

        players_payload = payload.get("players", {})
        players: dict[int, Player] = {}
        if isinstance(players_payload, Mapping):
            for user_id_raw, player_payload in players_payload.items():
                if not isinstance(player_payload, Mapping):
                    continue
                try:
                    user_id = int(user_id_raw)
                except (TypeError, ValueError):
                    continue
                try:
                    players[user_id] = Player.from_dict(player_payload, user_id=user_id)
                except Exception:  # noqa: BLE001 - continue restoring other players
                    logger.exception("Failed to restore player %s", user_id)

        hints_payload = payload.get("hints_used", {})
        hints_used: dict[str, dict[int, int]] = {}
        if isinstance(hints_payload, Mapping):
            for slot_key, usage in hints_payload.items():
                if isinstance(usage, Mapping):
                    inner: dict[int, int] = {}
                    for user_id, count in usage.items():
                        try:
                            inner[int(user_id)] = int(count)
                        except (TypeError, ValueError):
                            continue
                    hints_used[str(slot_key)] = inner
        elif isinstance(hints_payload, (int, float)):
            # Legacy representation – store under synthetic slot/user identifiers
            hints_used["_global"] = {0: int(hints_payload)}

        scoreboard_payload = payload.get("scoreboard", {})
        scoreboard: dict[int, int] = {}
        if isinstance(scoreboard_payload, Mapping):
            for user_id_raw, score_value in scoreboard_payload.items():
                try:
                    user_id = int(user_id_raw)
                    scoreboard[user_id] = int(score_value)
                except (TypeError, ValueError):
                    continue

        mode = str(payload.get("mode", "single"))
        status = str(payload.get("status", "running"))

        game_id_raw = payload.get("game_id")
        chat_id_raw = payload.get("chat_id")
        if chat_id_raw is None:
            raise ValueError("Legacy state payload missing chat identifier")
        chat_id = int(chat_id_raw)
        game_id = str(game_id_raw) if game_id_raw else str(chat_id)

        host_raw = payload.get("host_id")
        host_id = int(host_raw) if host_raw not in (None, "") else None

        turn_order_payload = payload.get("turn_order", [])
        turn_order: list[int] = []
        if isinstance(turn_order_payload, Iterable) and not isinstance(
            turn_order_payload, (str, bytes)
        ):
            for value in turn_order_payload:
                with suppress(TypeError, ValueError):
                    turn_order.append(int(value))

        join_codes_payload = payload.get("join_codes", {})
        join_codes: dict[str, str] = {}
        if isinstance(join_codes_payload, Mapping):
            join_codes = {
                str(code): str(target) for code, target in join_codes_payload.items()
            }

        timer_job_raw = payload.get("timer_job_id")
        timer_job_id = str(timer_job_raw) if timer_job_raw else None
        warn_job_raw = payload.get("warn_job_id")
        warn_job_id = str(warn_job_raw) if warn_job_raw else None

        created_at_raw = payload.get("created_at")
        created_at = (
            float(created_at_raw)
            if isinstance(created_at_raw, (int, float))
            else float(payload.get("started_at", time.time()))
        )

        dummy_raw = payload.get("dummy_user_id")
        dummy_user_id = int(dummy_raw) if dummy_raw not in (None, "") else None

        language_raw = payload.get("language")
        language = str(language_raw) if language_raw not in (None, "") else None
        theme_raw = payload.get("theme")
        theme = str(theme_raw) if theme_raw not in (None, "") else None

        hinted_cells_raw = payload.get("hinted_cells", [])
        hinted_cells_iter: Iterable[Any]
        if isinstance(hinted_cells_raw, Iterable) and not isinstance(
            hinted_cells_raw, (str, bytes)
        ):
            hinted_cells_iter = hinted_cells_raw
        else:
            hinted_cells_iter = []
        hinted_cells = {
            str(item)
            for item in hinted_cells_iter
            if not isinstance(item, Mapping)
        }

        return cls(
            game_id=game_id,
            chat_id=chat_id,
            host_id=host_id,
            puzzle_id=str(payload["puzzle_id"]),
            puzzle_ids=puzzle_ids,
            filled_cells=dict(payload.get("filled_cells", {})),
            solved_slots=solved_slots,
            score=int(payload.get("score", 0)),
            scoreboard=scoreboard,
            hints_used=hints_used,
            started_at=float(payload.get("started_at", time.time())),
            created_at=created_at,
            last_update=float(payload.get("last_update", time.time())),
            hinted_cells=hinted_cells or None,
            players=players,
            mode=mode,
            status=status,
            turn_order=turn_order,
            turn_index=int(payload.get("turn_index", 0)),
            timer_job_id=timer_job_id,
            warn_job_id=warn_job_id,
            join_codes=join_codes,
            test_mode=bool(payload.get("test_mode", False)),
            dummy_user_id=dummy_user_id,
            language=language,
            theme=theme,
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


def delete_puzzle(puzzle_id: str) -> None:
    """Delete a stored puzzle definition if present."""

    path = PUZZLES_DIR / f"{puzzle_id}{STATE_FILE_SUFFIX}"
    with suppress(FileNotFoundError):
        path.unlink()
        logger.debug("Deleted puzzle file %s", path)


def _derive_legacy_chat_id(identifier: str | int | None) -> Optional[int]:
    if identifier is None:
        return None
    if isinstance(identifier, int):
        return identifier
    if isinstance(identifier, str) and identifier.lstrip("-").isdigit():
        with suppress(ValueError):
            return int(identifier)
    return None


def load_state(identifier: str | int) -> Optional[GameState]:
    """Load player progress from disk by game identifier."""

    game_id = str(identifier)
    primary_path = STATES_DIR / f"{game_id}{STATE_FILE_SUFFIX}"
    payload = _load_json(primary_path)
    loaded_from_legacy = False

    if payload is None:
        legacy_chat_id = _derive_legacy_chat_id(identifier)
        if legacy_chat_id is not None:
            legacy_path = STATES_DIR / f"{legacy_chat_id}{STATE_FILE_SUFFIX}"
            if legacy_path != primary_path:
                payload = _load_json(legacy_path)
                loaded_from_legacy = payload is not None

    if payload is None:
        return None

    try:
        state = GameState.from_dict(payload)
    except Exception:  # noqa: BLE001 - log and treat as missing
        logger.exception("Failed to restore game state for identifier %s", identifier)
        return None

    if loaded_from_legacy and state.game_id != game_id:
        logger.info(
            "Migrating legacy state for chat %s to new game id %s",
            state.chat_id,
            state.game_id,
        )
        save_state(state)

    logger.debug("Loaded state for game %s", state.game_id)
    return state


def save_state(state: GameState) -> None:
    """Persist player progress to disk atomically."""

    path = STATES_DIR / f"{state.game_id}{STATE_FILE_SUFFIX}"
    data = _dump_json(state.to_dict())
    _write_atomic(path, data)
    logger.debug("Saved state for game %s to %s", state.game_id, path)

    legacy_name = str(state.chat_id)
    if legacy_name != state.game_id:
        legacy_path = STATES_DIR / f"{legacy_name}{STATE_FILE_SUFFIX}"
        with suppress(FileNotFoundError):
            legacy_path.unlink()
            logger.debug("Removed legacy state file %s", legacy_path)


def delete_state(identifier: str | int | GameState) -> None:
    """Remove stored state for a game (and legacy chat file) if it exists."""

    targets: set[str] = set()
    if isinstance(identifier, GameState):
        targets.add(str(identifier.game_id))
        targets.add(str(identifier.chat_id))
    else:
        targets.add(str(identifier))
        legacy = _derive_legacy_chat_id(identifier)
        if legacy is not None:
            targets.add(str(legacy))

    for name in targets:
        path = STATES_DIR / f"{name}{STATE_FILE_SUFFIX}"
        with suppress(FileNotFoundError):
            path.unlink()
            logger.debug("Deleted state file %s", path)


def load_all_states() -> Dict[str, GameState]:
    """Load all stored game states from disk."""

    _ensure_directory(STATES_DIR)
    results: Dict[str, GameState] = {}
    for path in STATES_DIR.glob(f"*{STATE_FILE_SUFFIX}"):
        payload = _load_json(path)
        if payload is None:
            continue
        try:
            state = GameState.from_dict(payload)
        except Exception:  # noqa: BLE001 - continue loading other states
            logger.exception("Failed to parse game state from %s", path)
            continue
        existing = results.get(state.game_id)
        if existing and existing.last_update >= state.last_update:
            continue
        results[state.game_id] = state
    logger.info("Restored %s game states from disk", len(results))
    return results


def prune_expired_states(active_states: Dict[str, GameState]) -> list[GameState]:
    """Identify expired states based on last update timestamps."""

    now = time.time()
    expired: list[GameState] = []
    for game_id, game_state in list(active_states.items()):
        if now - game_state.last_update > STATE_EXPIRATION_SECONDS:
            expired.append(active_states.pop(game_id))
    return expired


__all__ = [
    "GameState",
    "Player",
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
