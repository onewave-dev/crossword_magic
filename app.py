"""FastAPI application entrypoint for Telegram webhook processing."""

from __future__ import annotations

import asyncio
import html
import random
import re
import os
import time
from uuid import uuid4
from contextlib import suppress
from functools import wraps
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from telegram import (
    Chat,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
    Update,
    constants,
)
from telegram.constants import ChatType
from telegram.ext import (
    Application,
    CallbackContext,
    CallbackQueryHandler,
    CommandHandler,
    ConversationHandler,
    ContextTypes,
    MessageHandler,
    filters,
)
from telegram.request import HTTPXRequest

from utils.storage import (
    STATE_CLEANUP_INTERVAL,
    GameState,
    Player,
    delete_puzzle,
    delete_state,
    load_puzzle,
    load_all_states,
    load_state,
    prune_expired_states,
    save_puzzle,
    save_state,
)
from utils.crossword import (
    CompositeComponent,
    CompositePuzzle,
    Direction,
    Puzzle,
    SlotRef,
    composite_to_dict,
    find_slot_ref,
    iter_slot_refs,
    parse_slot_public_id,
    puzzle_from_dict,
    puzzle_to_dict,
)
from utils.fill_in_generator import (
    DisconnectedWordError,
    FillInGenerationError,
    generate_fill_in_puzzle,
)
from utils.llm_generator import WordClue, generate_clues
from utils.render import render_puzzle
from utils.validators import WordValidationError, validate_word_list

from utils.logging_config import configure_logging, get_logger, logging_context

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------

configure_logging(os.getenv("LOG_LEVEL", "INFO"))
logger = get_logger("app")


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class Settings:
    """Container for application environment variables."""

    telegram_bot_token: str
    public_url: str
    webhook_secret: str
    webhook_path: str = "/webhook"
    webhook_check_interval: int = 300
    admin_id: Optional[int] = None


def load_settings() -> Settings:
    """Load and validate required settings from environment variables."""

    required_vars = {
        "TELEGRAM_BOT_TOKEN": os.getenv("TELEGRAM_BOT_TOKEN"),
        "PUBLIC_URL": os.getenv("PUBLIC_URL"),
        "WEBHOOK_SECRET": os.getenv("WEBHOOK_SECRET"),
        "WEBHOOK_PATH": os.getenv("WEBHOOK_PATH", "/webhook"),
    }

    missing = [name for name, value in required_vars.items() if not value]
    if missing:
        logger.critical("Missing required environment variables: %s", ", ".join(missing))
        raise RuntimeError(f"Missing required environment variables: {', '.join(missing)}")

    logger.debug("Loaded environment variables: %s", {k: v for k, v in required_vars.items() if k != "TELEGRAM_BOT_TOKEN"})

    interval_raw = os.getenv("WEBHOOK_CHECK_INTERVAL", "300")
    check_interval = 300
    with suppress(ValueError):
        check_interval = max(int(interval_raw), 60)
    if not interval_raw.isdigit():
        logger.debug("WEBHOOK_CHECK_INTERVAL is not a digit, defaulting to 300 seconds")
        check_interval = 300

    admin_id_raw = os.getenv("ADMIN_ID")
    admin_id: Optional[int] = None
    if admin_id_raw:
        try:
            admin_id = int(admin_id_raw)
        except ValueError:
            logger.warning("Invalid ADMIN_ID provided, ignoring value: %s", admin_id_raw)
            admin_id = None

    return Settings(
        telegram_bot_token=required_vars["TELEGRAM_BOT_TOKEN"],
        public_url=required_vars["PUBLIC_URL"].rstrip("/"),
        webhook_secret=required_vars["WEBHOOK_SECRET"],
        webhook_path=required_vars["WEBHOOK_PATH"] if required_vars["WEBHOOK_PATH"].startswith("/") else f"/{required_vars['WEBHOOK_PATH']}",
        webhook_check_interval=check_interval,
        admin_id=admin_id,
    )


# ---------------------------------------------------------------------------
# Directories for storage
# ---------------------------------------------------------------------------


def ensure_storage_directories() -> None:
    """Ensure that persistent storage directories exist."""

    for path in ("/var/data/puzzles", "/var/data/states"):
        os.makedirs(path, exist_ok=True)
        logger.debug("Ensured storage directory exists: %s", path)


# ---------------------------------------------------------------------------
# FastAPI application and telegram application state
# ---------------------------------------------------------------------------


app = FastAPI()


class AppState:
    """Shared state container for the FastAPI application."""

    def __init__(self) -> None:
        self.settings: Optional[Settings] = None
        self.telegram_app: Optional[Application] = None
        self.webhook_task: Optional[asyncio.Task[None]] = None
        self.cleanup_task: Optional[asyncio.Task[None]] = None
        self.active_games: dict[str, GameState] = {}
        self.chat_to_game: dict[int, str] = {}
        self.player_chats: dict[int, int] = {}
        self.join_codes: dict[str, str] = {}
        self.generating_chats: set[int] = set()


state = AppState()


def get_telegram_application() -> Application:
    if state.telegram_app is None:
        logger.error("Telegram application is not initialized")
        raise HTTPException(status_code=503, detail="Telegram application is not initialized")
    return state.telegram_app


def _cleanup_chat_resources(chat_id: int, puzzle_id: str | None = None) -> None:
    """Remove in-memory and persisted resources for the provided chat."""

    game_id = state.chat_to_game.get(chat_id)
    if game_id:
        game_state = state.active_games.get(game_id)
        if game_state is not None:
            _cleanup_game_state(game_state)
            return
        with logging_context(chat_id=chat_id, puzzle_id=puzzle_id):
            state.generating_chats.discard(chat_id)
            state.chat_to_game.pop(chat_id, None)
            delete_state(game_id)
            for code, target in list(state.join_codes.items()):
                if target == game_id:
                    state.join_codes.pop(code, None)
            if puzzle_id:
                delete_puzzle(puzzle_id)
            logger.info("Cleaned up resources for chat %s", chat_id)
            return

    with logging_context(chat_id=chat_id, puzzle_id=puzzle_id):
        state.generating_chats.discard(chat_id)
        state.chat_to_game.pop(chat_id, None)
        delete_state(chat_id)
        for code, target in list(state.join_codes.items()):
            if target == str(chat_id):
                state.join_codes.pop(code, None)
        if puzzle_id:
            delete_puzzle(puzzle_id)
        logger.info("Cleaned up resources for chat %s", chat_id)


def _cleanup_game_state(game_state: GameState | None) -> None:
    if game_state is None:
        return
    with logging_context(chat_id=game_state.chat_id, puzzle_id=game_state.puzzle_id):
        state.generating_chats.discard(game_state.chat_id)
        state.chat_to_game.pop(game_state.chat_id, None)
        state.active_games.pop(game_state.game_id, None)
        delete_state(game_state)
        for code, target in list(state.join_codes.items()):
            if target == game_state.game_id:
                state.join_codes.pop(code, None)
        for user_id, mapped_chat in list(state.player_chats.items()):
            if mapped_chat == game_state.chat_id:
                state.player_chats.pop(user_id, None)
        if game_state.puzzle_id:
            delete_puzzle(game_state.puzzle_id)
        logger.info("Cleaned up resources for game %s", game_state.game_id)


def command_entrypoint(fallback=None):
    """Decorator for command handlers providing logging context and error handling."""

    def decorator(func):
        @wraps(func)
        async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE, *args, **kwargs):
            chat = update.effective_chat if update else None
            chat_id = chat.id if chat else None
            with logging_context(chat_id=chat_id):
                try:
                    return await func(update, context, *args, **kwargs)
                except Exception:  # noqa: BLE001 - ensure all exceptions are logged
                    logger.exception("Unhandled error in command %s", getattr(func, "__name__", "<unknown>"))
                    message = update.effective_message if update else None
                    if message is not None:
                        await message.reply_text(
                            "Произошла временная ошибка. Пожалуйста, попробуйте позже."
                        )
                    return fallback

        return wrapper

    return decorator


def register_webhook_route(path: str) -> None:
    """Register the webhook endpoint for the configured path."""

    router = app.router
    for route in list(router.routes):
        if getattr(route, "endpoint", None) is telegram_webhook:
            logger.debug("Removing existing webhook route bound to %s", getattr(route, "path", "<unknown>"))
            router.routes.remove(route)

    logger.debug("Registering webhook route at path %s", path)
    router.add_api_route(path, telegram_webhook, methods=["POST"], name="telegram_webhook")


# ---------------------------------------------------------------------------
# Game helpers and conversation configuration
# ---------------------------------------------------------------------------


LANGUAGE_STATE, THEME_STATE = range(2)

MODE_IDLE = "idle"
MODE_AWAIT_LANGUAGE = "await_language"
MODE_AWAIT_THEME = "await_theme"
MODE_IN_GAME = "in_game"

REMINDER_DELAY_SECONDS = 10 * 60

MAX_PUZZLE_SIZE = 15
MAX_REPLACEMENT_REQUESTS = 30

TURN_DURATION_SECONDS = 45
TURN_WARNING_SECONDS = 15
TOTAL_GAME_DURATION_SECONDS = 10 * 60
GAME_WARNING_SECONDS = 60
TURN_HINT_PENALTY = 1

TURN_TIMEOUT_JOB_PREFIX = "turn-timeout"
TURN_WARN_JOB_PREFIX = "turn-warn"
GAME_TIMEOUT_JOB_PREFIX = "game-timeout"

TURN_SELECT_CALLBACK = "turn:select"
TURN_SLOT_CALLBACK_PREFIX = "turn:slot:"
LOBBY_START_CALLBACK = "lobby:start"


def get_chat_mode(context: ContextTypes.DEFAULT_TYPE) -> str:
    """Return the current chat mode stored in chat_data."""

    chat_data = getattr(context, "chat_data", None)
    if not isinstance(chat_data, dict):
        chat_data = {}
        setattr(context, "chat_data", chat_data)
    return chat_data.get("chat_mode", MODE_IDLE)


def set_chat_mode(context: ContextTypes.DEFAULT_TYPE, mode: str) -> None:
    """Persist the provided chat mode in chat_data."""

    chat_data = getattr(context, "chat_data", None)
    if not isinstance(chat_data, dict):
        chat_data = {}
        setattr(context, "chat_data", chat_data)
    if mode == MODE_IDLE:
        chat_data.pop("chat_mode", None)
    else:
        chat_data["chat_mode"] = mode


def is_chat_mode_set(context: ContextTypes.DEFAULT_TYPE) -> bool:
    """Return True if a chat mode has been explicitly stored for the chat."""

    chat_data = getattr(context, "chat_data", None)
    return isinstance(chat_data, dict) and "chat_mode" in chat_data


def _normalise_thread_id(update: Update) -> int:
    message = update.effective_message
    thread_id = 0
    if message is not None and message.message_thread_id is not None:
        thread_id = message.message_thread_id
    logger.debug(
        "Normalised thread id for chat %s: %s",
        update.effective_chat.id if update.effective_chat else "<unknown>",
        thread_id,
    )
    return thread_id


def _coord_key(row: int, col: int, component: int | None = None) -> str:
    base = f"{row},{col}"
    if component is None:
        return base
    return f"{component}:{base}"


CYRILLIC_SLOT_LETTER_MAP = str.maketrans({"А": "A", "Д": "D"})


def _normalise_slot_id(slot_id: str) -> str:
    """Normalise slot identifiers to a canonical ASCII form."""

    return slot_id.strip().upper().translate(CYRILLIC_SLOT_LETTER_MAP)


INLINE_ANSWER_PATTERN = re.compile(
    # Accept common dash-like separators (hyphen-minus, hyphen, non-breaking hyphen, en/em dash, figure dash, minus) and colon
    # or just whitespace between slot and answer. Allow slot identifiers consisting solely of digits.
    r"^\s*([^\W\d_]*[0-9]+(?:-[0-9]+)?)\s*(?:[-‐‑–—‒−:]\s*|\s+)(.+)$",
    flags=re.UNICODE,
)

COMPLETION_CALLBACK_PREFIX = "complete:"
SAME_TOPIC_CALLBACK_PREFIX = f"{COMPLETION_CALLBACK_PREFIX}repeat:"
NEW_PUZZLE_CALLBACK_PREFIX = f"{COMPLETION_CALLBACK_PREFIX}new:"

BUTTON_NEW_GAME_KEY = "button_new_game_flow"
BUTTON_STEP_KEY = "step"
BUTTON_LANGUAGE_KEY = "language"
BUTTON_STEP_LANGUAGE = "language"
BUTTON_STEP_THEME = "theme"

GENERATION_NOTICE_KEY = "puzzle_generation_notice"

ADMIN_COMMAND_PATTERN = re.compile(r"(?i)^\s*adm key")
ADMIN_KEYS_ONLY_PATTERN = re.compile(r"(?i)^\s*adm keys\s*$")
ADMIN_SINGLE_KEY_PATTERN = re.compile(r"(?i)^\s*adm key\s+(.+)$")


def _parse_inline_answer(text: str | None) -> Optional[tuple[str, str]]:
    if not text:
        logger.info("Inline answer parsing skipped: no text provided")
        return None
    match = INLINE_ANSWER_PATTERN.match(text)
    if not match:
        logger.info(
            "Inline answer parsing skipped: pattern did not match",
            extra={"text": text},
        )
        return None
    slot_id, answer = match.groups()
    cleaned_answer = answer.strip()
    if not cleaned_answer:
        logger.warning(
            "Inline answer parsing skipped: answer part empty after stripping",
            extra={"text": text},
        )
        return None
    return _normalise_slot_id(slot_id), cleaned_answer


def _resolve_slot(puzzle: Puzzle | CompositePuzzle, slot_id: str) -> tuple[Optional[SlotRef], Optional[str]]:
    """Return slot reference for the provided identifier with ambiguity notice."""

    base_id, component_index = parse_slot_public_id(slot_id)
    if isinstance(puzzle, CompositePuzzle):
        if component_index is None:
            matches = [
                ref
                for ref in iter_slot_refs(puzzle)
                if ref.slot.slot_id.upper() == base_id
            ]
            if len(matches) > 1:
                options = ", ".join(ref.public_id for ref in matches)
                return None, f"Уточните компоненту: {options}"
            if matches:
                return matches[0], None
            return None, None
        return find_slot_ref(puzzle, slot_id), None
    # single puzzle
    return find_slot_ref(puzzle, slot_id), None


def _canonical_answer(word: str, language: str) -> str:
    transformed = (word or "").strip().upper()
    if language.lower() == "ru":
        transformed = transformed.replace("Ё", "Е")
    return transformed


def _canonical_letter_set(word: str, language: str) -> set[str]:
    """Return a canonicalised set of letters used for intersection checks."""

    return {char for char in _canonical_answer(word, language) if char.isalpha()}


def _ensure_hint_set(game_state: GameState) -> set[str]:
    if game_state.hinted_cells is None:
        game_state.hinted_cells = set()
    return game_state.hinted_cells


def _resolve_player_id(game_state: GameState, user_id: int | None = None) -> int | None:
    if user_id is not None:
        return user_id
    if game_state.host_id is not None:
        return game_state.host_id
    return game_state.chat_id


def _record_score(game_state: GameState, delta: int, user_id: int | None = None) -> None:
    if not delta:
        return
    player_id = _resolve_player_id(game_state, user_id)
    if player_id is None:
        return
    if player_id not in game_state.scoreboard:
        game_state.scoreboard[player_id] = 0
    game_state.scoreboard[player_id] = game_state.scoreboard.get(player_id, 0) + delta


def _record_hint_usage(
    game_state: GameState, slot_identifier: str, user_id: int | None = None
) -> None:
    player_id = _resolve_player_id(game_state, user_id)
    slot_key = _normalise_slot_id(slot_identifier)
    usage = game_state.hints_used.setdefault(slot_key, {})
    if player_id is None:
        player_id = 0
    usage[player_id] = usage.get(player_id, 0) + 1


def _ensure_player_profile(
    game_state: GameState,
    user_id: int,
    *,
    name: str | None = None,
    dm_chat_id: int | None = None,
) -> Player:
    player = game_state.players.get(user_id)
    if player is None:
        player = Player(user_id=user_id, name=name or f"Игрок {user_id}")
        game_state.players[user_id] = player
    if name and player.name != name:
        player.name = name
    if dm_chat_id and player.dm_chat_id != dm_chat_id:
        player.dm_chat_id = dm_chat_id
    if player.dm_chat_id:
        state.player_chats[player.user_id] = player.dm_chat_id
    if user_id not in game_state.scoreboard:
        game_state.scoreboard[user_id] = 0
    return player


def _resolve_game_state_for_message(
    chat: Chat | None, user_id: int | None
) -> GameState | None:
    if chat is None:
        return None
    primary = _load_state_for_chat(chat.id)
    if primary is not None:
        return primary
    if chat.type == ChatType.PRIVATE and user_id is not None:
        dm_chat_id = state.player_chats.get(user_id)
        if dm_chat_id and dm_chat_id != chat.id:
            mapped = _load_state_for_chat(dm_chat_id)
            if mapped is not None:
                return mapped
        for game in state.active_games.values():
            player = game.players.get(user_id)
            if player and player.dm_chat_id == chat.id:
                return game
    return None


def _cancel_jobs_by_name(job_queue, name: str | None) -> None:
    if not name or job_queue is None:
        return
    for job in job_queue.get_jobs_by_name(name):
        job.schedule_removal()


def _cancel_turn_jobs(context: ContextTypes.DEFAULT_TYPE, game_state: GameState) -> None:
    job_queue = getattr(context, "job_queue", None)
    if job_queue is None:
        return
    _cancel_jobs_by_name(job_queue, game_state.turn_job_id)
    _cancel_jobs_by_name(job_queue, game_state.turn_warn_job_id)
    game_state.turn_job_id = None
    game_state.turn_warn_job_id = None


def _cancel_game_jobs(context: ContextTypes.DEFAULT_TYPE, game_state: GameState) -> None:
    job_queue = getattr(context, "job_queue", None)
    if job_queue is None:
        return
    _cancel_jobs_by_name(job_queue, game_state.timer_job_id)
    _cancel_jobs_by_name(job_queue, game_state.warn_job_id)
    game_state.timer_job_id = None
    game_state.warn_job_id = None


def _schedule_turn_jobs(context: ContextTypes.DEFAULT_TYPE, game_state: GameState) -> None:
    job_queue = getattr(context, "job_queue", None)
    if job_queue is None:
        return
    _cancel_turn_jobs(context, game_state)
    if TURN_DURATION_SECONDS <= 0:
        return
    timeout_name = f"{TURN_TIMEOUT_JOB_PREFIX}-{game_state.game_id}"
    job_queue.run_once(
        _turn_timeout_job,
        when=TURN_DURATION_SECONDS,
        chat_id=game_state.chat_id,
        name=timeout_name,
    )
    game_state.turn_job_id = timeout_name
    if TURN_WARNING_SECONDS and TURN_WARNING_SECONDS < TURN_DURATION_SECONDS:
        warn_name = f"{TURN_WARN_JOB_PREFIX}-{game_state.game_id}"
        job_queue.run_once(
            _turn_warning_job,
            when=max(TURN_DURATION_SECONDS - TURN_WARNING_SECONDS, 1),
            chat_id=game_state.chat_id,
            name=warn_name,
        )
        game_state.turn_warn_job_id = warn_name


def _schedule_game_jobs(context: ContextTypes.DEFAULT_TYPE, game_state: GameState) -> None:
    job_queue = getattr(context, "job_queue", None)
    if job_queue is None:
        return
    _cancel_game_jobs(context, game_state)
    if TOTAL_GAME_DURATION_SECONDS <= 0:
        return
    timeout_name = f"{GAME_TIMEOUT_JOB_PREFIX}-{game_state.game_id}"
    job_queue.run_once(
        _game_timeout_job,
        when=TOTAL_GAME_DURATION_SECONDS,
        chat_id=game_state.chat_id,
        name=timeout_name,
    )
    game_state.timer_job_id = timeout_name
    if GAME_WARNING_SECONDS and GAME_WARNING_SECONDS < TOTAL_GAME_DURATION_SECONDS:
        warn_name = f"{GAME_TIMEOUT_JOB_PREFIX}-warn-{game_state.game_id}"
        job_queue.run_once(
            _game_warning_job,
            when=max(TOTAL_GAME_DURATION_SECONDS - GAME_WARNING_SECONDS, 1),
            chat_id=game_state.chat_id,
            name=warn_name,
        )
        game_state.warn_job_id = warn_name


async def _announce_current_turn(
    context: ContextTypes.DEFAULT_TYPE,
    game_state: GameState,
    *,
    prefix: str | None = None,
) -> None:
    player = _current_player(game_state)
    if player is None:
        return
    turn_text = prefix or f"Ход игрока {player.name}."
    keyboard = InlineKeyboardMarkup(
        [[InlineKeyboardButton("Выбрать подсказку", callback_data=TURN_SELECT_CALLBACK)]]
    )
    try:
        await context.bot.send_message(
            chat_id=game_state.chat_id,
            text=turn_text,
            reply_markup=keyboard,
        )
    except Exception:  # noqa: BLE001
        logger.exception("Failed to announce current turn in chat %s", game_state.chat_id)
    if player.dm_chat_id:
        try:
            await context.bot.send_message(
                chat_id=player.dm_chat_id,
                text="Ваш ход! Выберите подсказку, чтобы ответить на вопрос.",
                reply_markup=keyboard,
            )
        except Exception:  # noqa: BLE001
            logger.exception("Failed to notify player %s in DM about their turn", player.user_id)


async def _complete_turn(
    context: ContextTypes.DEFAULT_TYPE,
    game_state: GameState,
    *,
    reason: str,
    advance: bool = True,
    announce_next: bool = True,
) -> None:
    if game_state.mode != "turn_based":
        return
    _cancel_turn_jobs(context, game_state)
    if advance and game_state.turn_order:
        game_state.turn_index = (game_state.turn_index + 1) % len(game_state.turn_order)
    game_state.active_slot_id = None
    game_state.turn_started_at = time.time()
    _store_state(game_state)
    if announce_next and game_state.turn_order:
        await _announce_current_turn(context, game_state)
        _schedule_turn_jobs(context, game_state)


def _initialise_turn_based_mode(
    context: ContextTypes.DEFAULT_TYPE,
    game_state: GameState,
    *,
    randomise_order: bool = True,
) -> None:
    players = list(game_state.players.values())
    if not players and game_state.host_id is not None:
        player = Player(user_id=game_state.host_id, name=f"Игрок {game_state.host_id}")
        players.append(player)
        game_state.players[player.user_id] = player
    if not players:
        host_player = Player(user_id=game_state.chat_id, name=f"Игрок {game_state.chat_id}")
        game_state.players[host_player.user_id] = host_player
        players.append(host_player)
    players.sort(key=lambda entry: entry.joined_at)
    order = [player.user_id for player in players]
    if randomise_order and len(order) > 1:
        random.shuffle(order)
    game_state.turn_order = order
    game_state.turn_index = 0
    game_state.status = "running"
    game_state.active_slot_id = None
    game_state.turn_started_at = time.time()
    game_state.score = 0
    for player in players:
        game_state.scoreboard[player.user_id] = 0
        player.answers_ok = 0
        player.answers_fail = 0
    _schedule_game_jobs(context, game_state)
    _schedule_turn_jobs(context, game_state)
    _store_state(game_state)


def _collect_hint_usage_summary(game_state: GameState) -> dict[int, int]:
    summary: dict[int, int] = {}
    for usage in game_state.hints_used.values():
        for user_id, count in usage.items():
            summary[user_id] = summary.get(user_id, 0) + count
    return summary


def _build_turn_based_completion_keyboard(
    puzzle: Puzzle | CompositePuzzle | None, fallback_id: str
) -> InlineKeyboardMarkup:
    puzzle_id = puzzle.id if puzzle else fallback_id
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton(
                    "Реванш",
                    callback_data=f"{SAME_TOPIC_CALLBACK_PREFIX}{puzzle_id}",
                )
            ],
            [
                InlineKeyboardButton(
                    "Новая игра",
                    callback_data=f"{NEW_PUZZLE_CALLBACK_PREFIX}{puzzle_id}",
                )
            ],
        ]
    )


async def _finish_game(
    context: ContextTypes.DEFAULT_TYPE,
    game_state: GameState,
    *,
    reason: str,
    puzzle: Puzzle | CompositePuzzle | None = None,
) -> None:
    _cancel_turn_jobs(context, game_state)
    _cancel_game_jobs(context, game_state)
    game_state.status = "finished"
    game_state.active_slot_id = None
    game_state.turn_job_id = None
    game_state.turn_warn_job_id = None
    game_state.timer_job_id = None
    game_state.warn_job_id = None
    game_state.turn_started_at = None
    _store_state(game_state)

    if puzzle is None:
        puzzle = _load_puzzle_for_state(game_state)

    hint_summary = _collect_hint_usage_summary(game_state)
    leaderboard_lines: list[str] = []
    for user_id, score in sorted(
        game_state.scoreboard.items(),
        key=lambda item: item[1],
        reverse=True,
    ):
        player = game_state.players.get(user_id)
        name = player.name if player else f"Игрок {user_id}"
        solved = player.answers_ok if player else 0
        hints_used = hint_summary.get(user_id, 0)
        leaderboard_lines.append(
            f"{name}: {score} очков • решено {solved} • подсказок {hints_used}"
        )

    leaderboard_text = "\n".join(leaderboard_lines) if leaderboard_lines else "Нет данных."
    reason_map = {
        "completed": "Все слова разгаданы!",
        "timeout": "Истекло время игры.",
        "manual": "Игра остановлена хостом.",
    }
    reason_text = reason_map.get(reason, "Игра завершена.")
    message_text = f"🏁 {reason_text}\n\nИтоги:\n{leaderboard_text}"
    keyboard = _build_turn_based_completion_keyboard(puzzle, game_state.puzzle_id)
    await context.bot.send_message(
        chat_id=game_state.chat_id,
        text=message_text,
        reply_markup=keyboard,
    )
    for player in game_state.players.values():
        if not player.dm_chat_id:
            continue
        with suppress(Exception):
            await context.bot.send_message(
                chat_id=player.dm_chat_id,
                text=message_text,
                reply_markup=keyboard,
            )


def _current_player_id(game_state: GameState) -> int | None:
    if not game_state.turn_order:
        return None
    index = game_state.turn_index % len(game_state.turn_order)
    if index < 0:
        index = 0
    if index >= len(game_state.turn_order):
        return None
    return game_state.turn_order[index]


def _current_player(game_state: GameState) -> Player | None:
    user_id = _current_player_id(game_state)
    if user_id is None:
        return None
    return game_state.players.get(user_id)


def _is_current_player(game_state: GameState, user_id: int | None) -> bool:
    if user_id is None:
        return False
    return user_id == _current_player_id(game_state)


def _store_state(game_state: GameState) -> None:
    with logging_context(chat_id=game_state.chat_id, puzzle_id=game_state.puzzle_id):
        state.active_games[game_state.game_id] = game_state
        state.chat_to_game[game_state.chat_id] = game_state.game_id
        for code, target in list(state.join_codes.items()):
            if target == game_state.game_id and code not in game_state.join_codes:
                state.join_codes.pop(code, None)
        for code, target in game_state.join_codes.items():
            state.join_codes[code] = target
        save_state(game_state)
        logger.info("Game state persisted for game %s", game_state.game_id)


def _load_state_for_chat(chat_id: int) -> Optional[GameState]:
    with logging_context(chat_id=chat_id):
        game_id = state.chat_to_game.get(chat_id)
        if game_id and game_id in state.active_games:
            return state.active_games[game_id]
        identifier = game_id if game_id is not None else chat_id
        restored = load_state(identifier)
        if restored is None:
            if game_id is not None:
                state.chat_to_game.pop(chat_id, None)
            return None
        state.active_games[restored.game_id] = restored
        state.chat_to_game[restored.chat_id] = restored.game_id
        for code, target in list(state.join_codes.items()):
            if target == restored.game_id and code not in restored.join_codes:
                state.join_codes.pop(code, None)
        for code, target in restored.join_codes.items():
            state.join_codes[code] = target
        logger.info("Restored state from disk during command handling")
        return restored


def _load_puzzle_for_state(game_state: GameState) -> Optional[Puzzle | CompositePuzzle]:
    with logging_context(chat_id=game_state.chat_id, puzzle_id=game_state.puzzle_id):
        payload = load_puzzle(game_state.puzzle_id)
        if payload is None:
            logger.error("Puzzle referenced by chat is missing")
            return None
        logger.debug("Loaded puzzle definition for rendering or clues")
        return puzzle_from_dict(dict(payload))


def _format_clue_section(
    slot_refs: Iterable[SlotRef], solved_ids: set[str] | None = None
) -> str:
    solved_lookup = solved_ids if solved_ids is not None else set()
    lines: list[str] = []
    for slot_ref in slot_refs:
        slot = slot_ref.slot
        clue_text = html.escape(slot.clue or "(нет подсказки)")
        public_id = html.escape(slot_ref.public_id)
        line_text = f"{public_id}: {clue_text}"
        if _normalise_slot_id(slot_ref.public_id) in solved_lookup:
            line_text = f"<b>{line_text}</b> ✅"
        lines.append(line_text)
    return "\n".join(lines) if lines else "(подсказок нет)"


def _format_clues_message(
    puzzle: Puzzle | CompositePuzzle, game_state: GameState | None = None
) -> str:
    solved_ids: set[str] = (
        {_normalise_slot_id(entry) for entry in game_state.solved_slots}
        if game_state
        else set()
    )
    if isinstance(puzzle, CompositePuzzle):
        sections: list[str] = []
        for component in sorted(puzzle.components, key=lambda comp: comp.index):
            refs = [
                ref
                for ref in iter_slot_refs(puzzle)
                if ref.component_index == component.index
            ]
            across = [ref for ref in refs if ref.slot.direction is Direction.ACROSS]
            down = [ref for ref in refs if ref.slot.direction is Direction.DOWN]
            section = (
                f"Сетка {component.index}:\n"
                f"Across:\n{_format_clue_section(across, solved_ids)}\n\n"
                f"Down:\n{_format_clue_section(down, solved_ids)}"
            )
            sections.append(section)
        return "\n\n".join(sections)

    across = [
        ref for ref in iter_slot_refs(puzzle) if ref.slot.direction is Direction.ACROSS
    ]
    down = [
        ref for ref in iter_slot_refs(puzzle) if ref.slot.direction is Direction.DOWN
    ]
    across_text = _format_clue_section(across, solved_ids)
    down_text = _format_clue_section(down, solved_ids)
    return f"Across:\n{across_text}\n\nDown:\n{down_text}"


def _sorted_slot_refs(puzzle: Puzzle | CompositePuzzle) -> list[SlotRef]:
    return sorted(
        iter_slot_refs(puzzle),
        key=lambda ref: (
            ref.component_index if ref.component_index is not None else -1,
            ref.slot.direction.value,
            ref.slot.number,
            ref.slot.slot_id,
        ),
    )


def _format_admin_answers(puzzle: Puzzle | CompositePuzzle) -> str:
    lines: list[str] = []
    previous_component: Optional[int] = None
    is_composite = isinstance(puzzle, CompositePuzzle)
    for ref in _sorted_slot_refs(puzzle):
        component_index = ref.component_index
        if is_composite and component_index != previous_component:
            lines.append(f"[Компонента {component_index + 1}]")
            previous_component = component_index
        answer = ref.slot.answer or "(нет ответа)"
        lines.append(f"{ref.public_id}: {answer}")
    return "\n".join(lines) if lines else "Ответы отсутствуют."


def _format_slot_answers(slot_refs: Sequence[SlotRef]) -> str:
    if not slot_refs:
        return "Ответ не найден."
    lines = []
    for ref in slot_refs:
        answer = ref.slot.answer or "(нет ответа)"
        lines.append(f"{ref.public_id}: {answer}")
    return "\n".join(lines)


async def _send_clues_update(
    message: Message,
    puzzle: Puzzle | CompositePuzzle,
    game_state: GameState,
) -> None:
    if _all_slots_solved(puzzle, game_state):
        return
    text = _format_clues_message(puzzle, game_state)
    if game_state.mode == "turn_based" and game_state.status == "running":
        current = _current_player(game_state)
        if current:
            text += f"\n\nХод игрока: {current.name}"
        if game_state.active_slot_id:
            text += f"\nВыбран слот: {game_state.active_slot_id}"
    await message.reply_text(
        text,
        parse_mode=constants.ParseMode.HTML,
    )


def _build_completion_keyboard(puzzle: Puzzle | CompositePuzzle) -> InlineKeyboardMarkup:
    same_topic_data = f"{SAME_TOPIC_CALLBACK_PREFIX}{puzzle.id}"
    new_puzzle_data = f"{NEW_PUZZLE_CALLBACK_PREFIX}{puzzle.id}"
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton(
                    "Еще один кроссворд на эту же тему",
                    callback_data=same_topic_data,
                )
            ],
            [
                InlineKeyboardButton(
                    "Новый кроссворд",
                    callback_data=new_puzzle_data,
                )
            ],
        ]
    )


async def _send_generation_notice(
    context: ContextTypes.DEFAULT_TYPE,
    chat_id: int,
    text: str,
    *,
    message: Message | None = None,
) -> None:
    """Send a single informational message about puzzle generation per chat."""

    notice = context.chat_data.get(GENERATION_NOTICE_KEY)
    if notice and notice.get("active") and notice.get("text") == text:
        logger.debug(
            "Skipping duplicate generation notice for chat %s", chat_id
        )
        return

    context.chat_data[GENERATION_NOTICE_KEY] = {
        "active": True,
        "text": text,
        "started_at": time.monotonic(),
    }

    if message is not None:
        await message.reply_text(text)
    else:
        await context.bot.send_message(chat_id=chat_id, text=text)


def _clear_generation_notice(
    context: ContextTypes.DEFAULT_TYPE, chat_id: int | None
) -> None:
    """Clear generation notice tracking for the chat."""

    if context.chat_data.pop(GENERATION_NOTICE_KEY, None) is not None and chat_id is not None:
        logger.debug("Cleared generation notice flag for chat %s", chat_id)


async def _send_completion_options(
    context: ContextTypes.DEFAULT_TYPE,
    chat_id: int,
    message: Message | None,
    puzzle: Puzzle | CompositePuzzle,
) -> None:
    keyboard = _build_completion_keyboard(puzzle)
    text = "Продолжить?"
    if message is not None:
        await message.reply_text(text, reply_markup=keyboard)
        return
    await context.bot.send_message(chat_id=chat_id, text=text, reply_markup=keyboard)


async def _deliver_puzzle_via_bot(
    context: ContextTypes.DEFAULT_TYPE,
    chat_id: int,
    puzzle: Puzzle | CompositePuzzle,
    game_state: GameState,
) -> bool:
    if is_chat_mode_set(context) and get_chat_mode(context) != MODE_IN_GAME:
        logger.warning(
            "Attempted to deliver puzzle while chat %s in mode %s",
            chat_id,
            get_chat_mode(context),
        )
        return False
    image_path = None
    try:
        with logging_context(puzzle_id=puzzle.id):
            image_path = render_puzzle(puzzle, game_state)
            await context.bot.send_chat_action(
                chat_id=chat_id, action=constants.ChatAction.UPLOAD_PHOTO
            )
            with open(image_path, "rb") as photo:
                await context.bot.send_photo(
                    chat_id=chat_id,
                    photo=photo,
                    caption=(
                        f"Кроссворд готов!\nЯзык: {puzzle.language.upper()}\nТема: {puzzle.theme}"
                    ),
                )
            await context.bot.send_message(
                chat_id=chat_id,
                text=_format_clues_message(puzzle, game_state),
                parse_mode=constants.ParseMode.HTML,
            )
            await context.bot.send_message(
                chat_id=chat_id,
                text=(
                    "Отправляйте ответы прямо в чат в формате «A1 - ответ». "
                    "Если удобнее, можно пользоваться и командой /answer."
                ),
            )
            logger.info("Delivered freshly generated puzzle to chat %s", chat_id)
            return True
    except Exception:  # noqa: BLE001
        logger.exception("Failed to deliver puzzle to chat %s", chat_id)
        if image_path is not None:
            with suppress(OSError):
                image_path.unlink(missing_ok=True)
    return False


def _ensure_private_chat(update: Update) -> bool:
    chat = update.effective_chat
    is_private = bool(chat and chat.type == ChatType.PRIVATE)
    if not is_private and update.effective_message:
        logger.debug("Rejected command in non-private chat %s", chat.id if chat else "<unknown>")
    return is_private


async def _reject_group_chat(update: Update) -> bool:
    chat = update.effective_chat
    if chat and chat.type != ChatType.PRIVATE:
        game_state = _load_state_for_chat(chat.id)
        if game_state and game_state.mode == "turn_based":
            return True
    if _ensure_private_chat(update):
        return True
    if update.effective_message:
        await update.effective_message.reply_text(
            "Пожалуйста, используйте этого бота в личном чате."
        )
    return False


def _apply_answer_to_state(game_state: GameState, slot_ref: SlotRef, answer: str) -> None:
    with logging_context(chat_id=game_state.chat_id, puzzle_id=game_state.puzzle_id):
        slot = slot_ref.slot
        component = slot_ref.component_index
        logger.debug("Applying answer for slot %s", slot_ref.public_id)
        keys: list[str] = []
        for index, (row, col) in enumerate(slot.coordinates()):
            key = _coord_key(row, col, component)
            keys.append(key)
            if index < len(answer):
                game_state.filled_cells[key] = answer[index]
        hint_set = _ensure_hint_set(game_state)
        for key in keys:
            hint_set.discard(key)
        game_state.solved_slots.add(_normalise_slot_id(slot_ref.public_id))
        game_state.last_update = time.time()
        _store_state(game_state)


def _reveal_letter(
    game_state: GameState, slot_ref: SlotRef, answer: str, user_id: int | None = None
) -> Optional[tuple[int, str]]:
    hint_set = _ensure_hint_set(game_state)
    slot = slot_ref.slot
    component = slot_ref.component_index
    for index, (row, col) in enumerate(slot.coordinates()):
        key = _coord_key(row, col, component)
        if key in game_state.filled_cells:
            continue
        if index >= len(answer):
            break
        letter = answer[index]
        game_state.filled_cells[key] = letter
        hint_set.add(key)
        _record_hint_usage(game_state, slot_ref.public_id, user_id=user_id)
        game_state.last_update = time.time()
        _store_state(game_state)
        return index, letter
    return None


def _all_slots_solved(puzzle: Puzzle | CompositePuzzle, game_state: GameState) -> bool:
    solved = {_normalise_slot_id(entry) for entry in game_state.solved_slots}
    for slot_ref in iter_slot_refs(puzzle):
        if slot_ref.slot.answer and _normalise_slot_id(slot_ref.public_id) not in solved:
            return False
    return True


def _solve_remaining_slots(
    game_state: GameState, puzzle: Puzzle | CompositePuzzle
) -> list[tuple[str, str]]:
    solved_now: list[tuple[str, str]] = []
    hint_set = _ensure_hint_set(game_state)
    solved_ids = {_normalise_slot_id(entry) for entry in game_state.solved_slots}
    for slot_ref in iter_slot_refs(puzzle):
        slot = slot_ref.slot
        public_id = _normalise_slot_id(slot_ref.public_id)
        if not slot.answer:
            continue
        if public_id in solved_ids:
            continue
        answer = slot.answer
        for index, (row, col) in enumerate(slot.coordinates()):
            if index >= len(answer):
                break
            key = _coord_key(row, col, slot_ref.component_index)
            game_state.filled_cells[key] = answer[index]
            hint_set.discard(key)
        game_state.solved_slots.add(public_id)
        solved_ids.add(public_id)
        solved_now.append((slot_ref.public_id, answer))
    if solved_now:
        game_state.last_update = time.time()
        _store_state(game_state)
    return solved_now


def _cancel_reminder(context: ContextTypes.DEFAULT_TYPE) -> None:
    job = context.chat_data.pop("reminder_job", None)
    if job is not None:
        logger.debug("Cancelling existing reminder job %s", job.name)
        job.schedule_removal()


async def _reminder_job(context: CallbackContext) -> None:
    job = context.job
    if job is None:
        return
    chat_id = job.chat_id
    with logging_context(chat_id=chat_id):
        logger.debug("Sending reminder for chat %s", chat_id)
        try:
            await context.bot.send_message(
                chat_id=chat_id,
                text="Не забывайте про /hint, если нужна подсказка!",
            )
        except Exception:  # noqa: BLE001
            logger.exception("Failed to deliver reminder message to chat %s", chat_id)


async def _turn_warning_job(context: CallbackContext) -> None:
    job = context.job
    if job is None:
        return
    game_state = _load_state_for_chat(job.chat_id)
    if not game_state or game_state.mode != "turn_based" or game_state.status != "running":
        return
    player = _current_player(game_state)
    if player is None:
        return
    warning_text = f"⏰ {player.name}, осталось {TURN_WARNING_SECONDS} секунд до окончания хода."
    try:
        await context.bot.send_message(chat_id=game_state.chat_id, text=warning_text)
    except Exception:  # noqa: BLE001
        logger.exception("Failed to send turn warning to chat %s", game_state.chat_id)
    if player.dm_chat_id:
        try:
            await context.bot.send_message(
                chat_id=player.dm_chat_id,
                text="Осталось немного времени, завершите ход!",
            )
        except Exception:  # noqa: BLE001
            logger.exception("Failed to send turn warning DM to %s", player.user_id)


async def _turn_timeout_job(context: CallbackContext) -> None:
    job = context.job
    if job is None:
        return
    game_state = _load_state_for_chat(job.chat_id)
    if not game_state or game_state.mode != "turn_based" or game_state.status != "running":
        return
    player = _current_player(game_state)
    if player is not None:
        player.answers_fail += 1
        await context.bot.send_message(
            chat_id=game_state.chat_id,
            text=f"⏱ Ход игрока {player.name} завершён по таймауту.",
        )
        if player.dm_chat_id:
            with suppress(Exception):
                await context.bot.send_message(
                    chat_id=player.dm_chat_id,
                    text="Время вышло. Ход передан следующему игроку.",
                )
    await _complete_turn(context, game_state, reason="timeout")


async def _game_warning_job(context: CallbackContext) -> None:
    job = context.job
    if job is None:
        return
    game_state = _load_state_for_chat(job.chat_id)
    if not game_state or game_state.mode != "turn_based" or game_state.status != "running":
        return
    try:
        await context.bot.send_message(
            chat_id=game_state.chat_id,
            text="⚠️ До завершения игры остаётся одна минута!",
        )
    except Exception:  # noqa: BLE001
        logger.exception("Failed to send game warning for chat %s", game_state.chat_id)


async def _game_timeout_job(context: CallbackContext) -> None:
    job = context.job
    if job is None:
        return
    game_state = _load_state_for_chat(job.chat_id)
    if not game_state or game_state.mode != "turn_based" or game_state.status != "running":
        return
    await context.bot.send_message(
        chat_id=game_state.chat_id,
        text="⏳ Общее время истекло! Игра завершается.",
    )
    await _finish_game(context, game_state, reason="timeout")


def _assign_clues_to_slots(puzzle: Puzzle | CompositePuzzle, clues: Sequence[WordClue]) -> None:
    language = puzzle.language
    clue_map: dict[str, str] = {}
    for clue in clues:
        clue_map[_canonical_answer(clue.word, language)] = clue.clue
    if isinstance(puzzle, CompositePuzzle):
        for component in puzzle.components:
            _assign_clues_to_slots(component.puzzle, clues)
        return
    for slot in puzzle.slots:
        if not slot.answer:
            continue
        canonical = _canonical_answer(slot.answer, language)
        slot.clue = clue_map.get(canonical, f"Слово из {slot.length} букв")


def _build_word_components(clues: Sequence[WordClue], language: str) -> list[list[WordClue]]:
    """Split clues into connected components by shared letters."""

    canonical_forms = [_canonical_answer(clue.word, language) for clue in clues]
    graph: dict[int, set[int]] = {idx: set() for idx in range(len(clues))}
    for i in range(len(clues)):
        for j in range(i + 1, len(clues)):
            if set(canonical_forms[i]) & set(canonical_forms[j]):
                graph[i].add(j)
                graph[j].add(i)

    visited: set[int] = set()
    components: list[list[WordClue]] = []

    for idx in range(len(clues)):
        if idx in visited:
            continue
        stack = [idx]
        component_indices: list[int] = []
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            component_indices.append(current)
            stack.extend(graph[current])
        if component_indices:
            components.append([clues[i] for i in component_indices])
    components.sort(key=len, reverse=True)
    return components


def _generate_composite(
    chat_id: int,
    language: str,
    theme: str,
    components: Sequence[Sequence[WordClue]],
) -> tuple[CompositePuzzle, GameState]:
    composite_id = uuid4().hex
    composite_components: list[CompositeComponent] = []
    row_offset = 0
    for index, component_clues in enumerate(components, start=1):
        words = [clue.word for clue in component_clues]
        puzzle = generate_fill_in_puzzle(
            puzzle_id=f"{composite_id}-c{index}",
            theme=theme,
            language=language,
            words=words,
            max_size=MAX_PUZZLE_SIZE,
        )
        _assign_clues_to_slots(puzzle, component_clues)
        composite_components.append(
            CompositeComponent(
                index=index,
                puzzle=puzzle,
                row_offset=row_offset,
                col_offset=0,
            )
        )
        row_offset += puzzle.size_rows + 1

    composite = CompositePuzzle(
        id=composite_id,
        theme=theme,
        language=language,
        components=composite_components,
        gap_cells=1,
    )
    save_puzzle(composite.id, composite_to_dict(composite))
    now = time.time()
    game_state = GameState(
        chat_id=chat_id,
        puzzle_id=composite.id,
        puzzle_ids=[component.puzzle.id for component in composite_components],
        filled_cells={},
        solved_slots=set(),
        score=0,
        started_at=now,
        last_update=now,
        hinted_cells=set(),
        host_id=chat_id,
        game_id=str(chat_id),
        scoreboard={chat_id: 0},
    )
    return composite, game_state


def _generate_puzzle(
    chat_id: int, language: str, theme: str
) -> tuple[Puzzle | CompositePuzzle, GameState]:
    with logging_context(chat_id=chat_id):
        logger.info(
            "Starting puzzle generation (language=%s, theme=%s)",
            language,
            theme,
        )
        clues = generate_clues(theme=theme, language=language)
        logger.info("Received %s raw clues from LLM", len(clues))
        validated_clues = validate_word_list(language, clues, deduplicate=True)
        logger.info("Validated %s clues for placement", len(validated_clues))
        if not validated_clues:
            raise RuntimeError("Не удалось подобрать ни одного подходящего слова")

        max_attempt_words = min(len(validated_clues), 80)
        min_attempt_words = max(10, min(30, max_attempt_words))

        attempted_component_split = False
        replacement_prompt_words: set[str] = set()
        used_canonical_words: set[str] = {
            _canonical_answer(clue.word, language) for clue in validated_clues
        }
        replacement_requests = 0

        def request_replacement(
            word: str, attempt_clues: Sequence[WordClue]
        ) -> WordClue | None:
            nonlocal replacement_requests
            canonical = _canonical_answer(word, language)
            replacement_prompt_words.add(canonical)
            other_letters: set[str] = set()
            for clue in attempt_clues:
                if _canonical_answer(clue.word, language) == canonical:
                    continue
                other_letters.update(_canonical_letter_set(clue.word, language))
            while True:
                if replacement_requests >= MAX_REPLACEMENT_REQUESTS:
                    logger.warning(
                        "Reached maximum replacement requests (%s) while trying to replace %s",
                        MAX_REPLACEMENT_REQUESTS,
                        word,
                    )
                    return None
                replacement_requests += 1
                prompt_suffix = ", ".join(sorted(replacement_prompt_words))
                replacement_theme = (
                    f"{theme}. Подбери альтернативные слова для кроссворда вместо: {prompt_suffix}."
                )
                logger.info(
                    "Requesting replacement clues (attempt %s) for: %s",
                    replacement_requests,
                    prompt_suffix,
                )
                new_clues = generate_clues(theme=replacement_theme, language=language)
                new_validated = validate_word_list(language, new_clues, deduplicate=True)
                logger.info(
                    "Validated %s replacement candidates", len(new_validated)
                )
                for candidate in new_validated:
                    candidate_canonical = _canonical_answer(candidate.word, language)
                    if candidate_canonical in used_canonical_words:
                        continue
                    candidate_letters = _canonical_letter_set(candidate.word, language)
                    if other_letters and not (candidate_letters & other_letters):
                        logger.debug(
                            "Skipping replacement %s: no shared letters with current attempt",
                            candidate.word,
                        )
                        continue
                    used_canonical_words.add(candidate_canonical)
                    return candidate
                logger.warning(
                    "Replacement attempt %s did not provide new unique words",
                    replacement_requests,
                )


        for limit in range(max_attempt_words, min_attempt_words - 1, -1):
            candidate_clues = list(validated_clues[:limit])
            puzzle_id = uuid4().hex
            with logging_context(chat_id=chat_id, puzzle_id=puzzle_id):
                attempt_clues = list(candidate_clues)
                while True:
                    try:
                        puzzle = generate_fill_in_puzzle(
                            puzzle_id=puzzle_id,
                            theme=theme,
                            language=language,
                            words=[clue.word for clue in attempt_clues],
                            max_size=MAX_PUZZLE_SIZE,
                        )
                    except DisconnectedWordError as disconnected:
                        logger.info(
                            "Word %s could not be connected, requesting replacement",
                            disconnected.word,
                        )
                        replacement = request_replacement(
                            disconnected.word, attempt_clues
                        )
                        if replacement is None:
                            logger.debug(
                                "No replacement available for %s, abandoning attempt",
                                disconnected.word,
                            )
                            break
                        target_canonical = _canonical_answer(
                            disconnected.word, language
                        )
                        for idx, clue in enumerate(attempt_clues):
                            if (
                                _canonical_answer(clue.word, language)
                                == target_canonical
                            ):
                                attempt_clues[idx] = replacement
                                break
                        else:
                            attempt_clues.append(replacement)
                        continue
                    except FillInGenerationError as error:
                        logger.debug(
                            "Attempt with %s words failed to build grid: %s",
                            limit,
                            error,
                        )
                        if (
                            not attempted_component_split
                            and limit == max_attempt_words
                            and len(validated_clues[:limit]) > 1
                        ):
                            components = _build_word_components(
                                validated_clues[:limit], language
                            )
                            attempted_component_split = True
                            if len(components) > 1:
                                logger.info(
                                    "Detected %s disconnected word clusters, generating composite puzzle",
                                    len(components),
                                )
                                try:
                                    composite, game_state = _generate_composite(
                                        chat_id, language, theme, components
                                    )
                                except FillInGenerationError as composite_error:
                                    logger.debug(
                                        "Composite generation failed: %s", composite_error
                                    )
                                else:
                                    logger.info(
                                        "Generated composite puzzle with %s components",
                                        len(components),
                                    )
                                    _store_state(game_state)
                                    return composite, game_state
                        break
                    else:
                        logger.info(
                            "Constructed dynamic puzzle grid using %s candidate words",
                            len(attempt_clues),
                        )
                        _assign_clues_to_slots(puzzle, attempt_clues)
                        save_puzzle(puzzle.id, puzzle_to_dict(puzzle))
                        now = time.time()
                        game_state = GameState(
                            chat_id=chat_id,
                            puzzle_id=puzzle.id,
                            puzzle_ids=None,
                            filled_cells={},
                            solved_slots=set(),
                            score=0,
                            started_at=now,
                            last_update=now,
                            hinted_cells=set(),
                            host_id=chat_id,
                            game_id=str(chat_id),
                            scoreboard={chat_id: 0},
                        )
                        _store_state(game_state)
                        logger.info("Generated puzzle ready for delivery")
                        return puzzle, game_state

        raise RuntimeError("Не удалось сформировать кроссворд из сгенерированных слов")


# ---------------------------------------------------------------------------
# Telegram command handlers
# ---------------------------------------------------------------------------


@command_entrypoint(fallback=ConversationHandler.END)
async def start_new_game(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    _normalise_thread_id(update)
    if not await _reject_group_chat(update):
        return ConversationHandler.END
    chat = update.effective_chat
    message = update.effective_message
    chat_id = chat.id if chat else None
    logger.debug("Chat %s initiated /new", chat_id if chat_id is not None else "<unknown>")

    if chat_id is not None and chat_id in state.generating_chats:
        if message:
            await message.reply_text(
                "Мы всё ещё готовим ваш кроссворд. Пожалуйста, подождите."
            )
        return ConversationHandler.END

    if chat_id is not None:
        game_state = _load_state_for_chat(chat_id)
    else:
        game_state = None

    puzzle: Puzzle | CompositePuzzle | None = None
    if game_state is not None:
        puzzle = _load_puzzle_for_state(game_state)

    if (
        game_state is not None
        and puzzle is not None
        and not _all_slots_solved(puzzle, game_state)
    ):
        context.user_data.pop("new_game_language", None)
        set_chat_mode(context, MODE_IN_GAME)
        reminder_text = (
            "У вас уже есть активный кроссворд. Давайте продолжим текущую игру!"
        )
        if message:
            await message.reply_text(reminder_text)
        try:
            with logging_context(puzzle_id=puzzle.id):
                image_path = render_puzzle(puzzle, game_state)
                await context.bot.send_chat_action(
                    chat_id=chat_id,
                    action=constants.ChatAction.UPLOAD_PHOTO,
                )
                if message:
                    with open(image_path, "rb") as photo:
                        await message.reply_photo(photo=photo)
                if message:
                    await message.reply_text(
                        _format_clues_message(puzzle, game_state),
                        parse_mode=constants.ParseMode.HTML,
                    )
        except Exception:  # noqa: BLE001
            logger.exception("Failed to resend active puzzle to chat %s", chat_id)
            if message:
                await message.reply_text(
                    "Не удалось показать текущее состояние, но игра продолжается."
                )
        return ConversationHandler.END

    context.user_data["new_game_language"] = None
    set_chat_mode(context, MODE_AWAIT_LANGUAGE)
    if message:
        await message.reply_text(
            "Выберите язык кроссворда (например: ru, en, it, es).",
        )
    return LANGUAGE_STATE


@command_entrypoint(fallback=ConversationHandler.END)
async def handle_language(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    _normalise_thread_id(update)
    if not await _reject_group_chat(update):
        return ConversationHandler.END
    if is_chat_mode_set(context) and get_chat_mode(context) != MODE_AWAIT_LANGUAGE:
        logger.debug(
            "Ignoring language input while in mode %s",
            get_chat_mode(context),
        )
        return LANGUAGE_STATE
    message = update.effective_message
    if message is None or not message.text:
        return LANGUAGE_STATE
    language = message.text.strip().lower()
    if not language or not language.isalpha():
        await message.reply_text("Пожалуйста, введите язык одним словом, например ru.")
        return LANGUAGE_STATE
    logger.debug("Chat %s selected language %s", update.effective_chat.id if update.effective_chat else "<unknown>", language)
    context.user_data["new_game_language"] = language
    set_chat_mode(context, MODE_AWAIT_THEME)
    await message.reply_text("Отлично! Теперь укажите тему кроссворда.")
    return THEME_STATE


@command_entrypoint()
async def button_language_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    _normalise_thread_id(update)
    if not await _reject_group_chat(update):
        return
    if is_chat_mode_set(context) and get_chat_mode(context) != MODE_AWAIT_LANGUAGE:
        logger.debug(
            "Ignoring button language input while in mode %s",
            get_chat_mode(context),
        )
        return
    state = context.chat_data.get(BUTTON_NEW_GAME_KEY)
    if not state or state.get(BUTTON_STEP_KEY) != BUTTON_STEP_LANGUAGE:
        return
    message = update.effective_message
    if message is None or not message.text:
        return
    language = message.text.strip().lower()
    if not language or not language.isalpha():
        await message.reply_text("Пожалуйста, введите язык одним словом, например ru.")
        return
    state[BUTTON_LANGUAGE_KEY] = language
    state[BUTTON_STEP_KEY] = BUTTON_STEP_THEME
    set_chat_mode(context, MODE_AWAIT_THEME)
    logger.debug(
        "Chat %s selected language %s via button flow",
        update.effective_chat.id if update.effective_chat else "<unknown>",
        language,
    )
    await message.reply_text("Отлично! Теперь укажите тему кроссворда.")


@command_entrypoint(fallback=ConversationHandler.END)
async def handle_theme(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    _normalise_thread_id(update)
    if not await _reject_group_chat(update):
        set_chat_mode(context, MODE_IDLE)
        context.user_data.pop("new_game_language", None)
        return ConversationHandler.END
    if is_chat_mode_set(context) and get_chat_mode(context) != MODE_AWAIT_THEME:
        logger.debug(
            "Ignoring theme input while in mode %s",
            get_chat_mode(context),
        )
        return THEME_STATE
    message = update.effective_message
    chat = update.effective_chat
    if message is None or chat is None or not message.text:
        return THEME_STATE
    if chat.id in state.generating_chats:
        await message.reply_text(
            "Мы всё ещё готовим ваш кроссворд. Пожалуйста, подождите."
        )
        set_chat_mode(context, MODE_IDLE)
        context.user_data.pop("new_game_language", None)
        return ConversationHandler.END
    language = context.user_data.get("new_game_language")
    if not language:
        await message.reply_text("Сначала выберите язык через команду /new.")
        set_chat_mode(context, MODE_IDLE)
        context.user_data.pop("new_game_language", None)
        return ConversationHandler.END
    theme = message.text.strip()
    if not theme:
        await message.reply_text("Введите тему, например: Древний Рим.")
        return THEME_STATE
    logger.info("Chat %s selected theme %s", chat.id, theme)
    _cancel_reminder(context)
    await _send_generation_notice(
        context,
        chat.id,
        "Готовлю кроссворд, это может занять немного времени...",
        message=message,
    )
    loop = asyncio.get_running_loop()
    try:
        puzzle: Puzzle | CompositePuzzle | None = None
        game_state: GameState | None = None
        state.generating_chats.add(chat.id)
        try:
            puzzle, game_state = await loop.run_in_executor(
                None, _generate_puzzle, chat.id, language, theme
            )
        except Exception:  # noqa: BLE001
            logger.exception("Failed to generate puzzle for chat %s", chat.id)
            _cleanup_chat_resources(chat.id)
            _clear_generation_notice(context, chat.id)
            await message.reply_text(
                "Сейчас не получилось подготовить кроссворд. Попробуйте выполнить /new чуть позже."
            )
            set_chat_mode(context, MODE_IDLE)
            return ConversationHandler.END
        finally:
            state.generating_chats.discard(chat.id)

        set_chat_mode(context, MODE_IN_GAME)
        delivered = await _deliver_puzzle_via_bot(context, chat.id, puzzle, game_state)
        if not delivered:
            set_chat_mode(context, MODE_IDLE)
            _cleanup_game_state(game_state)
            _clear_generation_notice(context, chat.id)
            await message.reply_text(
                "Возникла ошибка при подготовке кроссворда. Попробуйте начать новую игру командой /new."
            )
            return ConversationHandler.END

        if context.job_queue:
            job = context.job_queue.run_once(
                _reminder_job,
                REMINDER_DELAY_SECONDS,
                chat_id=chat.id,
                name=f"hint-reminder-{chat.id}",
            )
            context.chat_data["reminder_job"] = job

        _clear_generation_notice(context, chat.id)
        logger.info("Delivered freshly generated puzzle to chat")
        return ConversationHandler.END
    finally:
        context.user_data.pop("new_game_language", None)


@command_entrypoint()
async def button_theme_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    _normalise_thread_id(update)
    if not await _reject_group_chat(update):
        return
    if is_chat_mode_set(context) and get_chat_mode(context) != MODE_AWAIT_THEME:
        logger.debug(
            "Ignoring button theme input while in mode %s",
            get_chat_mode(context),
        )
        return
    flow_state = context.chat_data.get(BUTTON_NEW_GAME_KEY)
    if not flow_state or flow_state.get(BUTTON_STEP_KEY) != BUTTON_STEP_THEME:
        return
    message = update.effective_message
    chat = update.effective_chat
    if message is None or not message.text or chat is None:
        return
    if chat.id in state.generating_chats:
        await message.reply_text(
            "Мы всё ещё готовим ваш предыдущий кроссворд. Пожалуйста, подождите."
        )
        return
    language = flow_state.get(BUTTON_LANGUAGE_KEY)
    if not language:
        await message.reply_text("Сначала выберите язык через команду /new.")
        flow_state[BUTTON_STEP_KEY] = BUTTON_STEP_LANGUAGE
        return
    theme = message.text.strip()
    if not theme:
        await message.reply_text("Введите тему, например: Древний Рим.")
        return
    logger.info("Chat %s requested theme '%s' via button flow", chat.id, theme)
    _cancel_reminder(context)
    await _send_generation_notice(
        context,
        chat.id,
        "Готовлю кроссворд, это может занять немного времени...",
        message=message,
    )
    loop = asyncio.get_running_loop()
    puzzle: Puzzle | CompositePuzzle | None = None
    game_state: GameState | None = None
    state.generating_chats.add(chat.id)
    try:
        puzzle, game_state = await loop.run_in_executor(
            None, _generate_puzzle, chat.id, language, theme
        )
    except Exception:  # noqa: BLE001
        logger.exception("Failed to generate puzzle for chat %s via button flow", chat.id)
        _cleanup_chat_resources(chat.id)
        context.chat_data.pop(BUTTON_NEW_GAME_KEY, None)
        _clear_generation_notice(context, chat.id)
        await message.reply_text(
            "Сейчас не получилось подготовить кроссворд. Попробуйте выполнить /new чуть позже."
        )
        return
    finally:
        state.generating_chats.discard(chat.id)
    context.chat_data.pop(BUTTON_NEW_GAME_KEY, None)
    set_chat_mode(context, MODE_IN_GAME)
    delivered = await _deliver_puzzle_via_bot(context, chat.id, puzzle, game_state)
    if not delivered:
        set_chat_mode(context, MODE_IDLE)
        _cleanup_game_state(game_state)
        _clear_generation_notice(context, chat.id)
        await message.reply_text(
            "Возникла ошибка при подготовке кроссворда. Попробуйте начать новую игру командой /new."
        )
        return
    if context.job_queue:
        job = context.job_queue.run_once(
            _reminder_job,
            REMINDER_DELAY_SECONDS,
            chat_id=chat.id,
            name=f"hint-reminder-{chat.id}",
        )
        context.chat_data["reminder_job"] = job

    _clear_generation_notice(context, chat.id)


@command_entrypoint(fallback=ConversationHandler.END)
async def cancel_new_game(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    _normalise_thread_id(update)
    context.user_data.pop("new_game_language", None)
    set_chat_mode(context, MODE_IDLE)
    chat = update.effective_chat
    if chat is not None:
        _clear_generation_notice(context, chat.id)
    if update.effective_message:
        await update.effective_message.reply_text("Создание кроссворда отменено.")
    return ConversationHandler.END


@command_entrypoint()
async def send_clues(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    _normalise_thread_id(update)
    if not await _reject_group_chat(update):
        return
    chat = update.effective_chat
    message = update.effective_message
    if chat is None or message is None:
        return
    logger.debug("Chat %s requested /clues", chat.id)
    if is_chat_mode_set(context) and get_chat_mode(context) != MODE_IN_GAME:
        await message.reply_text("Нет активной игры. Используйте /new для начала.")
        return
    game_state = _load_state_for_chat(chat.id)
    if not game_state:
        await message.reply_text("Нет активной игры. Используйте /new для начала.")
        return
    puzzle = _load_puzzle_for_state(game_state)
    if puzzle is None:
        await message.reply_text("Не удалось загрузить кроссворд. Попробуйте начать новую игру.")
        return
    with logging_context(puzzle_id=puzzle.id):
        logger.info("Sending clues to chat")
        await message.reply_text(
            _format_clues_message(puzzle, game_state),
            parse_mode=constants.ParseMode.HTML,
        )


@command_entrypoint()
async def send_state_image(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    _normalise_thread_id(update)
    if not await _reject_group_chat(update):
        return
    chat = update.effective_chat
    message = update.effective_message
    if chat is None or message is None:
        return
    logger.debug("Chat %s requested /state", chat.id)
    if is_chat_mode_set(context) and get_chat_mode(context) != MODE_IN_GAME:
        await message.reply_text("Нет активного кроссворда. Используйте /new.")
        return
    game_state = _load_state_for_chat(chat.id)
    if not game_state:
        await message.reply_text("Нет активного кроссворда. Используйте /new.")
        return
    puzzle = _load_puzzle_for_state(game_state)
    if puzzle is None:
        await message.reply_text("Кроссворд не найден. Попробуйте начать новую игру.")
        return
    try:
        with logging_context(puzzle_id=puzzle.id):
            image_path = render_puzzle(puzzle, game_state)
            await context.bot.send_chat_action(
                chat_id=chat.id, action=constants.ChatAction.UPLOAD_PHOTO
            )
            with open(image_path, "rb") as photo:
                await message.reply_photo(photo=photo)
            logger.info("Sent current puzzle state image")
    except Exception:  # noqa: BLE001
        logger.exception("Failed to render state image for chat %s", chat.id)
        await message.reply_text("Не удалось подготовить изображение. Попробуйте позже.")


async def _handle_answer_submission(
    context: ContextTypes.DEFAULT_TYPE,
    chat: Chat,
    message: Message,
    slot_id: str,
    raw_answer: str,
) -> None:
    normalised_slot_id = _normalise_slot_id(slot_id)
    answer_text = raw_answer.strip()
    user = getattr(message, "from_user", None)

    def log_abort(
        reason: str,
        *,
        slot_identifier: str | None = None,
        detail: str | None = None,
    ) -> None:
        logger.debug(
            "Answer submission aborted (chat=%s, slot=%s, reason=%s, detail=%s)",
            chat.id,
            slot_identifier or normalised_slot_id,
            reason,
            detail or "-",
        )

    current_mode = get_chat_mode(context)
    if is_chat_mode_set(context) and current_mode != MODE_IN_GAME:
        await message.reply_text("Нет активной игры. Используйте /new.")
        log_abort("invalid_mode", detail=current_mode)
        return

    if not answer_text:
        await message.reply_text("Введите ответ после слота.")
        log_abort("empty_answer_text")
        return

    logger.debug("Chat %s answering slot %s", chat.id, normalised_slot_id)
    game_state = _resolve_game_state_for_message(chat, user.id if user else None)
    if not game_state:
        await message.reply_text("Нет активного кроссворда. Используйте /new.")
        log_abort("missing_game_state")
        return
    puzzle = _load_puzzle_for_state(game_state)
    if puzzle is None:
        await message.reply_text("Не удалось загрузить кроссворд. Попробуйте начать заново.")
        log_abort("missing_puzzle")
        return

    player: Player | None = None
    if user is not None:
        player = _ensure_player_profile(
            game_state,
            user.id,
            name=user.full_name or user.username or str(user.id),
            dm_chat_id=chat.id if chat.type == ChatType.PRIVATE else None,
        )
        _store_state(game_state)

    is_turn_based = game_state.mode == "turn_based"
    if is_turn_based:
        if game_state.status != "running":
            await message.reply_text("Игра ещё не запущена. Подождите старта.")
            log_abort("turn_based_not_running")
            return
        if not _is_current_player(game_state, user.id if user else None):
            current_player = _current_player(game_state)
            notice = (
                f"Сейчас ход {current_player.name}."
                if current_player
                else "Сейчас ход другого игрока."
            )
            await message.reply_text(notice)
            log_abort("not_current_player")
            return

    with logging_context(puzzle_id=puzzle.id):
        async def refresh_clues_if_needed() -> None:
            await _send_clues_update(message, puzzle, game_state)

        solved_ids = {_normalise_slot_id(entry) for entry in game_state.solved_slots}
        is_numeric_slot = normalised_slot_id.isdigit()
        candidate_refs: list[SlotRef] = []
        selected_slot_ref: SlotRef | None = None
        slot_identifier = normalised_slot_id

        if is_numeric_slot:
            slot_number: int | None = None
            try:
                slot_number = int(normalised_slot_id)
            except ValueError:
                slot_number = None
            if slot_number is not None:
                candidate_refs = [
                    ref
                    for ref in iter_slot_refs(puzzle)
                    if ref.slot.number == slot_number
                ]
            if not candidate_refs:
                logger.warning("Answer received for missing slot %s", normalised_slot_id)
                await message.reply_text(f"Слот {normalised_slot_id} не найден.")
                await refresh_clues_if_needed()
                log_abort("slot_not_found")
                return
        else:
            resolved_slot_ref, ambiguity = _resolve_slot(puzzle, normalised_slot_id)
            if ambiguity:
                await message.reply_text(ambiguity)
                await refresh_clues_if_needed()
                log_abort("slot_reference_ambiguous", detail=ambiguity)
                return
            if resolved_slot_ref is None:
                logger.warning("Answer received for missing slot %s", normalised_slot_id)
                await message.reply_text(f"Слот {normalised_slot_id} не найден.")
                await refresh_clues_if_needed()
                log_abort("slot_not_found")
                return
            selected_slot_ref = resolved_slot_ref
            slot_identifier = _normalise_slot_id(resolved_slot_ref.public_id)
            if slot_identifier in solved_ids:
                await message.reply_text("Этот слот уже решён.")
                await refresh_clues_if_needed()
                log_abort("slot_already_solved", slot_identifier=slot_identifier)
                return
            if not resolved_slot_ref.slot.answer:
                await message.reply_text("Для этого слота не задан ответ.")
                await refresh_clues_if_needed()
                log_abort("slot_has_no_answer", slot_identifier=slot_identifier)
                return
            candidate_refs = [resolved_slot_ref]

        try:
            validated = validate_word_list(
                puzzle.language,
                [WordClue(word=answer_text, clue="")],
                deduplicate=False,
            )
        except WordValidationError as exc:
            logger.warning(
                "Rejected answer for slot %s due to validation: %s",
                slot_identifier,
                exc,
            )
            await message.reply_text(f"Слово не прошло проверку: {exc}")
            await refresh_clues_if_needed()
            log_abort(
                "answer_validation_failed",
                slot_identifier=slot_identifier,
                detail=str(exc),
            )
            return
        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "Unexpected error validating answer for slot %s",
                slot_identifier,
            )
            await message.reply_text("Не удалось проверить слово. Попробуйте позже.")
            await refresh_clues_if_needed()
            log_abort(
                "answer_validation_error",
                slot_identifier=slot_identifier,
                detail=str(exc),
            )
            return

        if not validated:
            logger.info("Answer for slot %s failed language rules", slot_identifier)
            await message.reply_text("Слово не соответствует правилам языка.")
            await refresh_clues_if_needed()
            log_abort("answer_not_validated", slot_identifier=slot_identifier)
            return

        candidate = validated[0].word
        candidate_canonical = _canonical_answer(candidate, puzzle.language)

        if is_numeric_slot:
            answerable_refs = [ref for ref in candidate_refs if ref.slot.answer]
            if not answerable_refs:
                await message.reply_text("Для этого слота не задан ответ.")
                await refresh_clues_if_needed()
                log_abort("slot_has_no_answer", slot_identifier=normalised_slot_id)
                return

            matching_refs = [
                ref
                for ref in answerable_refs
                if _canonical_answer(ref.slot.answer, puzzle.language)
                == candidate_canonical
            ]
            if not matching_refs:
                logger.info(
                    "Incorrect answer for slot number %s",
                    normalised_slot_id,
                )
                await message.reply_text("Ответ неверный, попробуйте ещё раз.")
                await refresh_clues_if_needed()
                log_abort(
                    "answer_incorrect",
                    slot_identifier=normalised_slot_id,
                )
                return

            matching_refs.sort(
                key=lambda ref: (
                    0
                    if _normalise_slot_id(ref.public_id) not in solved_ids
                    else 1,
                    0 if ref.slot.direction is Direction.ACROSS else 1,
                    _normalise_slot_id(ref.public_id),
                )
            )
            selected_slot_ref = matching_refs[0]
            slot_identifier = _normalise_slot_id(selected_slot_ref.public_id)
            if slot_identifier in solved_ids:
                await message.reply_text("Этот слот уже решён.")
                await refresh_clues_if_needed()
                log_abort("slot_already_solved", slot_identifier=slot_identifier)
                return

        if selected_slot_ref is None:
            logger.warning("Failed to resolve slot %s after validation", normalised_slot_id)
            await message.reply_text(f"Слот {normalised_slot_id} не найден.")
            await refresh_clues_if_needed()
            log_abort("slot_not_found", slot_identifier=normalised_slot_id)
            return

        slot = selected_slot_ref.slot
        public_id = _normalise_slot_id(selected_slot_ref.public_id)

        if is_turn_based and game_state.active_slot_id and game_state.active_slot_id != public_id:
            await message.reply_text(
                f"Сначала ответьте на выбранный слот {game_state.active_slot_id}.",
            )
            log_abort("slot_mismatch", slot_identifier=public_id)
            return

        if _canonical_answer(candidate, puzzle.language) != _canonical_answer(
            slot.answer,
            puzzle.language,
        ):
            logger.info("Incorrect answer for slot %s", selected_slot_ref.public_id)
            await message.reply_text("Ответ неверный, попробуйте ещё раз.")
            await refresh_clues_if_needed()
            log_abort("answer_incorrect", slot_identifier=public_id)
            if is_turn_based:
                if player is not None:
                    player.answers_fail += 1
                _store_state(game_state)
                try:
                    await context.bot.send_message(
                        chat_id=game_state.chat_id,
                        text=(
                            f"Ответ игрока {player.name if player else (user.full_name if user else 'игрок')}"
                            f" на {public_id} неверный."
                        ),
                    )
                except Exception:  # noqa: BLE001
                    logger.exception("Failed to announce incorrect answer in chat %s", game_state.chat_id)
                await _complete_turn(context, game_state, reason="incorrect")
            return

        game_state.score += slot.length
        _record_score(game_state, slot.length, user_id=user.id if user else None)
        if player is not None:
            player.answers_ok += 1
        game_state.active_slot_id = None
        _apply_answer_to_state(game_state, selected_slot_ref, candidate)
        logger.info("Accepted answer for slot %s", selected_slot_ref.public_id)

        try:
            image_path = render_puzzle(puzzle, game_state)
            await context.bot.send_chat_action(
                chat_id=chat.id, action=constants.ChatAction.UPLOAD_PHOTO
            )
            with open(image_path, "rb") as photo:
                await message.reply_photo(
                    photo=photo, caption=f"Верно! {selected_slot_ref.public_id}"
                )
        except Exception:  # noqa: BLE001
            logger.exception("Failed to render updated grid after correct answer")
            await message.reply_text(
                "Ответ принят, но не удалось обновить изображение. Попробуйте команду /state позже."
            )

        if _all_slots_solved(puzzle, game_state):
            if is_turn_based:
                await _finish_game(context, game_state, reason="completed", puzzle=puzzle)
            else:
                _cancel_reminder(context)
                set_chat_mode(context, MODE_IDLE)
                await message.reply_text(
                    "🎉 <b>Поздравляем!</b>\nВсе слова разгаданы! ✨",
                    parse_mode=constants.ParseMode.HTML,
                )
                await _send_completion_options(context, chat.id, message, puzzle)
        else:
            await refresh_clues_if_needed()
            if is_turn_based:
                try:
                    await context.bot.send_message(
                        chat_id=game_state.chat_id,
                        text=(
                            f"{player.name if player else (user.full_name if user else 'Игрок')}"
                            f" разгадал {selected_slot_ref.public_id}!"
                        ),
                    )
                except Exception:  # noqa: BLE001
                    logger.exception("Failed to announce correct answer in chat %s", game_state.chat_id)
                await _complete_turn(context, game_state, reason="correct")


@command_entrypoint()
async def admin_answer_request_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    _normalise_thread_id(update)
    if not await _reject_group_chat(update):
        return

    message = update.effective_message
    chat = update.effective_chat
    user = update.effective_user
    if chat is None or message is None or user is None or not message.text:
        return

    settings = state.settings
    if settings is None or settings.admin_id is None:
        return
    if user.id != settings.admin_id:
        logger.debug("Ignoring admin request from non-admin user %s", user.id)
        return

    text = message.text.strip()
    if not text:
        return

    game_state = _load_state_for_chat(chat.id)
    if not game_state:
        await message.reply_text("Нет активной игры для выдачи ответов.")
        return

    puzzle = _load_puzzle_for_state(game_state)
    if puzzle is None:
        await message.reply_text("Не удалось загрузить кроссворд для ответов.")
        return

    if ADMIN_KEYS_ONLY_PATTERN.match(text):
        logger.info("Admin requested all answers for chat %s", chat.id)
        await message.reply_text(_format_admin_answers(puzzle))
        return

    single_match = ADMIN_SINGLE_KEY_PATTERN.match(text)
    if not single_match:
        return

    query = single_match.group(1).strip()
    if not query:
        await message.reply_text("Укажите номер вопроса после команды.")
        return

    slot_ref, ambiguity = _resolve_slot(puzzle, query)
    if slot_ref:
        logger.info("Admin requested answer for slot %s", slot_ref.public_id)
        await message.reply_text(_format_slot_answers([slot_ref]))
        return

    if ambiguity:
        await message.reply_text(ambiguity)
        return

    if query.isdigit():
        number = int(query)
        matches = [ref for ref in _sorted_slot_refs(puzzle) if ref.slot.number == number]
        if matches:
            logger.info(
                "Admin requested answers for number %s (matches: %s)",
                number,
                ", ".join(ref.public_id for ref in matches),
            )
            await message.reply_text(_format_slot_answers(matches))
            return

    await message.reply_text("Вопрос с таким номером не найден.")


@command_entrypoint()
async def answer_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    _normalise_thread_id(update)
    if not await _reject_group_chat(update):
        return
    chat = update.effective_chat
    message = update.effective_message
    if chat is None or message is None:
        return
    if not context.args or len(context.args) < 2:
        await message.reply_text("Использование: /answer <слот> <слово>")
        return

    slot_id = context.args[0]
    raw_answer = " ".join(context.args[1:])
    await _handle_answer_submission(context, chat, message, slot_id, raw_answer)


@command_entrypoint()
async def inline_answer_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    _normalise_thread_id(update)
    if not await _reject_group_chat(update):
        chat = update.effective_chat
        logger.debug(
            "Inline answer handler rejected non-private chat",
            extra={
                "chat_id": chat.id if chat else None,
                "chat_type": chat.type if chat else None,
            },
        )
        return
    chat = update.effective_chat
    message = update.effective_message
    current_mode = get_chat_mode(context)
    if is_chat_mode_set(context) and current_mode != MODE_IN_GAME:
        if "new_game_language" in context.user_data:
            chat_id = chat.id if chat else None
            if chat is None or message is None:
                logger.info(
                    "Skipping inline answer: /new conversation active but no message available",
                    extra={"chat_id": chat_id, "has_message": message is not None},
                )
                return

            if chat_id in state.generating_chats:
                logger.info(
                    "Skipping inline answer: setup or generation is in progress",
                    extra={
                        "chat_id": chat_id,
                        "message_id": message.message_id,
                        "generating": True,
                    },
                )
            else:
                logger.info(
                    "Skipping inline answer: language/theme selection is in progress",
                    extra={
                        "chat_id": chat_id,
                        "message_id": message.message_id,
                        "generating": False,
                    },
                )

            await message.reply_text(
                "Сначала завершите выбор языка/темы или дождитесь подготовки кроссворда."
                " Если хотите прервать, отправьте /cancel."
            )
        else:
            logger.debug(
                "Ignoring inline answer while chat %s in mode %s",
                chat.id if chat else None,
                current_mode,
            )
        return
    if not is_chat_mode_set(context) and "new_game_language" in context.user_data:
        chat_id = chat.id if chat else None
        if chat is None or message is None:
            logger.info(
                "Skipping inline answer: /new conversation active but no message available",
                extra={"chat_id": chat_id, "has_message": message is not None},
            )
            return
        await message.reply_text(
            "Сначала завершите выбор языка/темы или дождитесь подготовки кроссворда."
            " Если хотите прервать, отправьте /cancel."
        )
        return

    if chat is None or message is None:
        logger.debug(
            "Inline answer handler aborted: missing chat or message",
            extra={
                "chat_id": chat.id if chat else None,
                "has_message": message is not None,
            },
        )
        return
    raw_text = message.text
    if not raw_text or not raw_text.strip():
        caption = getattr(message, "caption", None)
        if caption and caption.strip():
            raw_text = caption

    if not raw_text or not raw_text.strip():
        logger.debug(
            "Inline answer handler aborted: message text missing",
            extra={
                "chat_id": chat.id,
                "message_id": message.message_id,
            },
        )
        return

    parsed = _parse_inline_answer(raw_text)
    if not parsed:
        user = getattr(update, "effective_user", None)
        game_state = _resolve_game_state_for_message(chat, user.id if user else None)
        if game_state:
            logger.info(
                "Inline answer handler aborted: failed to parse inline answer",
                extra={
                    "chat_id": chat.id,
                    "message_id": message.message_id,
                    "text": raw_text,
                },
            )
            await message.reply_text(
                "Не удалось распознать ответ. Используйте формат «A1 - слово»."
            )
        else:
            logger.debug(
                "Inline answer ignored: no active game and input not recognised",
                extra={
                    "chat_id": chat.id,
                    "message_id": message.message_id,
                    "text": raw_text,
                },
            )
        return

    slot_id, answer_text = parsed
    await _handle_answer_submission(context, chat, message, slot_id, answer_text)


@command_entrypoint()
async def hint_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    _normalise_thread_id(update)
    if not await _reject_group_chat(update):
        return
    chat = update.effective_chat
    message = update.effective_message
    user = update.effective_user
    if chat is None or message is None:
        return
    logger.debug("Chat %s requested /hint", chat.id)
    if is_chat_mode_set(context) and get_chat_mode(context) != MODE_IN_GAME:
        await message.reply_text("Нет активной игры. Используйте /new.")
        return

    game_state = _resolve_game_state_for_message(chat, user.id if user else None)
    if not game_state:
        await message.reply_text("Нет активной игры. Используйте /new.")
        return
    puzzle = _load_puzzle_for_state(game_state)
    if puzzle is None:
        await message.reply_text("Не удалось загрузить кроссворд.")
        return

    player: Player | None = None
    if user is not None:
        player = _ensure_player_profile(
            game_state,
            user.id,
            name=user.full_name or user.username or str(user.id),
            dm_chat_id=chat.id if chat.type == ChatType.PRIVATE else None,
        )
        _store_state(game_state)

    is_turn_based = game_state.mode == "turn_based"
    if is_turn_based:
        if game_state.status != "running":
            await message.reply_text("Игра ещё не запущена. Подождите старта.")
            return
        if not _is_current_player(game_state, user.id if user else None):
            current = _current_player(game_state)
            notice = (
                f"Сейчас ход {current.name}."
                if current
                else "Сейчас ход другого игрока."
            )
            await message.reply_text(notice)
            return

    with logging_context(puzzle_id=puzzle.id):
        slot_ref: Optional[SlotRef] = None
        if context.args:
            slot_ref, ambiguity = _resolve_slot(puzzle, context.args[0])
            if ambiguity:
                await message.reply_text(ambiguity)
                return
            if slot_ref is None:
                await message.reply_text(f"Слот {context.args[0]} не найден.")
                return
        else:
            for candidate in sorted(
                iter_slot_refs(puzzle),
                key=lambda ref: (
                    ref.component_index or 0,
                    ref.slot.direction.value,
                    ref.slot.number,
                ),
            ):
                public_id = _normalise_slot_id(candidate.public_id)
                if public_id in {_normalise_slot_id(entry) for entry in game_state.solved_slots}:
                    continue
                if not candidate.slot.answer:
                    continue
                slot_ref = candidate
                break
            if slot_ref is None:
                await message.reply_text("Нет слотов для подсказки.")
                return

        if not slot_ref.slot.answer:
            await message.reply_text("Для этого слота нет ответа.")
            return

        user_id = user.id if user else None
        penalty_applied = False
        result = _reveal_letter(game_state, slot_ref, slot_ref.slot.answer, user_id=user_id)
        if result is None:
            _record_hint_usage(game_state, slot_ref.public_id, user_id=user_id)
            game_state.last_update = time.time()
            _store_state(game_state)
            reply_text = (
                f"Все буквы в {slot_ref.public_id} уже открыты. Подсказка: {slot_ref.slot.clue or 'нет'}"
            )
            logger.info("Hint requested for already revealed slot %s", slot_ref.public_id)
        else:
            position, letter = result
            reply_text = (
                f"Открыта буква №{position + 1} в {slot_ref.public_id}: {letter}\n"
                f"Подсказка: {slot_ref.slot.clue or 'нет'}"
            )
            logger.info(
                "Revealed letter %s at position %s for slot %s",
                letter,
                position + 1,
                slot_ref.public_id,
            )
        if is_turn_based and TURN_HINT_PENALTY:
            _record_score(game_state, -TURN_HINT_PENALTY, user_id=user_id)
            game_state.score = max(game_state.score - TURN_HINT_PENALTY, 0)
            penalty_applied = True
            _store_state(game_state)

        if penalty_applied:
            name = None
            dm_target = None
            if player is not None:
                name = player.name
                dm_target = player.dm_chat_id
            elif user is not None:
                name = user.full_name or user.username or str(user.id)
                dm_target = chat.id if chat.type == ChatType.PRIVATE else None
            if name is None:
                name = "Игрок"
            penalty_text = f"{name} использует подсказку: -{TURN_HINT_PENALTY} очков."
            try:
                await context.bot.send_message(chat_id=game_state.chat_id, text=penalty_text)
            except Exception:  # noqa: BLE001
                logger.exception("Failed to announce hint penalty in chat %s", game_state.chat_id)
            if dm_target:
                with suppress(Exception):
                    await context.bot.send_message(
                        chat_id=dm_target,
                        text=f"Штраф за подсказку: -{TURN_HINT_PENALTY} очков.",
                    )

        try:
            image_path = render_puzzle(puzzle, game_state)
            await context.bot.send_chat_action(
                chat_id=chat.id, action=constants.ChatAction.UPLOAD_PHOTO
            )
            with open(image_path, "rb") as photo:
                await message.reply_photo(photo=photo, caption=reply_text)
        except Exception:  # noqa: BLE001
            logger.exception("Failed to render grid after hint for slot %s", slot.slot_id)
            await message.reply_text(
                "Подсказка сохранена, но не удалось обновить изображение. Попробуйте /state позже."
            )


@command_entrypoint()
async def solve_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    _normalise_thread_id(update)
    if not await _reject_group_chat(update):
        return
    chat = update.effective_chat
    message = update.effective_message
    if chat is None or message is None:
        return
    logger.debug("Chat %s requested /solve", chat.id)
    if is_chat_mode_set(context) and get_chat_mode(context) != MODE_IN_GAME:
        await message.reply_text("Нет активной игры.")
        return

    game_state = _load_state_for_chat(chat.id)
    if not game_state:
        await message.reply_text("Нет активной игры.")
        return
    puzzle = _load_puzzle_for_state(game_state)
    if puzzle is None:
        await message.reply_text("Кроссворд не найден. Запустите /new.")
        return

    with logging_context(puzzle_id=puzzle.id):
        solved_now = _solve_remaining_slots(game_state, puzzle)
        if not solved_now:
            await message.reply_text("Все ответы уже открыты.")
            return

        _cancel_reminder(context)

        try:
            image_path = render_puzzle(puzzle, game_state)
            await context.bot.send_chat_action(
                chat_id=chat.id, action=constants.ChatAction.UPLOAD_PHOTO
            )
            with open(image_path, "rb") as photo:
                await message.reply_photo(photo=photo, caption="Кроссворд раскрыт полностью.")
        except Exception:  # noqa: BLE001
            logger.exception("Failed to render puzzle after solve command")
            await message.reply_text(
                "Кроссворд решён, но не удалось подготовить изображение. Попробуйте /state позже."
            )
            return

        solved_lines = "\n".join(f"{slot_id}: {answer}" for slot_id, answer in solved_now)
        await message.reply_text(f"Оставшиеся ответы:\n{solved_lines}")
        set_chat_mode(context, MODE_IDLE)
        await _send_completion_options(context, chat.id, message, puzzle)
        logger.info("Revealed remaining slots via /solve (%s entries)", len(solved_now))


@command_entrypoint()
async def finish_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    _normalise_thread_id(update)
    if not await _reject_group_chat(update):
        return
    chat = update.effective_chat
    message = update.effective_message
    user = update.effective_user
    if chat is None or message is None or user is None:
        return
    logger.debug("Chat %s requested /finish", chat.id)
    game_state = _resolve_game_state_for_message(chat, user.id)
    if not game_state:
        await message.reply_text("Нет активной игры.")
        return
    puzzle = _load_puzzle_for_state(game_state)
    if game_state.mode == "turn_based":
        if game_state.host_id and user.id != game_state.host_id:
            await message.reply_text("Завершить игру может только хост.")
            return
        await _finish_game(context, game_state, reason="manual", puzzle=puzzle)
        return
    if puzzle is None:
        await message.reply_text("Не удалось загрузить кроссворд.")
        return
    solved_now = _solve_remaining_slots(game_state, puzzle)
    if solved_now:
        _cancel_reminder(context)
        try:
            image_path = render_puzzle(puzzle, game_state)
            await context.bot.send_chat_action(
                chat_id=chat.id, action=constants.ChatAction.UPLOAD_PHOTO
            )
            with open(image_path, "rb") as photo:
                await message.reply_photo(photo=photo, caption="Игра завершена по команде /finish.")
        except Exception:  # noqa: BLE001
            logger.exception("Failed to render puzzle after finish command")
            await message.reply_text("Игра завершена. Не удалось обновить изображение.")
        solved_lines = "\n".join(f"{slot_id}: {answer}" for slot_id, answer in solved_now)
        await message.reply_text(f"Ответы:\n{solved_lines}")
    else:
        await message.reply_text("Все ответы уже открыты.")
    set_chat_mode(context, MODE_IDLE)
    if puzzle is not None:
        await _send_completion_options(context, chat.id, message, puzzle)
@command_entrypoint()
async def quit_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    _normalise_thread_id(update)
    if not await _reject_group_chat(update):
        return
    chat = update.effective_chat
    message = update.effective_message
    if chat is None or message is None:
        return

    logger.info("Chat %s requested /quit", chat.id)
    if is_chat_mode_set(context) and get_chat_mode(context) != MODE_IN_GAME:
        set_chat_mode(context, MODE_IDLE)
        await message.reply_text("Нет активной игры. Используйте /new, чтобы начать новую сессию.")
        return
    game_state = _load_state_for_chat(chat.id)

    _cancel_reminder(context)

    if game_state is not None:
        _cleanup_game_state(game_state)
    else:
        _cleanup_chat_resources(chat.id)

    set_chat_mode(context, MODE_IDLE)
    await message.reply_text("Сессия завершена. Нажмите /start, чтобы начать заново")


@command_entrypoint()
async def completion_callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    _normalise_thread_id(update)
    query = update.callback_query
    if query is None:
        return
    await query.answer()
    if not await _reject_group_chat(update):
        return
    chat = update.effective_chat
    if chat is None:
        return
    current_mode = get_chat_mode(context)
    if is_chat_mode_set(context) and current_mode == MODE_IN_GAME:
        logger.debug(
            "Ignoring completion callback while chat %s in active game mode",
            chat.id,
        )
        return

    data = (query.data or "").strip()
    message = query.message
    if message is not None:
        with suppress(Exception):
            await message.edit_reply_markup(reply_markup=None)

    if data.startswith(SAME_TOPIC_CALLBACK_PREFIX):
        puzzle_id = data[len(SAME_TOPIC_CALLBACK_PREFIX) :]
        if not puzzle_id:
            await context.bot.send_message(
                chat_id=chat.id,
                text="Не удалось определить предыдущую игру. Используйте /new для начала нового кроссворда.",
            )
            return
        game_state = _load_state_for_chat(chat.id)
        if not game_state or game_state.puzzle_id != puzzle_id:
            await context.bot.send_message(
                chat_id=chat.id,
                text="Похоже, уже начата другая игра. Используйте /new, чтобы начать новый кроссворд.",
            )
            return
        puzzle = _load_puzzle_for_state(game_state)
        if puzzle is None:
            await context.bot.send_message(
                chat_id=chat.id,
                text="Не удалось загрузить предыдущий кроссворд. Попробуйте начать новую игру командой /new.",
            )
            _cleanup_game_state(game_state)
            return
        language = puzzle.language
        theme = puzzle.theme
        logger.info(
            "Chat %s requested another puzzle for same theme (%s, %s)",
            chat.id,
            language,
            theme,
        )
        if chat.id in state.generating_chats:
            await context.bot.send_message(
                chat_id=chat.id,
                text="Мы всё ещё готовим ваш предыдущий кроссворд. Пожалуйста, подождите.",
            )
            return
        _cancel_reminder(context)
        _cleanup_game_state(game_state)
        set_chat_mode(context, MODE_AWAIT_THEME)
        await _send_generation_notice(
            context,
            chat.id,
            f"Готовлю новый кроссворд на тему «{theme}» на языке {language.upper()}...",
        )
        loop = asyncio.get_running_loop()
        new_puzzle: Puzzle | CompositePuzzle | None = None
        new_state: GameState | None = None
        state.generating_chats.add(chat.id)
        try:
            new_puzzle, new_state = await loop.run_in_executor(
                None, _generate_puzzle, chat.id, language, theme
            )
        except Exception:  # noqa: BLE001
            logger.exception(
                "Failed to regenerate puzzle for chat %s on same theme", chat.id
            )
            _cleanup_chat_resources(chat.id)
            _clear_generation_notice(context, chat.id)
            await context.bot.send_message(
                chat_id=chat.id,
                text="Сейчас не получилось подготовить кроссворд. Попробуйте выполнить /new чуть позже.",
            )
            return
        finally:
            state.generating_chats.discard(chat.id)
        set_chat_mode(context, MODE_IN_GAME)
        delivered = await _deliver_puzzle_via_bot(context, chat.id, new_puzzle, new_state)
        if not delivered:
            set_chat_mode(context, MODE_IDLE)
            _cleanup_game_state(new_state)
            _clear_generation_notice(context, chat.id)
            await context.bot.send_message(
                chat_id=chat.id,
                text="Возникла ошибка при подготовке кроссворда. Попробуйте начать новую игру командой /new.",
            )
            return
        if context.job_queue:
            job = context.job_queue.run_once(
                _reminder_job,
                REMINDER_DELAY_SECONDS,
                chat_id=chat.id,
                name=f"hint-reminder-{chat.id}",
            )
            context.chat_data["reminder_job"] = job
        _clear_generation_notice(context, chat.id)
        return

    if data.startswith(NEW_PUZZLE_CALLBACK_PREFIX):
        puzzle_id = data[len(NEW_PUZZLE_CALLBACK_PREFIX) :]
        game_state = _load_state_for_chat(chat.id)
        if game_state and (not puzzle_id or game_state.puzzle_id == puzzle_id):
            _cleanup_game_state(game_state)
        _cancel_reminder(context)
        context.user_data.pop("new_game_language", None)
        context.chat_data[BUTTON_NEW_GAME_KEY] = {BUTTON_STEP_KEY: BUTTON_STEP_LANGUAGE}
        set_chat_mode(context, MODE_AWAIT_LANGUAGE)
        await context.bot.send_message(
            chat_id=chat.id,
            text="Выберите язык кроссворда (например: ru, en, it, es).",
        )
        logger.info("Prompted chat %s to start new puzzle flow", chat.id)
        return

    logger.debug("Unhandled completion callback payload: %s", data)


@command_entrypoint()
async def lobby_start_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    if query is None:
        return
    await query.answer()
    chat = query.message.chat if query.message else update.effective_chat
    user = query.from_user
    if chat is None or user is None:
        return
    game_state = _resolve_game_state_for_message(chat, user.id)
    if not game_state:
        await context.bot.send_message(chat_id=chat.id, text="Игра не найдена.")
        return
    if game_state.mode != "turn_based":
        await context.bot.send_message(chat_id=chat.id, text="Этот режим не поддерживает запуск из лобби.")
        return
    if game_state.status != "lobby":
        await context.bot.send_message(chat_id=chat.id, text="Игра уже запущена.")
        return
    if game_state.host_id and user.id != game_state.host_id:
        await context.bot.send_message(chat_id=chat.id, text="Только хост может запустить игру.")
        return
    _initialise_turn_based_mode(context, game_state)
    await context.bot.send_message(chat_id=chat.id, text="Игра началась!")
    await _announce_current_turn(context, game_state)


@command_entrypoint()
async def turn_select_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    if query is None:
        return
    await query.answer()
    message_chat = query.message.chat if query.message else update.effective_chat
    user = query.from_user
    if message_chat is None or user is None:
        return
    game_state = _resolve_game_state_for_message(message_chat, user.id)
    if not game_state or game_state.mode != "turn_based":
        await context.bot.send_message(chat_id=message_chat.id, text="Нет активной пошаговой игры.")
        return
    if game_state.status != "running":
        await context.bot.send_message(chat_id=message_chat.id, text="Игра ещё не запущена.")
        return
    if not _is_current_player(game_state, user.id):
        current = _current_player(game_state)
        if current:
            await query.answer(f"Сейчас ход {current.name}", show_alert=True)
        return
    puzzle = _load_puzzle_for_state(game_state)
    if puzzle is None:
        await query.answer("Кроссворд не найден", show_alert=True)
        return
    unsolved: list[SlotRef] = []
    solved_lookup = {_normalise_slot_id(entry) for entry in game_state.solved_slots}
    for slot_ref in iter_slot_refs(puzzle):
        if not slot_ref.slot.answer:
            continue
        public_id = _normalise_slot_id(slot_ref.public_id)
        if public_id in solved_lookup:
            continue
        unsolved.append(slot_ref)
    if not unsolved:
        await query.answer("Все слова уже разгаданы", show_alert=True)
        return
    player = _ensure_player_profile(
        game_state,
        user.id,
        name=user.full_name or user.username or str(user.id),
        dm_chat_id=message_chat.id if message_chat.type == ChatType.PRIVATE else None,
    )
    _store_state(game_state)
    buttons: list[list[InlineKeyboardButton]] = []
    row: list[InlineKeyboardButton] = []
    for idx, slot_ref in enumerate(unsolved, start=1):
        callback_data = f"{TURN_SLOT_CALLBACK_PREFIX}{slot_ref.public_id}"
        row.append(InlineKeyboardButton(slot_ref.public_id, callback_data=callback_data))
        if idx % 3 == 0:
            buttons.append(row)
            row = []
    if row:
        buttons.append(row)
    keyboard = InlineKeyboardMarkup(buttons)
    target_chat_id = player.dm_chat_id or message_chat.id
    await context.bot.send_message(
        chat_id=target_chat_id,
        text="Выберите слот для ответа:",
        reply_markup=keyboard,
    )


@command_entrypoint()
async def turn_slot_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    if query is None:
        return
    data = query.data or ""
    if not data.startswith(TURN_SLOT_CALLBACK_PREFIX):
        return
    slot_id = data[len(TURN_SLOT_CALLBACK_PREFIX) :]
    await query.answer()
    message_chat = query.message.chat if query.message else update.effective_chat
    user = query.from_user
    if message_chat is None or user is None:
        return
    game_state = _resolve_game_state_for_message(message_chat, user.id)
    if not game_state or game_state.mode != "turn_based":
        await context.bot.send_message(chat_id=message_chat.id, text="Нет активной пошаговой игры.")
        return
    if game_state.status != "running":
        await context.bot.send_message(chat_id=message_chat.id, text="Игра ещё не запущена.")
        return
    if not _is_current_player(game_state, user.id):
        current = _current_player(game_state)
        if current:
            await query.answer(f"Сейчас ход {current.name}", show_alert=True)
        return
    puzzle = _load_puzzle_for_state(game_state)
    if puzzle is None:
        await query.answer("Кроссворд не найден", show_alert=True)
        return
    slot_ref, ambiguity = _resolve_slot(puzzle, slot_id)
    if ambiguity:
        await query.answer(ambiguity, show_alert=True)
        return
    if slot_ref is None or not slot_ref.slot.answer:
        await query.answer("Слот не найден", show_alert=True)
        return
    public_id = _normalise_slot_id(slot_ref.public_id)
    if public_id in {_normalise_slot_id(entry) for entry in game_state.solved_slots}:
        await query.answer("Слот уже решён", show_alert=True)
        return
    player = _ensure_player_profile(
        game_state,
        user.id,
        name=user.full_name or user.username or str(user.id),
        dm_chat_id=message_chat.id if message_chat.type == ChatType.PRIVATE else None,
    )
    game_state.active_slot_id = public_id
    _store_state(game_state)
    clue_text = slot_ref.slot.clue or "(без подсказки)"
    announcement = (
        f"@{user.username}" if user.username else player.name
    ) + f" выбрал слот {slot_ref.public_id}: {clue_text}"
    await context.bot.send_message(chat_id=game_state.chat_id, text=announcement)
    if player.dm_chat_id:
        with suppress(Exception):
            await context.bot.send_message(
                chat_id=player.dm_chat_id,
                text=f"Вы выбрали {slot_ref.public_id}. Отправьте ответ командой /answer или сообщением.",
            )

def configure_telegram_handlers(telegram_application: Application) -> None:
    conversation = ConversationHandler(
        entry_points=[CommandHandler(["new", "start"], start_new_game)],
        states={
            LANGUAGE_STATE: [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_language)],
            THEME_STATE: [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_theme)],
        },
        fallbacks=[CommandHandler("cancel", cancel_new_game)],
        name="new_game_conversation",
    )
    telegram_application.add_handler(conversation)
    telegram_application.add_handler(
    MessageHandler(filters.Regex(ADMIN_COMMAND_PATTERN), admin_answer_request_handler)
)
    telegram_application.add_handler(
    MessageHandler(
        filters.TEXT & ~filters.COMMAND,
        inline_answer_handler,
        block=False
    )
)
    telegram_application.add_handler(
        MessageHandler(
            filters.TEXT & ~filters.COMMAND,
            button_language_handler,
            block=False
        )
    )
    telegram_application.add_handler(
        MessageHandler(
            filters.TEXT & ~filters.COMMAND,
            button_theme_handler,
            block=False
        )
    )
    telegram_application.add_handler(CommandHandler("clues", send_clues))
    telegram_application.add_handler(CommandHandler("state", send_state_image))
    telegram_application.add_handler(CommandHandler("answer", answer_command))
    telegram_application.add_handler(CommandHandler(["hint", "open"], hint_command))
    telegram_application.add_handler(CommandHandler("solve", solve_command))
    telegram_application.add_handler(CommandHandler("finish", finish_command))
    telegram_application.add_handler(CommandHandler("quit", quit_command))
    telegram_application.add_handler(CommandHandler("cancel", cancel_new_game))
    telegram_application.add_handler(
        CallbackQueryHandler(
            completion_callback_handler,
            pattern=fr"^{COMPLETION_CALLBACK_PREFIX}",
        )
    )
    telegram_application.add_handler(
        CallbackQueryHandler(
            lobby_start_handler,
            pattern=fr"^{LOBBY_START_CALLBACK}$",
        )
    )
    telegram_application.add_handler(
        CallbackQueryHandler(
            turn_select_handler,
            pattern=fr"^{TURN_SELECT_CALLBACK}$",
        )
    )
    telegram_application.add_handler(
        CallbackQueryHandler(
            turn_slot_handler,
            pattern=fr"^{TURN_SLOT_CALLBACK_PREFIX}.*",
        )
    )


# ---------------------------------------------------------------------------
# Webhook monitoring
# ---------------------------------------------------------------------------


async def monitor_webhook(application: Application, settings: Settings) -> None:
    """Background task to periodically ensure webhook registration is valid."""

    logger.debug("Starting webhook monitor task with interval %s seconds", settings.webhook_check_interval)
    expected_url = f"{settings.public_url}{settings.webhook_path}"
    while True:
        try:
            info = await application.bot.get_webhook_info()
            logger.debug("Current webhook info: url=%s, pending=%s", info.url, info.pending_update_count)
            current_secret = getattr(info, "secret_token", None)
            if info.url != expected_url or (current_secret and current_secret != settings.webhook_secret):
                logger.warning("Webhook mismatch detected. Expected url=%s secret token=%s", expected_url, settings.webhook_secret)
                await application.bot.set_webhook(
                    url=expected_url,
                    secret_token=settings.webhook_secret,
                    allowed_updates=["message","callback_query","chat_member","my_chat_member","message_reaction","message_reaction_count"],
                )
                logger.info("Webhook re-registered due to mismatch")
        except Exception:  # noqa: BLE001 - We want to log all failures
            logger.exception("Failed to validate or reset webhook")

        await asyncio.sleep(settings.webhook_check_interval)


# ---------------------------------------------------------------------------
# Game state restoration and cleanup
# ---------------------------------------------------------------------------


async def cleanup_states_periodically(app_state: AppState) -> None:
    """Periodically prune expired game states from memory and disk."""

    logger.debug(
        "Starting state cleanup task with interval %s seconds", STATE_CLEANUP_INTERVAL
    )
    while True:
        try:
            expired_states = prune_expired_states(app_state.active_games)
            if expired_states:
                for expired_state in expired_states:
                    _cleanup_game_state(expired_state)
                logger.info("Removed %s expired game states", len(expired_states))
        except Exception:  # noqa: BLE001 - log any cleanup issues
            logger.exception("State cleanup task encountered an error")

        await asyncio.sleep(STATE_CLEANUP_INTERVAL)


# ---------------------------------------------------------------------------
# Lifecycle events
# ---------------------------------------------------------------------------


@app.on_event("startup")
async def on_startup() -> None:
    logger.debug("FastAPI startup initiated")
    ensure_storage_directories()

    restored_games = load_all_states()
    state.active_games = restored_games
    state.chat_to_game = {game_state.chat_id: game_id for game_id, game_state in restored_games.items()}
    state.join_codes = {}
    for game_state in restored_games.values():
        for code, target in game_state.join_codes.items():
            state.join_codes[code] = target
    if restored_games:
        logger.info("Restored %s active game states", len(restored_games))
    else:
        logger.debug("No persisted game states found during startup")

    settings = load_settings()
    state.settings = settings

    logger.debug("Building Telegram application")
    # python-telegram-bot handles authentication via the provided bot token, so we do not
    # need to inject custom authorization headers. Creating the request without extra
    # kwargs keeps compatibility with future library versions.
    httpx_request = HTTPXRequest()
    telegram_application = (
        Application.builder()
        .token(settings.telegram_bot_token)
        .request(httpx_request)
        .updater(None)
        .build()
    )

    configure_telegram_handlers(telegram_application)

    await telegram_application.initialize()
    logger.info("Telegram application initialized")

    await telegram_application.start()
    logger.info("Telegram application started")

    state.telegram_app = telegram_application

    register_webhook_route(settings.webhook_path)

    expected_url = f"{settings.public_url}{settings.webhook_path}"
    logger.debug("Ensuring webhook is configured for %s", expected_url)
    await telegram_application.bot.set_webhook(
        url=expected_url,
        secret_token=settings.webhook_secret,
        allowed_updates=["message","callback_query","chat_member","my_chat_member","message_reaction","message_reaction_count"],
    )
    logger.info("Webhook configured at %s", expected_url)

    state.webhook_task = asyncio.create_task(monitor_webhook(telegram_application, settings))
    state.cleanup_task = asyncio.create_task(cleanup_states_periodically(state))


@app.on_event("shutdown")
async def on_shutdown() -> None:
    logger.debug("FastAPI shutdown initiated")

    if state.cleanup_task:
        logger.debug("Cancelling state cleanup task")
        state.cleanup_task.cancel()
        with suppress(asyncio.CancelledError):
            await state.cleanup_task
        state.cleanup_task = None

    if state.webhook_task:
        logger.debug("Cancelling webhook monitor task")
        state.webhook_task.cancel()
        with suppress(asyncio.CancelledError):
            await state.webhook_task
        state.webhook_task = None

    if state.telegram_app:
        logger.debug("Shutting down Telegram application")

        if getattr(state.telegram_app, "running", False):   
            logger.debug("Stopping Telegram application")
            await state.telegram_app.stop()

        await state.telegram_app.shutdown()
        state.telegram_app = None
        logger.info("Telegram application shut down")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/healthz")
async def healthz() -> JSONResponse:
    logger.debug("Health check requested")
    return JSONResponse({"status": "ok"})


async def telegram_webhook(
    request: Request,
    telegram_application: Application = Depends(get_telegram_application),
) -> JSONResponse:
    settings = state.settings
    if settings is None:
        logger.error("Application settings are not available during webhook call")
        raise HTTPException(status_code=503, detail="Application settings unavailable")

    secret_header = request.headers.get("X-Telegram-Bot-Api-Secret-Token")
    secret_query = request.query_params.get("secret_token")
    if settings.webhook_secret not in {secret_header, secret_query}:
        logger.warning(
            "Webhook secret mismatch: header=%s query=%s",
            secret_header,
            secret_query,
        )
        raise HTTPException(status_code=403, detail="Invalid secret token")

    try:
        payload = await request.json()
        logger.debug("Received webhook payload: %s", payload)
        update = Update.de_json(payload, telegram_application.bot)
    except Exception as exc:  # noqa: BLE001 - we need to report deserialization errors
        logger.exception("Failed to deserialize Telegram update")
        raise HTTPException(status_code=400, detail="Invalid update payload") from exc

    try:
        await telegram_application.process_update(update)
    except Exception as exc:  # noqa: BLE001 - log any processing errors
        logger.exception("Failed to process Telegram update")
        raise HTTPException(status_code=500, detail="Failed to process update") from exc
    logger.debug("Update processed successfully")
    return JSONResponse({"ok": True})


async def _set_webhook(telegram_application: Application, settings: Settings) -> None:
    expected_url = f"{settings.public_url}{settings.webhook_path}"
    logger.debug("Setting webhook to %s", expected_url)
    await telegram_application.bot.set_webhook(
        url=expected_url,
        secret_token=settings.webhook_secret,
        allowed_updates=["message","callback_query","chat_member","my_chat_member","message_reaction","message_reaction_count"],
    )


@app.get("/set_webhook")
async def set_webhook(
    telegram_application: Application = Depends(get_telegram_application),
) -> JSONResponse:
    settings = state.settings
    if settings is None:
        logger.error("Attempted to set webhook without settings available")
        raise HTTPException(status_code=503, detail="Application settings unavailable")

    await _set_webhook(telegram_application, settings)
    return JSONResponse({"status": "webhook set"})


@app.get("/reset_webhook")
async def reset_webhook(
    telegram_application: Application = Depends(get_telegram_application),
) -> JSONResponse:
    settings = state.settings
    if settings is None:
        logger.error("Attempted to reset webhook without settings available")
        raise HTTPException(status_code=503, detail="Application settings unavailable")

    logger.debug("Deleting current webhook before reconfiguration")
    await telegram_application.bot.delete_webhook(drop_pending_updates=False)
    await _set_webhook(telegram_application, settings)
    return JSONResponse({"status": "webhook reset"})


# ---------------------------------------------------------------------------
# Dynamic webhook path mounting
# ---------------------------------------------------------------------------


__all__ = ["app"]

