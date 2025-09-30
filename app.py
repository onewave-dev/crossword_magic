"""FastAPI application entrypoint for Telegram webhook processing."""

from __future__ import annotations

import asyncio
import html
import os
import random
import re
import secrets
import string
import time
from contextlib import suppress
from dataclasses import dataclass
from functools import wraps
from typing import Iterable, Optional, Sequence
from uuid import uuid4

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from telegram import (
    Chat,
    ForceReply,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    KeyboardButton,
    Message,
    ReplyKeyboardMarkup,
    ReplyKeyboardRemove,
    Update,
    User,
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
    Job,
    MessageHandler,
    filters,
)
from telegram.request import HTTPXRequest
from telegram.error import Forbidden, TelegramError

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
        self.lobby_messages: dict[str, tuple[int, int]] = {}
        self.scheduled_jobs: dict[str, Job] = {}
        self.lobby_generation_tasks: dict[str, asyncio.Task[None]] = {}


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
            task = state.lobby_generation_tasks.pop(game_id, None)
            if task is not None:
                task.cancel()
            delete_state(game_id)
            for code, target in list(state.join_codes.items()):
                if target == game_id:
                    state.join_codes.pop(code, None)
            state.lobby_messages.pop(game_id, None)
            if puzzle_id:
                delete_puzzle(puzzle_id)
            logger.info("Cleaned up resources for chat %s", chat_id)
            return

    with logging_context(chat_id=chat_id, puzzle_id=puzzle_id):
        state.generating_chats.discard(chat_id)
        state.chat_to_game.pop(chat_id, None)
        task = state.lobby_generation_tasks.pop(str(chat_id), None)
        if task is not None:
            task.cancel()
        delete_state(chat_id)
        for code, target in list(state.join_codes.items()):
            if target == str(chat_id):
                state.join_codes.pop(code, None)
        state.lobby_messages.pop(str(chat_id), None)
        if puzzle_id:
            delete_puzzle(puzzle_id)
        logger.info("Cleaned up resources for chat %s", chat_id)


def _cleanup_game_state(game_state: GameState | None) -> None:
    if game_state is None:
        return
    with logging_context(chat_id=game_state.chat_id, puzzle_id=game_state.puzzle_id):
        _cancel_job(game_state.turn_timer_job_id)
        _cancel_job(game_state.turn_warn_job_id)
        _cancel_job(game_state.game_timer_job_id)
        _cancel_job(game_state.game_warn_job_id)
        _cancel_job(game_state.dummy_job_id)
        state.generating_chats.discard(game_state.chat_id)
        state.chat_to_game.pop(game_state.chat_id, None)
        state.active_games.pop(game_state.game_id, None)
        task = state.lobby_generation_tasks.pop(game_state.game_id, None)
        if task is not None:
            task.cancel()
        delete_state(game_state)
        for code, target in list(state.join_codes.items()):
            if target == game_state.game_id:
                state.join_codes.pop(code, None)
        for user_id, mapped_chat in list(state.player_chats.items()):
            if mapped_chat == game_state.chat_id:
                state.player_chats.pop(user_id, None)
        state.lobby_messages.pop(game_state.game_id, None)
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


MENU_STATE, LANGUAGE_STATE, THEME_STATE = range(3)

MODE_IDLE = "idle"
MODE_AWAIT_LANGUAGE = "await_language"
MODE_AWAIT_THEME = "await_theme"
MODE_IN_GAME = "in_game"

REMINDER_DELAY_SECONDS = 10 * 60
GENERATION_UPDATE_FIRST_DELAY_SECONDS = 60
GENERATION_UPDATE_INTERVAL_SECONDS = 120
GENERATION_TYPING_INITIAL_DELAY_SECONDS = 5
GENERATION_TYPING_INTERVAL_SECONDS = 25
GENERATION_NOTICE_TEMPLATES = [
    "{base}",
    "{base}\nУстраивайтесь поудобнее, скоро всё пришлю! ✨",
    "Колдую над сеткой... {base}",
    "Подбираю лучшие слова и подсказки. {base}",
]
GENERATION_UPDATE_TEMPLATES = [
    "Подбираю пересечения, чтобы всё сошлось идеально. Спасибо за ожидание!",
    "Проверяю подсказки и ответы — уже почти готово!",
    "Ещё пара штрихов, и кроссворд окажется у вас. Спасибо, что ждёте!",
    "Сверяю сетку и шлифую вопросы. Финальный рывок!",
]
GENERATION_TYPING_ACTIONS = (
    constants.ChatAction.TYPING,
    constants.ChatAction.CHOOSE_STICKER,
)
GAME_TIME_LIMIT_SECONDS = 10 * 60
GAME_WARNING_SECONDS = 60
TURN_TIME_LIMIT_SECONDS = 60
TURN_WARNING_SECONDS = 15
HINT_PENALTY = 1

TURN_SELECT_CALLBACK_PREFIX = "turn_select:"
TURN_SLOT_CALLBACK_PREFIX = "turn_slot:"

MAX_PUZZLE_SIZE = 15
MAX_REPLACEMENT_REQUESTS = 30
# After this many consecutive replacement attempts without a usable word we
# temporarily relax the intersection requirement for LLM suggestions.
SOFT_REPLACEMENT_RELAXATION_THRESHOLD = 3

_ADMIN_FIRST_RAW = os.getenv("ADMIN_FIRST", "false").strip().lower()
ADMIN_FIRST = _ADMIN_FIRST_RAW in {"1", "true", "yes", "on"}

try:
    DUMMY_ACCURACY = float(os.getenv("DUMMY_ACCURACY", "0.8"))
except ValueError:
    DUMMY_ACCURACY = 0.8
DUMMY_ACCURACY = min(max(DUMMY_ACCURACY, 0.0), 1.0)

_DEFAULT_DELAY_MIN = 3.0
_DEFAULT_DELAY_MAX = 8.0
delay_env = os.getenv("DUMMY_DELAY_RANGE")
if delay_env:
    parts = [part.strip() for part in delay_env.split(",") if part.strip()]
    if len(parts) >= 2:
        try:
            delay_min = float(parts[0])
            delay_max = float(parts[1])
        except ValueError:
            delay_min, delay_max = _DEFAULT_DELAY_MIN, _DEFAULT_DELAY_MAX
    else:
        try:
            single = float(parts[0])
            delay_min = max(0.1, single)
            delay_max = delay_min
        except (IndexError, ValueError):
            delay_min, delay_max = _DEFAULT_DELAY_MIN, _DEFAULT_DELAY_MAX
else:
    delay_min, delay_max = _DEFAULT_DELAY_MIN, _DEFAULT_DELAY_MAX
if delay_max < delay_min:
    delay_min, delay_max = delay_max, delay_min
delay_min = max(0.1, delay_min)
delay_max = max(delay_min, delay_max)
DUMMY_DELAY_RANGE = (delay_min, delay_max)

DUMMY_USER_ID = -1
DUMMY_NAME = "Dummy"

ADMIN_TEST_GAME_CALLBACK_PREFIX = "admin_test:"


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


def _get_new_game_storage(context: ContextTypes.DEFAULT_TYPE, chat: Chat | None) -> dict:
    if chat and chat.type in GROUP_CHAT_TYPES:
        data = getattr(context, "chat_data", None)
        if not isinstance(data, dict):
            data = {}
            setattr(context, "chat_data", data)
        return data
    data = getattr(context, "user_data", None)
    if not isinstance(data, dict):
        data = {}
        setattr(context, "user_data", data)
    return data


def _get_pending_language(context: ContextTypes.DEFAULT_TYPE, chat: Chat | None) -> str | None:
    storage = _get_new_game_storage(context, chat)
    value = storage.get("new_game_language")
    if value is None:
        return None
    return str(value)


def _set_pending_language(
    context: ContextTypes.DEFAULT_TYPE, chat: Chat | None, language: str | None
) -> None:
    storage = _get_new_game_storage(context, chat)
    if language is None:
        storage["new_game_language"] = None
    else:
        storage["new_game_language"] = str(language)


def _clear_pending_language(context: ContextTypes.DEFAULT_TYPE, chat: Chat | None) -> None:
    storage = _get_new_game_storage(context, chat)
    storage.pop("new_game_language", None)


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

NEW_GAME_MENU_CALLBACK_PREFIX = "new_game_mode:"
NEW_GAME_MODE_SOLO = f"{NEW_GAME_MENU_CALLBACK_PREFIX}solo"
NEW_GAME_MODE_GROUP = f"{NEW_GAME_MENU_CALLBACK_PREFIX}group"

BUTTON_NEW_GAME_KEY = "button_new_game_flow"
BUTTON_STEP_KEY = "step"
BUTTON_LANGUAGE_KEY = "language"
BUTTON_STEP_LANGUAGE = "language"
BUTTON_STEP_THEME = "theme"

GENERATION_NOTICE_KEY = "puzzle_generation_notice"

ADMIN_COMMAND_PATTERN = re.compile(r"(?i)^\s*adm key")
ADMIN_KEYS_ONLY_PATTERN = re.compile(r"(?i)^\s*adm keys\s*$")
ADMIN_SINGLE_KEY_PATTERN = re.compile(r"(?i)^\s*adm key\s+(.+)$")

GROUP_CHAT_TYPES = {ChatType.GROUP, ChatType.SUPERGROUP}

LOBBY_INVITE_CALLBACK_PREFIX = "lobby_invite:"
LOBBY_LINK_CALLBACK_PREFIX = "lobby_link:"
LOBBY_START_CALLBACK_PREFIX = "lobby_start:"
LOBBY_WAIT_CALLBACK_PREFIX = "lobby_wait:"

MAX_LOBBY_PLAYERS = 6

JOIN_CODE_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
JOIN_CODE_LENGTH = 6


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


def _register_player_chat(user_id: int, chat_id: int | None) -> None:
    if chat_id is None:
        return
    state.player_chats[user_id] = chat_id


def _lookup_player_chat(user_id: int) -> int | None:
    return state.player_chats.get(user_id)


def _remember_job(job: Job | None) -> None:
    if job is None:
        return
    state.scheduled_jobs[job.name] = job


def _cancel_job(job_name: str | None) -> None:
    if not job_name:
        return
    job = state.scheduled_jobs.pop(job_name, None)
    if job is not None:
        job.schedule_removal()


def _current_player_id(game_state: GameState) -> int | None:
    if not game_state.turn_order:
        return None
    if not game_state.players:
        return None
    if game_state.turn_index >= len(game_state.turn_order):
        game_state.turn_index = 0
    return game_state.turn_order[game_state.turn_index]


def _current_player(game_state: GameState) -> Player | None:
    player_id = _current_player_id(game_state)
    if player_id is None:
        return None
    return game_state.players.get(player_id)


def _resolve_player_from_chat(
    game_state: GameState, chat: Chat | None, message: Message | None
) -> int | None:
    if message and getattr(message, "from_user", None):
        user = message.from_user  # type: ignore[assignment]
        if user and user.id in game_state.players:
            return user.id
    if chat and chat.type == ChatType.PRIVATE:
        for player in game_state.players.values():
            if player.dm_chat_id == chat.id:
                return player.user_id
    return None


def _count_hints_for_player(game_state: GameState, player_id: int) -> int:
    total = 0
    for usage in game_state.hints_used.values():
        total += usage.get(player_id, 0)
    return total


def _cancel_turn_timers(game_state: GameState) -> None:
    _cancel_job(game_state.turn_timer_job_id)
    _cancel_job(game_state.turn_warn_job_id)
    game_state.turn_timer_job_id = None
    game_state.turn_warn_job_id = None


def _cancel_game_timers(game_state: GameState) -> None:
    _cancel_job(game_state.game_timer_job_id)
    _cancel_job(game_state.game_warn_job_id)
    game_state.game_timer_job_id = None
    game_state.game_warn_job_id = None


def _cancel_dummy_job(game_state: GameState) -> None:
    _cancel_job(game_state.dummy_job_id)
    game_state.dummy_job_id = None
    game_state.dummy_turn_started_at = None
    game_state.dummy_planned_delay = 0.0


def _schedule_game_timers(
    context: ContextTypes.DEFAULT_TYPE, game_state: GameState
) -> None:
    if not context.job_queue:
        return
    _cancel_game_timers(game_state)
    data = {"game_id": game_state.game_id}
    if GAME_TIME_LIMIT_SECONDS > GAME_WARNING_SECONDS > 0:
        warn_name = f"game-warn-{game_state.game_id}"
        warn_job = context.job_queue.run_once(
            _game_warning_job,
            GAME_TIME_LIMIT_SECONDS - GAME_WARNING_SECONDS,
            chat_id=game_state.chat_id,
            name=warn_name,
            data=data,
        )
        _remember_job(warn_job)
        game_state.game_warn_job_id = warn_name
    timeout_name = f"game-timeout-{game_state.game_id}"
    timeout_job = context.job_queue.run_once(
        _game_timeout_job,
        GAME_TIME_LIMIT_SECONDS,
        chat_id=game_state.chat_id,
        name=timeout_name,
        data=data,
    )
    _remember_job(timeout_job)
    game_state.game_timer_job_id = timeout_name


def _schedule_turn_timers(
    context: ContextTypes.DEFAULT_TYPE, game_state: GameState
) -> None:
    if not context.job_queue:
        return
    _cancel_turn_timers(game_state)
    current_player = _current_player(game_state)
    if current_player is None:
        return
    data = {"game_id": game_state.game_id, "player_id": current_player.user_id}
    if TURN_TIME_LIMIT_SECONDS > TURN_WARNING_SECONDS > 0:
        warn_name = f"turn-warn-{game_state.game_id}"
        warn_job = context.job_queue.run_once(
            _turn_warning_job,
            TURN_TIME_LIMIT_SECONDS - TURN_WARNING_SECONDS,
            chat_id=game_state.chat_id,
            name=warn_name,
            data=data,
        )
        _remember_job(warn_job)
        game_state.turn_warn_job_id = warn_name
    timeout_name = f"turn-timeout-{game_state.game_id}"
    timeout_job = context.job_queue.run_once(
        _turn_timeout_job,
        TURN_TIME_LIMIT_SECONDS,
        chat_id=game_state.chat_id,
        name=timeout_name,
        data=data,
    )
    _remember_job(timeout_job)
    game_state.turn_timer_job_id = timeout_name


def _schedule_dummy_turn(
    context: ContextTypes.DEFAULT_TYPE,
    game_state: GameState,
    puzzle: Puzzle | CompositePuzzle,
) -> None:
    if not (game_state.test_mode and context.job_queue):
        _cancel_dummy_job(game_state)
        return
    player = _current_player(game_state)
    if (
        player is None
        or player.user_id != game_state.dummy_user_id
        or not player.is_bot
        or not _iter_available_slots(puzzle, game_state)
    ):
        _cancel_dummy_job(game_state)
        return
    _cancel_dummy_job(game_state)
    delay = random.uniform(*DUMMY_DELAY_RANGE)
    job_name = f"dummy-turn-{game_state.game_id}"
    data = {"game_id": game_state.game_id, "planned_delay": delay}
    job = context.job_queue.run_once(
        _dummy_turn_job,
        delay,
        chat_id=game_state.chat_id,
        name=job_name,
        data=data,
    )
    _remember_job(job)
    game_state.dummy_job_id = job_name
    game_state.dummy_turn_started_at = time.time()
    game_state.dummy_planned_delay = delay
    logger.debug(
        "Scheduled dummy response in %.2f seconds for game %s", delay, game_state.game_id
    )


def _alphabet_for_language(language: str | None) -> str:
    if not language:
        return string.ascii_uppercase
    normalised = language.lower()
    if normalised == "ru":
        return "АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ"
    if normalised == "it":
        return "AÁÀBCDEÈÉFGHIÌÍJKLMNOÒÓPQRSTUÙÚVWXYZ"
    if normalised == "es":
        return "AÁBCDEÉFGHIÍJKLMNÑOÓPQRSTUÚÜVWXYZ"
    return string.ascii_uppercase


def _generate_dummy_incorrect_answer(
    slot_ref: SlotRef, language: str | None
) -> str:
    answer = slot_ref.slot.answer or ""
    if not answer:
        alphabet = _alphabet_for_language(language)
        return "".join(random.choice(alphabet) for _ in range(slot_ref.slot.length))
    letters = list(answer)
    index = random.randrange(len(letters))
    alphabet = _alphabet_for_language(language)
    current_letter = letters[index]
    candidates = [
        ch for ch in alphabet if ch.upper() != current_letter.upper()
    ]
    if not candidates:
        candidates = list(alphabet) or [current_letter or "X"]
    replacement = random.choice(candidates)
    if current_letter.islower():
        replacement = replacement.lower()
    letters[index] = replacement
    candidate_word = "".join(letters)
    if candidate_word.upper() == (answer or "").upper():
        letters[index] = replacement.lower() if replacement.isupper() else replacement.upper()
        candidate_word = "".join(letters)
    return candidate_word


def _select_dummy_slot(
    game_state: GameState, puzzle: Puzzle | CompositePuzzle
) -> SlotRef | None:
    candidates: list[tuple[int, int, int, str, SlotRef]] = []
    for ref in _iter_available_slots(puzzle, game_state):
        revealed = 0
        for row, col in ref.slot.coordinates():
            key = _coord_key(row, col, ref.component_index)
            if key in game_state.filled_cells:
                revealed += 1
        candidates.append(
            (
                revealed,
                ref.slot.length,
                ref.slot.number,
                f"{ref.slot.direction.value}:{ref.public_id}",
                ref,
            )
        )
    if not candidates:
        return None
    min_revealed = min(entry[0] for entry in candidates)
    filtered = [entry for entry in candidates if entry[0] == min_revealed]
    if game_state.dummy_failures > game_state.dummy_successes:
        filtered.sort(key=lambda entry: (entry[1], entry[2], entry[3]))
    else:
        filtered.sort(key=lambda entry: (entry[0], entry[1], entry[2], entry[3]))
    return filtered[0][4]


async def _dummy_turn_job(context: CallbackContext) -> None:
    job = context.job
    if job is None:
        return
    state.scheduled_jobs.pop(job.name, None)
    data = job.data or {}
    game_id = data.get("game_id")
    planned_delay = float(data.get("planned_delay", 0.0) or 0.0)
    if not game_id:
        return
    game_state = _load_state_by_game_id(game_id)
    if (
        not game_state
        or game_state.status != "running"
        or not game_state.test_mode
        or game_state.dummy_user_id is None
    ):
        return
    if game_state.dummy_job_id == job.name:
        game_state.dummy_job_id = None
    puzzle = _load_puzzle_for_state(game_state)
    if puzzle is None:
        return
    current_player = _current_player(game_state)
    if (
        current_player is None
        or current_player.user_id != game_state.dummy_user_id
        or not current_player.is_bot
    ):
        return
    slot_ref = _select_dummy_slot(game_state, puzzle)
    if slot_ref is None:
        await _finish_game(
            context,
            game_state,
            reason="Все слова разгаданы. Тест завершён.",
        )
        return
    normalised_slot = _normalise_slot_id(slot_ref.public_id)
    actual_delay = 0.0
    if game_state.dummy_turn_started_at is not None:
        actual_delay = max(0.0, time.time() - game_state.dummy_turn_started_at)
    else:
        actual_delay = max(planned_delay, 0.0)
    attempt_success = random.random() <= DUMMY_ACCURACY
    attempt_answer = (
        slot_ref.slot.answer
        if attempt_success
        else _generate_dummy_incorrect_answer(slot_ref, puzzle.language)
    )
    dummy_player = game_state.players.get(game_state.dummy_user_id)
    game_state.dummy_turns += 1
    game_state.dummy_total_delay += actual_delay
    game_state.dummy_turn_started_at = None
    game_state.dummy_planned_delay = 0.0
    info_prefix = (
        f"🤖 {DUMMY_NAME}"
        if dummy_player is None or not dummy_player.name
        else f"🤖 {dummy_player.name}"
    )
    message_text = f"{info_prefix}: /answer {slot_ref.public_id} {attempt_answer}"
    try:
        await context.bot.send_message(chat_id=game_state.chat_id, text=message_text)
    except Exception:  # noqa: BLE001
        logger.exception(
            "Failed to announce dummy answer attempt in game %s", game_state.game_id
        )
    log_result = "success" if attempt_success else "fail"
    points = slot_ref.slot.length if attempt_success else 0
    with logging_context(chat_id=game_state.chat_id, puzzle_id=game_state.puzzle_id):
        logger.info(
            "Dummy turn: slot=%s delay=%.2fs result=%s points=%s",
            normalised_slot,
            actual_delay,
            log_result,
            points,
        )
    game_state.last_update = time.time()
    if attempt_success:
        game_state.dummy_successes += 1
        game_state.score += slot_ref.slot.length
        _record_score(game_state, slot_ref.slot.length, user_id=game_state.dummy_user_id)
        if dummy_player:
            dummy_player.answers_ok += 1
        _cancel_turn_timers(game_state)
        game_state.active_slot_id = normalised_slot
        _apply_answer_to_state(game_state, slot_ref, attempt_answer)
        game_state.active_slot_id = None
        _store_state(game_state)
        success_text = (
            f"{info_prefix} разгадал {slot_ref.public_id}! (+{slot_ref.slot.length} очков)"
        )
        try:
            await context.bot.send_message(
                chat_id=game_state.chat_id, text=success_text
            )
        except Exception:  # noqa: BLE001
            logger.exception(
                "Failed to notify about dummy success in game %s", game_state.game_id
            )
        if _all_slots_solved(puzzle, game_state):
            await _finish_game(
                context,
                game_state,
                reason="Все слова разгаданы. Тест завершён.",
            )
            return
        _advance_turn(game_state)
        _store_state(game_state)
        await _announce_turn(context, game_state, puzzle)
        return

    # Failure branch
    game_state.dummy_failures += 1
    if dummy_player:
        dummy_player.answers_fail += 1
    _cancel_turn_timers(game_state)
    failure_text = f"{info_prefix} ошибся на {slot_ref.public_id}."
    try:
        await context.bot.send_message(chat_id=game_state.chat_id, text=failure_text)
    except Exception:  # noqa: BLE001
        logger.exception(
            "Failed to notify about dummy failure in game %s", game_state.game_id
        )
    _advance_turn(game_state)
    _store_state(game_state)
    await _announce_turn(context, game_state, puzzle)
def _advance_turn(game_state: GameState) -> int | None:
    if not game_state.turn_order:
        return None
    if game_state.turn_index >= len(game_state.turn_order):
        game_state.turn_index = 0
    game_state.turn_index = (game_state.turn_index + 1) % len(game_state.turn_order)
    game_state.active_slot_id = None
    game_state.last_update = time.time()
    return _current_player_id(game_state)


def _build_turn_keyboard(game_state: GameState) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton(
                    "Выбрать подсказку",
                    callback_data=f"{TURN_SELECT_CALLBACK_PREFIX}{game_state.game_id}",
                )
            ]
        ]
    )


def _iter_available_slots(
    puzzle: Puzzle | CompositePuzzle, game_state: GameState
) -> list[SlotRef]:
    solved = {_normalise_slot_id(entry) for entry in game_state.solved_slots}
    slots: list[SlotRef] = []
    for ref in _sorted_slot_refs(puzzle):
        if not ref.slot.answer:
            continue
        if _normalise_slot_id(ref.public_id) in solved:
            continue
        slots.append(ref)
    return slots


def _build_slot_keyboard(
    game_state: GameState, puzzle: Puzzle | CompositePuzzle
) -> InlineKeyboardMarkup:
    buttons: list[list[InlineKeyboardButton]] = []
    row: list[InlineKeyboardButton] = []
    for ref in _iter_available_slots(puzzle, game_state):
        callback = f"{TURN_SLOT_CALLBACK_PREFIX}{game_state.game_id}|{_normalise_slot_id(ref.public_id)}"
        label = ref.public_id
        row.append(InlineKeyboardButton(label, callback_data=callback))
        if len(row) == 3:
            buttons.append(row)
            row = []
    if row:
        buttons.append(row)
    if not buttons:
        buttons = [[InlineKeyboardButton("Нет доступных слотов", callback_data="noop")]]
    return InlineKeyboardMarkup(buttons)


def _find_slot_by_identifier(
    puzzle: Puzzle | CompositePuzzle, identifier: str
) -> SlotRef | None:
    normalised = _normalise_slot_id(identifier)
    for ref in iter_slot_refs(puzzle):
        if _normalise_slot_id(ref.public_id) == normalised:
            return ref
    return None


async def _announce_turn(
    context: ContextTypes.DEFAULT_TYPE,
    game_state: GameState,
    puzzle: Puzzle | CompositePuzzle,
    *,
    prefix: str | None = None,
) -> None:
    player = _current_player(game_state)
    if player is None:
        return
    parts = []
    if prefix:
        parts.append(prefix)
    parts.append(f"Ход игрока {player.name}. Выберите подсказку.")
    text = "\n".join(parts)
    keyboard = _build_turn_keyboard(game_state)
    try:
        await context.bot.send_message(
            chat_id=game_state.chat_id,
            text=text,
            reply_markup=keyboard,
        )
    except Exception:  # noqa: BLE001
        logger.exception("Failed to announce turn in group for game %s", game_state.game_id)
    if player.dm_chat_id:
        try:
            await context.bot.send_message(
                chat_id=player.dm_chat_id,
                text=text,
                reply_markup=keyboard,
            )
        except Exception:  # noqa: BLE001
            logger.exception(
                "Failed to send DM announcement to player %s", player.user_id
            )
    _schedule_turn_timers(context, game_state)
    if game_state.test_mode:
        _schedule_dummy_turn(context, game_state, puzzle)
    game_state.last_update = time.time()
    _store_state(game_state)


async def _handle_turn_timeout(
    context: ContextTypes.DEFAULT_TYPE, game_state: GameState
) -> None:
    player = _current_player(game_state)
    puzzle = _load_puzzle_for_state(game_state)
    if puzzle is None:
        return
    dummy_timeout = (
        game_state.test_mode
        and player is not None
        and player.user_id == game_state.dummy_user_id
    )
    if dummy_timeout:
        elapsed = 0.0
        if game_state.dummy_turn_started_at is not None:
            elapsed = max(0.0, time.time() - game_state.dummy_turn_started_at)
        elif game_state.dummy_planned_delay:
            elapsed = max(0.0, game_state.dummy_planned_delay)
        game_state.dummy_turns += 1
        game_state.dummy_failures += 1
        game_state.dummy_total_delay += elapsed
        game_state.dummy_turn_started_at = None
        game_state.dummy_planned_delay = 0.0
        if player:
            player.answers_fail += 1
        with logging_context(chat_id=game_state.chat_id, puzzle_id=game_state.puzzle_id):
            logger.info(
                "Dummy timeout: slot=%s delay=%.2fs",
                game_state.active_slot_id or "-",
                elapsed,
            )
        _cancel_dummy_job(game_state)
    elif player:
        player.answers_fail += 1
    _cancel_turn_timers(game_state)
    message = "Ход пропущен по таймеру."
    if player:
        message = f"{player.name} не успел ответить. Ход переходит дальше."
    try:
        await context.bot.send_message(chat_id=game_state.chat_id, text=message)
    except Exception:  # noqa: BLE001
        logger.exception("Failed to notify about turn timeout for game %s", game_state.game_id)
    _advance_turn(game_state)
    _store_state(game_state)
    await _announce_turn(context, game_state, puzzle)


def _format_leaderboard(game_state: GameState) -> str:
    entries: list[tuple[int, int, int, str]] = []
    for player_id, player in game_state.players.items():
        score = game_state.scoreboard.get(player_id, 0)
        solved = player.answers_ok
        hints = _count_hints_for_player(game_state, player_id)
        entries.append((score, solved, hints, player.name))
    if not entries:
        return "Нет результатов."
    entries.sort(key=lambda item: (-item[0], -item[1], item[2], item[3].lower()))
    lines = []
    for index, (score, solved, hints, name) in enumerate(entries, start=1):
        lines.append(
            f"{index}. {name} — {score} очков, решено: {solved}, подсказки: {hints}"
        )
    return "\n".join(lines)


async def _finish_game(
    context: ContextTypes.DEFAULT_TYPE,
    game_state: GameState,
    *,
    reason: str | None = None,
) -> None:
    if game_state.status == "finished":
        return
    puzzle = _load_puzzle_for_state(game_state)
    if puzzle is None:
        return
    _cancel_turn_timers(game_state)
    _cancel_game_timers(game_state)
    _cancel_dummy_job(game_state)
    game_state.status = "finished"
    game_state.active_slot_id = None
    game_state.last_update = time.time()
    summary = _format_leaderboard(game_state)
    lines = ["Игра завершена!"]
    if reason:
        lines.append(reason)
    lines.append("")
    lines.append("Итоги:")
    lines.append(summary)
    dummy_summary: str | None = None
    if game_state.test_mode:
        turns = game_state.dummy_turns
        successes = game_state.dummy_successes
        failures = game_state.dummy_failures
        accuracy = (successes / turns * 100) if turns else 0.0
        average_delay = (game_state.dummy_total_delay / turns) if turns else 0.0
        dummy_summary = (
            "🤖 Статистика дамми — "
            f"ходов: {turns}, верных: {successes}, ошибок: {failures}, "
            f"точность: {accuracy:.0f}%, средняя задержка: {average_delay:.1f} с."
        )
        with logging_context(chat_id=game_state.chat_id, puzzle_id=game_state.puzzle_id):
            logger.info(
                "Dummy summary: turns=%s successes=%s failures=%s accuracy=%.1f%% avg_delay=%.2fs",
                turns,
                successes,
                failures,
                accuracy,
                average_delay,
            )
    if dummy_summary:
        lines.append(dummy_summary)
    keyboard = InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton(
                    "Реванш",
                    callback_data=f"{SAME_TOPIC_CALLBACK_PREFIX}{game_state.puzzle_id}",
                )
            ],
            [
                InlineKeyboardButton(
                    "Новая игра",
                    callback_data=f"{NEW_PUZZLE_CALLBACK_PREFIX}{game_state.puzzle_id}",
                )
            ],
        ]
    )
    try:
        await context.bot.send_message(
            chat_id=game_state.chat_id,
            text="\n".join(lines),
            reply_markup=keyboard,
        )
    except Exception:  # noqa: BLE001
        logger.exception("Failed to send finish summary for game %s", game_state.game_id)
    _store_state(game_state)


def _user_display_name(user: User | None) -> str:
    if user is None:
        return "Игрок"
    if user.full_name:
        return user.full_name
    if user.username:
        return f"@{user.username}"
    return str(user.id)


def _player_display_name(player: Player) -> str:
    if player.name:
        return player.name
    return str(player.user_id)


def _assign_join_code(game_state: GameState) -> str:
    for _ in range(64):
        code = "".join(secrets.choice(JOIN_CODE_ALPHABET) for _ in range(JOIN_CODE_LENGTH))
        if code not in state.join_codes and code not in game_state.join_codes:
            game_state.join_codes[code] = game_state.game_id
            return code
    raise RuntimeError("Не удалось сгенерировать код присоединения")


def _build_lobby_keyboard(game_state: GameState) -> InlineKeyboardMarkup:
    invite_button = InlineKeyboardButton(
        text="Пригласить из контактов",
        callback_data=f"{LOBBY_INVITE_CALLBACK_PREFIX}{game_state.game_id}",
    )
    link_button = InlineKeyboardButton(
        text="Создать ссылку",
        callback_data=f"{LOBBY_LINK_CALLBACK_PREFIX}{game_state.game_id}",
    )
    has_min_players = len(game_state.players) >= 2
    if has_min_players:
        start_callback = f"{LOBBY_START_CALLBACK_PREFIX}{game_state.game_id}"
    else:
        start_callback = f"{LOBBY_WAIT_CALLBACK_PREFIX}{game_state.game_id}"
    start_button = InlineKeyboardButton(text="Старт", callback_data=start_callback)
    rows = [[invite_button, link_button], [start_button]]
    settings = state.settings
    if (
        settings
        and settings.admin_id is not None
        and game_state.host_id == settings.admin_id
    ):
        rows.append(
            [
                InlineKeyboardButton(
                    "[адм.] Тестовая игра 1×1",
                    callback_data=f"{ADMIN_TEST_GAME_CALLBACK_PREFIX}{game_state.chat_id}",
                )
            ]
        )
    return InlineKeyboardMarkup(rows)


def _format_lobby_text(game_state: GameState) -> str:
    language = (game_state.language or "?").upper()
    theme = game_state.theme or "(тема не выбрана)"
    players = ", ".join(
        _player_display_name(player) for player in game_state.players.values()
    )
    generation_task = state.lobby_generation_tasks.get(game_state.game_id or "")
    generating = bool(generation_task and not generation_task.done())
    if generating:
        puzzle_status = "Статус пазла: генерируется…"
    elif game_state.puzzle_id:
        puzzle_status = "Статус пазла: готов к старту ✅"
    else:
        puzzle_status = "Статус пазла: ожидает подготовки"
    return (
        "Комната готова!\n"
        f"Язык: {language}\n"
        f"Тема: {theme}\n"
        f"{puzzle_status}\n"
        f"Игроки ({len(game_state.players)}/{MAX_LOBBY_PLAYERS}): {players or 'ещё нет участников'}"
    )


async def _publish_lobby_message(
    context: ContextTypes.DEFAULT_TYPE,
    game_state: GameState,
    *,
    message: Message | None = None,
) -> None:
    chat_id = game_state.chat_id
    keyboard = _build_lobby_keyboard(game_state)
    text = _format_lobby_text(game_state)
    sent = await context.bot.send_message(
        chat_id=chat_id,
        text=text,
        reply_markup=keyboard,
    )
    state.lobby_messages[game_state.game_id] = (chat_id, sent.message_id)


async def _update_lobby_message(
    context: ContextTypes.DEFAULT_TYPE, game_state: GameState
) -> None:
    entry = state.lobby_messages.get(game_state.game_id)
    if not entry:
        return
    chat_id, message_id = entry
    keyboard = _build_lobby_keyboard(game_state)
    text = _format_lobby_text(game_state)
    try:
        await context.bot.edit_message_text(
            chat_id=chat_id,
            message_id=message_id,
            text=text,
            reply_markup=keyboard,
        )
    except TelegramError:
        logger.exception("Failed to update lobby message for game %s", game_state.game_id)


async def _run_lobby_puzzle_generation(
    context: ContextTypes.DEFAULT_TYPE,
    game_id: str,
    language: str,
    theme: str,
) -> None:
    base_state = _load_state_by_game_id(game_id)
    if base_state is None:
        logger.warning("Lobby generation requested for unknown game %s", game_id)
        state.lobby_generation_tasks.pop(game_id, None)
        return
    chat_id = base_state.chat_id
    loop = asyncio.get_running_loop()
    state.generating_chats.add(chat_id)
    puzzle: Puzzle | CompositePuzzle | None = None
    generated_state: GameState | None = None
    try:
        puzzle, generated_state = await loop.run_in_executor(
            None, _generate_puzzle, chat_id, language, theme
        )
    except asyncio.CancelledError:
        logger.info("Lobby puzzle generation cancelled for game %s", game_id)
        state.lobby_generation_tasks.pop(game_id, None)
        raise
    except Exception:
        logger.exception("Failed to generate puzzle for lobby %s", game_id)
        refreshed = _load_state_by_game_id(game_id)
        if refreshed and refreshed.status == "lobby":
            refreshed.puzzle_id = ""
            refreshed.puzzle_ids = None
            refreshed.hinted_cells = set()
            refreshed.last_update = time.time()
            _store_state(refreshed)
            try:
                if game_id in state.lobby_messages:
                    await _update_lobby_message(context, refreshed)
                else:
                    await _publish_lobby_message(context, refreshed)
            except TelegramError:
                logger.exception(
                    "Failed to update lobby message after generation error for %s",
                    game_id,
                )
        try:
            await context.bot.send_message(
                chat_id=chat_id,
                text=(
                    "Не удалось подготовить кроссворд. Попробуйте выбрать тему ещё раз."
                ),
            )
        except TelegramError:
            logger.exception("Failed to notify chat %s about generation failure", chat_id)
        state.lobby_generation_tasks.pop(game_id, None)
        return
    finally:
        state.generating_chats.discard(chat_id)

    refreshed = _load_state_by_game_id(game_id)
    if refreshed is None:
        logger.warning("Game state missing after generation for lobby %s", game_id)
        state.lobby_generation_tasks.pop(game_id, None)
        return
    if refreshed.status != "lobby":
        logger.info(
            "Skipping lobby update for game %s because status is %s",
            game_id,
            refreshed.status,
        )
        state.lobby_generation_tasks.pop(game_id, None)
        return
    if generated_state is None or puzzle is None:
        state.lobby_generation_tasks.pop(game_id, None)
        return
    refreshed.puzzle_id = generated_state.puzzle_id
    refreshed.puzzle_ids = generated_state.puzzle_ids
    refreshed.filled_cells.clear()
    refreshed.solved_slots.clear()
    refreshed.hinted_cells = set()
    refreshed.score = 0
    refreshed.scoreboard.clear()
    refreshed.turn_order.clear()
    refreshed.turn_index = 0
    refreshed.active_slot_id = None
    refreshed.language = language
    refreshed.theme = theme
    refreshed.last_update = time.time()
    refreshed.status = "lobby"
    _store_state(refreshed)
    try:
        if game_id in state.lobby_messages:
            await _update_lobby_message(context, refreshed)
        else:
            await _publish_lobby_message(context, refreshed)
    except TelegramError:
        logger.exception("Failed to publish lobby update for game %s", game_id)
    else:
        try:
            await context.bot.send_message(
                chat_id=chat_id,
                text="Кроссворд готов! Нажмите «Старт», чтобы начать игру.",
            )
        except TelegramError:
            logger.exception(
                "Failed to notify chat %s about puzzle readiness", chat_id
            )
    state.lobby_generation_tasks.pop(game_id, None)


def _load_state_by_game_id(game_id: str) -> GameState | None:
    if not game_id:
        return None
    if game_id in state.active_games:
        return state.active_games[game_id]
    restored = load_state(game_id)
    if restored is None:
        return None
    state.active_games[restored.game_id] = restored
    state.chat_to_game[restored.chat_id] = restored.game_id
    for code, target in list(state.join_codes.items()):
        if target == restored.game_id and code not in restored.join_codes:
            state.join_codes.pop(code, None)
    for code, target in restored.join_codes.items():
        state.join_codes[code] = target
    return restored


def _ensure_player_entry(
    game_state: GameState, user: User, name: str, dm_chat_id: int | None
) -> Player:
    existing = game_state.players.get(user.id)
    if existing:
        if name:
            existing.name = name
        if dm_chat_id is not None:
            existing.dm_chat_id = dm_chat_id
        game_state.scoreboard.setdefault(user.id, 0)
        return existing
    player = Player(user_id=user.id, name=name, dm_chat_id=dm_chat_id)
    game_state.players[user.id] = player
    game_state.scoreboard.setdefault(user.id, 0)
    return player


async def _build_join_link(context: ContextTypes.DEFAULT_TYPE, code: str) -> str | None:
    username = context.bot.username
    if not username:
        try:
            me = await context.bot.get_me()
        except TelegramError:
            logger.exception("Failed to resolve bot username for deep link")
            return None
        username = me.username or ""
    if not username:
        return None
    return f"https://t.me/{username}?start=join_{code}"


async def track_player_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat = update.effective_chat
    user = update.effective_user
    if chat and user and chat.type == ChatType.PRIVATE:
        _register_player_chat(user.id, chat.id)


async def track_player_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat = update.effective_chat
    user = update.effective_user
    if chat and user and chat.type == ChatType.PRIVATE:
        _register_player_chat(user.id, chat.id)


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
    if game_state.mode == "turn_based":
        extras: list[str] = []
        current_player = _current_player(game_state)
        if current_player:
            extras.append(f"Сейчас ход {html.escape(current_player.name)}.")
        if game_state.scoreboard:
            board_parts: list[str] = []
            for player_id, score in sorted(
                game_state.scoreboard.items(), key=lambda item: (-item[1], item[0])
            ):
                player = game_state.players.get(player_id)
                name = html.escape(player.name if player else str(player_id))
                board_parts.append(f"{name}: {score}")
            if board_parts:
                extras.append("Очки: " + ", ".join(board_parts))
        if extras:
            text = f"{text}\n\n" + "\n".join(extras)
    await message.reply_text(text, parse_mode=constants.ParseMode.HTML)


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

    chat_data = getattr(context, "chat_data", None)
    getter = getattr(chat_data, "get", None)
    existing_notice = getter(GENERATION_NOTICE_KEY) if callable(getter) else None
    reason = text
    if (
        isinstance(existing_notice, dict)
        and existing_notice.get("active")
        and existing_notice.get("reason") == reason
    ):
        logger.debug(
            "Skipping duplicate generation notice for chat %s", chat_id
        )
        return

    _cancel_generation_updates(context, chat_id)
    remover = getattr(chat_data, "pop", None)
    if callable(remover):
        remover(GENERATION_NOTICE_KEY, None)

    base_text = text or "Готовлю кроссворд, это может занять немного времени..."
    variations: list[str] = []
    for template in GENERATION_NOTICE_TEMPLATES:
        try:
            formatted = template.format(base=base_text)
        except Exception:  # noqa: BLE001
            formatted = base_text
        variations.append(formatted)
    if base_text not in variations:
        variations.append(base_text)
    unique_options = list(dict.fromkeys(variations))
    chosen_text = random.choice(unique_options) if unique_options else base_text

    notice_state = {
        "active": True,
        "text": chosen_text,
        "reason": reason,
        "started_at": time.monotonic(),
        "update_cycle": list(GENERATION_UPDATE_TEMPLATES),
        "update_index": 0,
    }
    random.shuffle(notice_state["update_cycle"])
    context.chat_data[GENERATION_NOTICE_KEY] = notice_state
    _schedule_generation_updates(context, chat_id)

    if message is not None:
        await message.reply_text(chosen_text)
    else:
        await context.bot.send_message(chat_id=chat_id, text=chosen_text)


def _clear_generation_notice(
    context: ContextTypes.DEFAULT_TYPE, chat_id: int | None
) -> None:
    """Clear generation notice tracking for the chat."""

    _cancel_generation_updates(context, chat_id)
    removed = context.chat_data.pop(GENERATION_NOTICE_KEY, None)
    if removed is not None and chat_id is not None:
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


def _schedule_generation_updates(
    context: ContextTypes.DEFAULT_TYPE, chat_id: int
) -> None:
    chat_data = getattr(context, "chat_data", None)
    get_chat_data = getattr(chat_data, "get", None)
    if not callable(get_chat_data):
        return
    notice = get_chat_data(GENERATION_NOTICE_KEY)
    if not isinstance(notice, dict):
        return
    updates = list(GENERATION_UPDATE_TEMPLATES)
    random.shuffle(updates)
    notice["update_cycle"] = updates
    notice["update_index"] = 0
    notice.pop("last_update_text", None)
    notice.pop("last_typing_action", None)
    job_queue = getattr(context, "job_queue", None)
    if not job_queue or not hasattr(job_queue, "run_repeating"):
        return
    update_job_name = f"generation-update-{chat_id}"
    _cancel_job(update_job_name)
    update_job = job_queue.run_repeating(
        _generation_update_job,
        interval=GENERATION_UPDATE_INTERVAL_SECONDS,
        first=GENERATION_UPDATE_FIRST_DELAY_SECONDS,
        chat_id=chat_id,
        name=update_job_name,
    )
    if update_job is not None:
        _remember_job(update_job)
        notice["update_job_id"] = update_job_name
    typing_job_name = f"generation-typing-{chat_id}"
    _cancel_job(typing_job_name)
    typing_job = job_queue.run_repeating(
        _generation_typing_job,
        interval=GENERATION_TYPING_INTERVAL_SECONDS,
        first=GENERATION_TYPING_INITIAL_DELAY_SECONDS,
        chat_id=chat_id,
        name=typing_job_name,
    )
    if typing_job is not None:
        _remember_job(typing_job)
        notice["typing_job_id"] = typing_job_name


def _cancel_generation_updates(
    context: ContextTypes.DEFAULT_TYPE, chat_id: int | None
) -> None:
    chat_data = getattr(context, "chat_data", None)
    get_chat_data = getattr(chat_data, "get", None)
    if not callable(get_chat_data):
        return
    notice = get_chat_data(GENERATION_NOTICE_KEY)
    if not isinstance(notice, dict):
        return
    update_job_id = notice.pop("update_job_id", None)
    typing_job_id = notice.pop("typing_job_id", None)
    _cancel_job(update_job_id)
    _cancel_job(typing_job_id)
    notice.pop("update_cycle", None)
    notice.pop("update_index", None)
    notice.pop("last_update_text", None)
    notice.pop("last_typing_action", None)


async def _generation_update_job(context: CallbackContext) -> None:
    job = context.job
    if job is None:
        return
    chat_id = job.chat_id
    chat_data = getattr(context, "chat_data", None)
    notice = None
    if chat_data is not None:
        getter = getattr(chat_data, "get", None)
        if callable(getter):
            notice = getter(GENERATION_NOTICE_KEY)
    if not isinstance(notice, dict) or not notice.get("active"):
        if isinstance(notice, dict):
            notice.pop("update_job_id", None)
        _cancel_job(job.name)
        return
    updates = notice.get("update_cycle") or []
    if not updates:
        updates = list(GENERATION_UPDATE_TEMPLATES)
        random.shuffle(updates)
    index = int(notice.get("update_index", 0))
    if index >= len(updates):
        random.shuffle(updates)
        index = 0
    message = updates[index]
    last_message = notice.get("last_update_text")
    if last_message == message and len(updates) > 1:
        index = (index + 1) % len(updates)
        message = updates[index]
    next_index = index + 1
    if next_index >= len(updates):
        random.shuffle(updates)
        if len(updates) > 1 and updates[0] == message:
            updates.append(updates.pop(0))
        next_index = 0
    notice["update_cycle"] = updates
    notice["update_index"] = next_index
    notice["last_update_text"] = message
    try:
        await context.bot.send_message(chat_id=chat_id, text=message)
    except Exception:  # noqa: BLE001
        logger.exception(
            "Failed to send generation status update to chat %s", chat_id
        )


async def _generation_typing_job(context: CallbackContext) -> None:
    job = context.job
    if job is None:
        return
    chat_id = job.chat_id
    chat_data = getattr(context, "chat_data", None)
    notice = None
    if chat_data is not None:
        getter = getattr(chat_data, "get", None)
        if callable(getter):
            notice = getter(GENERATION_NOTICE_KEY)
    if not isinstance(notice, dict) or not notice.get("active"):
        if isinstance(notice, dict):
            notice.pop("typing_job_id", None)
        _cancel_job(job.name)
        return
    actions = list(GENERATION_TYPING_ACTIONS)
    if not actions:
        return
    action = random.choice(actions)
    last_action = notice.get("last_typing_action")
    if len(actions) > 1 and action == last_action:
        alternatives = [item for item in actions if item != last_action]
        if alternatives:
            action = random.choice(alternatives)
    notice["last_typing_action"] = action
    try:
        await context.bot.send_chat_action(chat_id=chat_id, action=action)
    except Exception:  # noqa: BLE001
        logger.exception(
            "Failed to send generation typing action to chat %s", chat_id
        )


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
        game_state.active_slot_id = None
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
    state.scheduled_jobs.pop(job.name, None)
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


async def _game_warning_job(context: CallbackContext) -> None:
    job = context.job
    if job is None:
        return
    state.scheduled_jobs.pop(job.name, None)
    data = job.data or {}
    game_id = data.get("game_id")
    game_state = _load_state_by_game_id(game_id)
    if not game_state or game_state.status != "running":
        return
    text = "До завершения игры осталось одна минута!"
    try:
        await context.bot.send_message(chat_id=game_state.chat_id, text=text)
    except Exception:  # noqa: BLE001
        logger.exception("Failed to send game warning for %s", game_id)


async def _game_timeout_job(context: CallbackContext) -> None:
    job = context.job
    if job is None:
        return
    state.scheduled_jobs.pop(job.name, None)
    data = job.data or {}
    game_id = data.get("game_id")
    game_state = _load_state_by_game_id(game_id)
    if not game_state or game_state.status != "running":
        return
    await _finish_game(context, game_state, reason="Время игры истекло.")


async def _turn_warning_job(context: CallbackContext) -> None:
    job = context.job
    if job is None:
        return
    state.scheduled_jobs.pop(job.name, None)
    data = job.data or {}
    game_id = data.get("game_id")
    player_id = data.get("player_id")
    if not game_id or player_id is None:
        return
    game_state = _load_state_by_game_id(game_id)
    if not game_state or game_state.status != "running":
        return
    if _current_player_id(game_state) != player_id:
        return
    player = game_state.players.get(player_id)
    if not player:
        return
    warning = f"{player.name}, осталось {TURN_WARNING_SECONDS} секунд на ход!"
    try:
        await context.bot.send_message(chat_id=game_state.chat_id, text=warning)
        if player.dm_chat_id:
            await context.bot.send_message(chat_id=player.dm_chat_id, text=warning)
    except Exception:  # noqa: BLE001
        logger.exception("Failed to send turn warning for game %s", game_id)


async def _turn_timeout_job(context: CallbackContext) -> None:
    job = context.job
    if job is None:
        return
    state.scheduled_jobs.pop(job.name, None)
    data = job.data or {}
    game_id = data.get("game_id")
    player_id = data.get("player_id")
    if not game_id or player_id is None:
        return
    game_state = _load_state_by_game_id(game_id)
    if not game_state or game_state.status != "running":
        return
    if _current_player_id(game_state) != player_id:
        return
    await _handle_turn_timeout(context, game_state)


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
            component_indices.sort()
            components.append([clues[i] for i in component_indices])
    components.sort(key=len, reverse=True)
    return components


def _select_connected_clue_set(
    components: Sequence[Sequence[WordClue]], language: str, size: int
) -> list[WordClue] | None:
    """Return a connected subset of clues with the requested size if available."""

    if size <= 0:
        return []

    def _extract_from_component(component: Sequence[WordClue]) -> list[WordClue] | None:
        if len(component) < size:
            return None
        letter_sets = [
            _canonical_letter_set(clue.word, language) for clue in component
        ]
        indices = list(range(len(component)))
        for start_pos, start_idx in enumerate(indices):
            selected = [start_idx]
            remaining = indices[:start_pos] + indices[start_pos + 1 :]
            current_letters = set(letter_sets[start_idx])
            while len(selected) < size and remaining:
                found = False
                for pos, idx in enumerate(remaining):
                    if letter_sets[idx] & current_letters:
                        selected.append(idx)
                        current_letters.update(letter_sets[idx])
                        remaining.pop(pos)
                        found = True
                        break
                if not found:
                    break
            if len(selected) == size:
                return [component[i] for i in sorted(selected)]
        return None

    for component in components:
        subset = _extract_from_component(component)
        if subset is not None:
            return subset
    return None


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


def _clone_puzzle_for_test(
    puzzle: Puzzle | CompositePuzzle,
) -> tuple[Puzzle | CompositePuzzle, str, list[str] | None]:
    suffix = uuid4().hex[:8]
    if isinstance(puzzle, CompositePuzzle):
        payload = composite_to_dict(puzzle)
        new_id = f"{puzzle.id}-adm-{suffix}"
        payload["id"] = new_id
        component_ids: list[str] = []
        for index, component_payload in enumerate(payload.get("components", []), start=1):
            puzzle_payload = component_payload.get("puzzle")
            if isinstance(puzzle_payload, dict):
                new_component_id = f"{new_id}-c{index}"
                puzzle_payload["id"] = new_component_id
                component_ids.append(new_component_id)
        save_puzzle(new_id, payload)
        cloned = puzzle_from_dict(dict(payload))
        return cloned, new_id, component_ids or None
    payload = puzzle_to_dict(puzzle)  # type: ignore[arg-type]
    new_id = f"{puzzle.id}-adm-{suffix}"
    payload["id"] = new_id
    save_puzzle(new_id, payload)
    cloned = puzzle_from_dict(dict(payload))
    return cloned, new_id, None


def _generate_puzzle(
    chat_id: int, language: str, theme: str
) -> tuple[Puzzle | CompositePuzzle, GameState]:
    with logging_context(chat_id=chat_id):
        logger.info(
            "Starting puzzle generation (language=%s, theme=%s)",
            language,
            theme,
        )
        attempted_component_split = False
        replacement_prompt_words: set[str] = set()
        used_canonical_words: set[str] = set()

        @dataclass(slots=True)
        class RejectionInfo:
            """Metadata describing why a candidate was skipped in an attempt."""

            last_attempt: int
            reasons: set[str]

        rejected_canonical_words: dict[str, RejectionInfo] = {}
        replacement_requests = 0
        current_attempt_index = 0
        replacement_failure_streak = 0
        clues: Sequence[WordClue] | None = None
        validated_clues: list[WordClue] = []
        word_components: list[list[WordClue]] = []
        fallback_component: list[WordClue] | None = None
        max_attempt_words = 0
        min_attempt_words = 0

        def _refresh_clues(*, initial: bool) -> None:
            """Pull a fresh list of clues and reset tracking structures."""

            nonlocal clues
            nonlocal validated_clues
            nonlocal word_components
            nonlocal fallback_component
            nonlocal max_attempt_words
            nonlocal min_attempt_words
            nonlocal attempted_component_split
            nonlocal replacement_requests
            nonlocal current_attempt_index
            nonlocal replacement_failure_streak

            if not initial:
                logger.info(
                    "Regenerating clue list after exhausting replacement attempts"
                )

            replacement_prompt_words.clear()
            used_canonical_words.clear()
            rejected_canonical_words.clear()
            replacement_requests = 0
            current_attempt_index = 0
            attempted_component_split = False
            replacement_failure_streak = 0

            clues = generate_clues(theme=theme, language=language)
            logger.info("Received %s raw clues from LLM", len(clues))
            validated_clues = validate_word_list(language, clues, deduplicate=True)
            logger.info("Validated %s clues for placement", len(validated_clues))
            if not validated_clues:
                raise RuntimeError("Не удалось подобрать ни одного подходящего слова")

            word_components = _build_word_components(validated_clues, language)
            logger.debug(
                "Identified %s connected word components after validation",
                len(word_components),
            )

            max_attempt_words = min(len(validated_clues), 80)
            # Allow attempts that use roughly half of the validated pool while
            # preserving the legacy absolute minimum so difficult topics remain
            # feasible without forcing extremely sparse grids.
            dynamic_floor = int(max_attempt_words * 0.5)
            min_attempt_words = min(
                max_attempt_words,
                max(8, min(30, dynamic_floor)),
            )

            fallback_component = None
            if word_components:
                largest_component = max(word_components, key=len)
                fallback_size = min(len(largest_component), max_attempt_words)
                fallback_component = list(largest_component[:fallback_size])

        _refresh_clues(initial=True)

        def _start_new_attempt(clues: Sequence[WordClue]) -> None:
            """Reset tracking sets for a new attempt candidate list."""

            nonlocal current_attempt_index
            nonlocal replacement_failure_streak
            current_attempt_index += 1
            used_canonical_words.clear()
            used_canonical_words.update(
                _canonical_answer(clue.word, language) for clue in clues
            )
            replacement_failure_streak = 0
            for canonical, info in list(rejected_canonical_words.items()):
                if canonical in used_canonical_words:
                    rejected_canonical_words.pop(canonical, None)
                    continue
                if info.last_attempt < current_attempt_index:
                    info.reasons.clear()
        def request_replacement(
            word: str, attempt_clues: Sequence[WordClue]
        ) -> WordClue | None:
            nonlocal replacement_requests
            nonlocal replacement_failure_streak
            canonical = _canonical_answer(word, language)
            replacement_prompt_words.add(canonical)
            used_canonical_words.discard(canonical)
            rejected_canonical_words[canonical] = RejectionInfo(
                last_attempt=current_attempt_index,
                reasons={"original"},
            )
            other_letters: set[str] = set()
            other_letter_sets: list[set[str]] = []
            target_letter_set = _canonical_letter_set(word, language)
            for clue in attempt_clues:
                if _canonical_answer(clue.word, language) == canonical:
                    continue
                letters = _canonical_letter_set(clue.word, language)
                other_letters.update(letters)
                other_letter_sets.append(letters)
            other_letters_text = ", ".join(sorted(other_letters))
            while True:
                if replacement_requests >= MAX_REPLACEMENT_REQUESTS:
                    logger.warning(
                        "Reached maximum replacement requests (%s) while trying to replace %s",
                        MAX_REPLACEMENT_REQUESTS,
                        word,
                    )
                    _refresh_clues(initial=False)
                    return None
                replacement_requests += 1
                prompt_suffix = ", ".join(sorted(replacement_prompt_words))
                avoided_words = set(used_canonical_words)
                avoided_words.update(
                    canonical_word
                    for canonical_word, info in rejected_canonical_words.items()
                    if info.last_attempt == current_attempt_index
                )
                avoided_words_text = ", ".join(sorted(avoided_words))
                soft_mode = (
                    replacement_failure_streak
                    >= SOFT_REPLACEMENT_RELAXATION_THRESHOLD
                )
                if other_letters_text:
                    if soft_mode:
                        letter_clause = (
                            "Старайся использовать буквы из: "
                            f"{other_letters_text}, допускаются редкие исключения."
                        )
                    else:
                        letter_clause = (
                            "Каждое слово должно содержать хотя бы одну букву из: "
                            f"{other_letters_text}."
                        )
                else:
                    letter_clause = "Предложи новые слова без повторов."
                replacement_theme = (
                    f"{theme}. Подбери 6-8 новых слов для кроссворда вместо: {prompt_suffix}. "
                    f"{letter_clause} Избегай слов: {avoided_words_text}."
                )
                logger.debug(
                    "Replacement prompt letters: %s; avoiding words: %s",
                    other_letters_text or "—",
                    avoided_words_text or "—",
                )
                logger.info(
                    "Requesting replacement clues (attempt %s) for: %s",
                    replacement_requests,
                    prompt_suffix,
                )
                new_clues = generate_clues(
                    theme=replacement_theme,
                    language=language,
                    min_results=6,
                    max_results=8,
                )
                new_validated = validate_word_list(language, new_clues, deduplicate=True)
                logger.info(
                    "Validated %s replacement candidates", len(new_validated)
                )
                scored_candidates: list[tuple[int, int, str, WordClue]] = []
                for candidate in new_validated:
                    candidate_canonical = _canonical_answer(candidate.word, language)
                    if candidate_canonical in used_canonical_words:
                        rejected_canonical_words[candidate_canonical] = RejectionInfo(
                            last_attempt=current_attempt_index,
                            reasons={"duplicate"},
                        )
                        continue
                    candidate_record = rejected_canonical_words.get(
                        candidate_canonical
                    )
                    if (
                        candidate_record is not None
                        and candidate_record.last_attempt == current_attempt_index
                    ):
                        continue
                    candidate_letters = _canonical_letter_set(candidate.word, language)
                    if other_letters and not soft_mode and not (
                        candidate_letters & other_letters
                    ):
                        logger.debug(
                            "Skipping replacement %s: no shared letters with current attempt",
                            candidate.word,
                        )
                        rejected_canonical_words[candidate_canonical] = RejectionInfo(
                            last_attempt=current_attempt_index,
                            reasons={"no_intersection"},
                        )
                        continue
                    if soft_mode and not (
                        candidate_letters & other_letters
                        or candidate_letters & target_letter_set
                    ):
                        logger.debug(
                            "Skipping %s even in relaxed mode: no overlap with target letters",
                            candidate.word,
                        )
                        rejected_canonical_words[candidate_canonical] = RejectionInfo(
                            last_attempt=current_attempt_index,
                            reasons={"relaxed_no_target"},
                        )
                        continue
                    score = (
                        len(candidate_letters & other_letters)
                        if other_letters
                        else sum(
                            len(candidate_letters & letters)
                            for letters in other_letter_sets
                        )
                    )
                    scored_candidates.append(
                        (score, len(candidate_canonical), candidate.word, candidate)
                    )

                for score, _, _, candidate in sorted(
                    scored_candidates,
                    key=lambda item: (-item[0], -item[1], item[2]),
                ):
                    candidate_canonical = _canonical_answer(candidate.word, language)
                    rejected_canonical_words.pop(candidate_canonical, None)
                    used_canonical_words.add(candidate_canonical)
                    replacement_failure_streak = 0
                    logger.debug(
                        "Selected replacement %s with score %s",
                        candidate.word,
                        score,
                    )
                    return candidate
                logger.warning(
                    "Replacement attempt %s did not provide new unique words",
                    replacement_requests,
                )
                replacement_failure_streak += 1


        restart_marker = object()

        def attempt_generation(
            candidate_clues: Sequence[WordClue], limit: int
        ) -> tuple[Puzzle | CompositePuzzle, GameState] | None:
            nonlocal attempted_component_split
            puzzle_id = uuid4().hex
            with logging_context(chat_id=chat_id, puzzle_id=puzzle_id):
                attempt_clues = list(candidate_clues)
                _start_new_attempt(attempt_clues)
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
                                "No replacement available for %s, restarting with new clues",
                                disconnected.word,
                            )
                            return restart_marker
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
                            and len(validated_clues) > 1
                        ):
                            components = word_components
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
            return None

        while True:
            restarted = False
            for limit in range(max_attempt_words, min_attempt_words - 1, -1):
                candidate_clues = _select_connected_clue_set(
                    word_components, language, limit
                )
                if not candidate_clues:
                    logger.debug(
                        "Skipping attempt with %s words: no connected subset available",
                        limit,
                    )
                    continue
                result = attempt_generation(candidate_clues, limit)
                if result is restart_marker:
                    restarted = True
                    break
                if result is not None:
                    return result

            if restarted:
                continue

            if fallback_component:
                logger.debug(
                    "Falling back to largest connected component with %s words",
                    len(fallback_component),
                )
                result = attempt_generation(
                    fallback_component, len(fallback_component)
                )
                if result is restart_marker:
                    continue
                if result is not None:
                    return result

            raise RuntimeError(
                "Не удалось сформировать кроссворд из сгенерированных слов"
            )


# ---------------------------------------------------------------------------
# Telegram command handlers
# ---------------------------------------------------------------------------


async def _start_new_private_game(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> int:
    chat = update.effective_chat
    message = update.effective_message
    chat_id = chat.id if chat else None
    logger.debug(
        "Chat %s initiated private /new", chat_id if chat_id is not None else "<unknown>"
    )

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
        _clear_pending_language(context, chat)
        set_chat_mode(context, MODE_IN_GAME)
        reminder_text = (
            "У вас уже есть активный кроссворд. Давате продолжим текущую игру!"
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

    _set_pending_language(context, chat, None)
    set_chat_mode(context, MODE_AWAIT_LANGUAGE)
    if message:
        await message.reply_text(
            "Выберите язык кроссворда (например: ru, en, it, es).",
        )
    return LANGUAGE_STATE


async def _start_new_group_game(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> int:
    chat = update.effective_chat
    message = update.effective_message
    user = update.effective_user
    if chat is None or message is None:
        return ConversationHandler.END
    if chat.id in state.generating_chats:
        await message.reply_text(
            "Мы всё ещё готовим предыдущую игру. Пожалуйста, подождите."
        )
        return ConversationHandler.END

    existing = _load_state_for_chat(chat.id)
    if (
        existing is not None
        and existing.mode != "single"
        and existing.status == "running"
    ):
        await message.reply_text(
            "В этой группе уже идёт игра. Завершите её или используйте /quit."
        )
        return ConversationHandler.END

    if existing is not None:
        _cleanup_game_state(existing)

    state.lobby_messages.pop(str(chat.id), None)
    context.chat_data.pop("lobby_message_id", None)

    now = time.time()
    host_id = user.id if user else None
    host_name = _user_display_name(user)
    dm_chat_id = _lookup_player_chat(host_id) if host_id else None
    game_state = GameState(
        chat_id=chat.id,
        puzzle_id="",
        filled_cells={},
        solved_slots=set(),
        score=0,
        started_at=now,
        last_update=now,
        hinted_cells=set(),
        host_id=host_id,
        game_id=str(chat.id),
        scoreboard={},
        mode="turn_based",
        status="lobby",
        players={},
    )
    if user and host_id is not None:
        _ensure_player_entry(game_state, user, host_name, dm_chat_id)
    game_state.language = None
    game_state.theme = None
    _store_state(game_state)
    set_chat_mode(context, MODE_AWAIT_LANGUAGE)
    _set_pending_language(context, chat, None)
    await message.reply_text(
        f"{host_name} создаёт новую игру! Укажите язык кроссворда (например: ru, en)."
    )
    return LANGUAGE_STATE


async def _process_join_code(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    code_raw: str,
) -> None:
    chat = update.effective_chat
    message = update.effective_message
    user = update.effective_user
    if chat is None or message is None or user is None:
        return
    if chat.type != ChatType.PRIVATE:
        await message.reply_text("Присоединение доступно только в личном чате с ботом.")
        return
    code = code_raw.strip()
    if not code:
        await message.reply_text("Укажите код для присоединения.")
        return
    code_upper = code.upper()
    game_id = state.join_codes.get(code_upper)
    if not game_id:
        await message.reply_text("Не удалось найти игру по этому коду.")
        return
    game_state = _load_state_by_game_id(game_id)
    if game_state is None or game_state.status != "lobby":
        await message.reply_text("Игра не готова к присоединению.")
        return
    if len(game_state.players) >= MAX_LOBBY_PLAYERS and user.id not in game_state.players:
        await message.reply_text("Достигнут лимит игроков в этой комнате (6).")
        return
    _register_player_chat(user.id, chat.id)
    existing = game_state.players.get(user.id)
    if existing:
        existing.dm_chat_id = chat.id
        _store_state(game_state)
        await message.reply_text(
            f"Вы уже в игре «{game_state.theme or 'без темы'}». Ожидаем старт."
        )
        await context.bot.send_message(
            chat_id=game_state.chat_id,
            text=f"{existing.name} снова с нами!",
        )
        return

    stored_name = context.user_data.get("player_name") if isinstance(
        context.user_data, dict
    ) else None
    if stored_name:
        player = _ensure_player_entry(game_state, user, str(stored_name), chat.id)
        _store_state(game_state)
        await message.reply_text(
            f"Добро пожаловать обратно, {player.name}! Ждите начала игры."
        )
        await context.bot.send_message(
            chat_id=game_state.chat_id,
            text=f"{player.name} подключился к игре!",
        )
        await _update_lobby_message(context, game_state)
        return

    context.user_data["pending_join"] = {"game_id": game_state.game_id, "code": code_upper}
    await message.reply_text(
        "Как вас представить другим игрокам?", reply_markup=ForceReply(selective=True)
    )


def _reset_new_game_context(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Clear chat and user state before launching a new game flow."""

    chat = update.effective_chat
    chat_id = chat.id if chat else None
    _cancel_reminder(context)
    _clear_generation_notice(context, chat_id)
    _clear_pending_language(context, chat)
    if isinstance(getattr(context, "chat_data", None), dict):
        context.chat_data.pop(BUTTON_NEW_GAME_KEY, None)
        context.chat_data.pop("lobby_message_id", None)
    set_chat_mode(context, MODE_IDLE)
    if isinstance(getattr(context, "user_data", None), dict):
        context.user_data.pop("pending_join", None)


@command_entrypoint(fallback=ConversationHandler.END)
async def start_new_game(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    _normalise_thread_id(update)
    chat = update.effective_chat
    message = update.effective_message
    user = update.effective_user
    if chat and chat.type == ChatType.PRIVATE and context.args:
        first_arg = context.args[0]
        if first_arg and first_arg.lower().startswith("join_"):
            await _process_join_code(update, context, first_arg[5:])
            return ConversationHandler.END

    if message is None:
        return ConversationHandler.END

    settings = state.settings
    admin_id = settings.admin_id if settings else None
    is_admin = user is not None and admin_id is not None and user.id == admin_id

    keyboard_rows = [
        [InlineKeyboardButton("Отгадывать одному", callback_data=NEW_GAME_MODE_SOLO)],
        [InlineKeyboardButton("Играть с друзьями", callback_data=NEW_GAME_MODE_GROUP)],
    ]
    if is_admin:
        target_chat_id = chat.id if chat else 0
        keyboard_rows.append(
            [
                InlineKeyboardButton(
                    "[адм.] Тестовая сессия",
                    callback_data=f"{ADMIN_TEST_GAME_CALLBACK_PREFIX}{target_chat_id}",
                )
            ]
        )

    description_lines = [
        "<b>Привет! 👋</b>",
        "Вы попали к боту <b>«Кроссворды»</b>. Выберите режим ниже.",
        "<b>Выберите, как хотите играть:</b>",
        "",
        "<b>Доступные варианты:</b>",
        "▫️ <b>Одиночная игра</b> — бот подготовит личный кроссворд.",
        "▫️ <b>Игра с друзьями</b> — создадим комнату для совместной игры.",
    ]
    if is_admin:
        description_lines.append(
            "▫️ <b>[адм.] Тестовая сессия</b> — копия текущей игры для проверки."
        )
    description_lines.extend(
        [
            "",
            "<i>Нажмите кнопку, чтобы начать.</i>",
        ]
    )

    await message.reply_text(
        "\n".join(description_lines),
        reply_markup=InlineKeyboardMarkup(keyboard_rows),
        disable_web_page_preview=True,
        parse_mode=constants.ParseMode.HTML,
    )

    return MENU_STATE


@command_entrypoint()
async def new_game_menu_callback_handler(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> int:
    _normalise_thread_id(update)
    query = update.callback_query
    if query is None or not query.data:
        return MENU_STATE

    data = query.data
    if not data.startswith(NEW_GAME_MENU_CALLBACK_PREFIX):
        return MENU_STATE

    chat = update.effective_chat
    mode = data[len(NEW_GAME_MENU_CALLBACK_PREFIX) :]

    if mode == "solo":
        if chat is None or chat.type != ChatType.PRIVATE:
            await query.answer(
                "Одиночный режим доступен только в личном чате.", show_alert=True
            )
            return MENU_STATE
        await query.answer()
        _reset_new_game_context(update, context)
        return await _start_new_private_game(update, context)

    if mode == "group":
        if chat is None or chat.type not in GROUP_CHAT_TYPES:
            await query.answer(
                "Режим для друзей запускается в групповом чате.", show_alert=True
            )
            return MENU_STATE
        await query.answer()
        _reset_new_game_context(update, context)
        return await _start_new_group_game(update, context)

    await query.answer()
    return MENU_STATE


@command_entrypoint()
async def new_game_menu_admin_proxy_handler(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> int:
    _normalise_thread_id(update)
    _reset_new_game_context(update, context)
    return ConversationHandler.END


@command_entrypoint(fallback=ConversationHandler.END)
async def handle_language(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    _normalise_thread_id(update)
    chat = update.effective_chat
    message = update.effective_message
    if chat is None or message is None or not message.text:
        return LANGUAGE_STATE
    if is_chat_mode_set(context) and get_chat_mode(context) != MODE_AWAIT_LANGUAGE:
        logger.debug(
            "Ignoring language input while in mode %s",
            get_chat_mode(context),
        )
        return LANGUAGE_STATE
    language = message.text.strip().lower()
    if not language or not language.isalpha():
        await message.reply_text("Пожалуйста, введите язык одним словом, например ru.")
        return LANGUAGE_STATE
    logger.debug(
        "Chat %s selected language %s",
        chat.id if chat else "<unknown>",
        language,
    )
    if chat.type in GROUP_CHAT_TYPES:
        game_state = _load_state_for_chat(chat.id)
        if not game_state or game_state.status != "lobby":
            await message.reply_text("Создайте новую игру командой /new в этом чате.")
            set_chat_mode(context, MODE_IDLE)
            _clear_pending_language(context, chat)
            return ConversationHandler.END
        game_state.language = language
        game_state.last_update = time.time()
        _store_state(game_state)
        _set_pending_language(context, chat, language)
        set_chat_mode(context, MODE_AWAIT_THEME)
        await message.reply_text("Отлично! Теперь укажите тему кроссворда.")
        return THEME_STATE

    _set_pending_language(context, chat, language)
    set_chat_mode(context, MODE_AWAIT_THEME)
    await message.reply_text("Отлично! Теперь укажите тему кроссворда.")
    return THEME_STATE


@command_entrypoint()
async def button_language_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    _normalise_thread_id(update)
    if is_chat_mode_set(context) and get_chat_mode(context) != MODE_AWAIT_LANGUAGE:
        logger.debug(
            "Ignoring button language input while in mode %s",
            get_chat_mode(context),
        )
        return
    flow_state = context.chat_data.get(BUTTON_NEW_GAME_KEY)
    if not flow_state or flow_state.get(BUTTON_STEP_KEY) != BUTTON_STEP_LANGUAGE:
        return
    message = update.effective_message
    chat = update.effective_chat
    if chat is None or message is None or not message.text:
        return
    language = message.text.strip().lower()
    if not language or not language.isalpha():
        await message.reply_text("Пожалуйста, введите язык одним словом, например ru.")
        return
    flow_state[BUTTON_LANGUAGE_KEY] = language
    flow_state[BUTTON_STEP_KEY] = BUTTON_STEP_THEME
    set_chat_mode(context, MODE_AWAIT_THEME)
    _set_pending_language(context, chat, language)
    logger.debug(
        "Chat %s selected language %s via button flow",
        update.effective_chat.id if update.effective_chat else "<unknown>",
        language,
    )
    await message.reply_text("Отлично! Теперь укажите тему кроссворда.")


@command_entrypoint(fallback=ConversationHandler.END)
async def handle_theme(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    _normalise_thread_id(update)
    chat = update.effective_chat
    message = update.effective_message
    if chat is None or message is None or not message.text:
        return THEME_STATE
    if is_chat_mode_set(context) and get_chat_mode(context) != MODE_AWAIT_THEME:
        logger.debug(
            "Ignoring theme input while in mode %s",
            get_chat_mode(context),
        )
        return THEME_STATE

    if chat.type in GROUP_CHAT_TYPES:
        game_state = _load_state_for_chat(chat.id)
        if not game_state or game_state.status != "lobby":
            await message.reply_text("Создайте новую игру командой /new в этом чате.")
            set_chat_mode(context, MODE_IDLE)
            _clear_pending_language(context, chat)
            return ConversationHandler.END
        theme = message.text.strip()
        if not theme:
            await message.reply_text("Введите тему, например: Древний Рим.")
            return THEME_STATE
        language = game_state.language or _get_pending_language(context, chat)
        if not language:
            await message.reply_text("Сначала выберите язык через команду /new.")
            set_chat_mode(context, MODE_IDLE)
            _clear_pending_language(context, chat)
            return ConversationHandler.END
        game_state.language = language
        game_state.theme = theme
        previous_puzzle_id = game_state.puzzle_id
        if previous_puzzle_id:
            delete_puzzle(previous_puzzle_id)
        game_state.puzzle_id = ""
        game_state.puzzle_ids = None
        game_state.filled_cells.clear()
        game_state.solved_slots.clear()
        game_state.hinted_cells = set()
        game_state.score = 0
        game_state.scoreboard.clear()
        game_state.turn_order.clear()
        game_state.turn_index = 0
        game_state.active_slot_id = None
        game_state.status = "lobby"
        game_state.last_update = time.time()
        _store_state(game_state)
        _clear_pending_language(context, chat)
        set_chat_mode(context, MODE_IDLE)
        await message.reply_text(
            "Тема сохранена! Готовим кроссворд, это может занять немного времени."
        )
        existing_task = state.lobby_generation_tasks.get(game_state.game_id)
        if existing_task:
            state.lobby_generation_tasks.pop(game_state.game_id, None)
            if not existing_task.done():
                existing_task.cancel()
                with suppress(asyncio.CancelledError):
                    await existing_task
        if game_state.game_id not in state.lobby_messages:
            await _publish_lobby_message(context, game_state)
        else:
            await _update_lobby_message(context, game_state)
        generation_task = asyncio.create_task(
            _run_lobby_puzzle_generation(
                context, game_state.game_id, language, theme
            )
        )
        state.lobby_generation_tasks[game_state.game_id] = generation_task
        return ConversationHandler.END

    if chat.id in state.generating_chats:
        await message.reply_text(
            "Мы всё ещё готовим ваш кроссворд. Пожалуйста, подождите."
        )
        set_chat_mode(context, MODE_IDLE)
        _clear_pending_language(context, chat)
        return ConversationHandler.END

    language = _get_pending_language(context, chat)
    if not language:
        await message.reply_text("Сначала выберите язык через команду /new.")
        set_chat_mode(context, MODE_IDLE)
        _clear_pending_language(context, chat)
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
        _clear_pending_language(context, chat)


@command_entrypoint()
async def button_theme_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    _normalise_thread_id(update)
    flow_state = context.chat_data.get(BUTTON_NEW_GAME_KEY)
    if not flow_state or flow_state.get(BUTTON_STEP_KEY) != BUTTON_STEP_THEME:
        return
    chat = update.effective_chat
    message = update.effective_message
    if chat is None or message is None or not message.text:
        return
    if chat.type in GROUP_CHAT_TYPES:
        await handle_theme(update, context)
        flow_state[BUTTON_STEP_KEY] = BUTTON_STEP_LANGUAGE
        return
    if is_chat_mode_set(context) and get_chat_mode(context) != MODE_AWAIT_THEME:
        logger.debug(
            "Ignoring button theme input while in mode %s",
            get_chat_mode(context),
        )
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


@command_entrypoint()
async def join_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    _normalise_thread_id(update)
    chat = update.effective_chat
    message = update.effective_message
    if chat is None or message is None:
        return
    if chat.type != ChatType.PRIVATE:
        await message.reply_text("Используйте эту команду в личном чате с ботом.")
        return
    if not context.args:
        await message.reply_text("Использование: /join <код>")
        return
    await _process_join_code(update, context, context.args[0])


@command_entrypoint()
async def join_name_response_handler(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    _normalise_thread_id(update)
    chat = update.effective_chat
    message = update.effective_message
    user = update.effective_user
    if chat is None or message is None or user is None:
        return
    if chat.type != ChatType.PRIVATE:
        return
    pending = context.user_data.get("pending_join")
    if not isinstance(pending, dict):
        return
    reply = message.reply_to_message
    if reply is None or reply.from_user is None or reply.from_user.id != context.bot.id:
        return
    game_id = pending.get("game_id")
    if not game_id:
        context.user_data.pop("pending_join", None)
        return
    name = message.text.strip() if message.text else ""
    if not name:
        await message.reply_text(
            "Имя не может быть пустым. Попробуйте снова.",
            reply_markup=ForceReply(selective=True),
        )
        return
    context.user_data["player_name"] = name
    context.user_data.pop("pending_join", None)
    game_state = _load_state_by_game_id(game_id)
    if not game_state or game_state.status != "lobby":
        await message.reply_text("Игра уже недоступна для присоединения.")
        return
    if len(game_state.players) >= MAX_LOBBY_PLAYERS and user.id not in game_state.players:
        await message.reply_text("К сожалению, комната уже заполнена.")
        return
    _register_player_chat(user.id, chat.id)
    player = _ensure_player_entry(game_state, user, name, chat.id)
    _store_state(game_state)
    await message.reply_text(
        f"Приятно познакомиться, {player.name}! Ждите начала игры.",
        reply_markup=ReplyKeyboardRemove(),
    )
    await context.bot.send_message(
        chat_id=game_state.chat_id,
        text=f"{player.name} присоединился к игре!",
    )
    await _update_lobby_message(context, game_state)


@command_entrypoint()
async def lobby_invite_callback_handler(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    _normalise_thread_id(update)
    query = update.callback_query
    if query is None or not query.data:
        return
    if not query.data.startswith(LOBBY_INVITE_CALLBACK_PREFIX):
        return
    game_id = query.data[len(LOBBY_INVITE_CALLBACK_PREFIX) :]
    game_state = _load_state_by_game_id(game_id)
    if not game_state or game_state.status != "lobby":
        await query.answer("Лобби недоступно.", show_alert=True)
        return
    user = update.effective_user
    if user is None:
        return
    dm_chat_id = _lookup_player_chat(user.id) or user.id
    text = (
        "Выберите контакт, которому хотите отправить приглашение, или поделитесь ссылкой"
        " через кнопку «Создать ссылку»."
    )
    keyboard = ReplyKeyboardMarkup(
        [[KeyboardButton(text="Поделиться контактом", request_contact=True)]],
        resize_keyboard=True,
        one_time_keyboard=True,
    )
    try:
        await context.bot.send_message(
            chat_id=dm_chat_id,
            text=text,
            reply_markup=keyboard,
        )
    except Forbidden:
        await query.answer(
            "Не могу отправить сообщение — напишите боту в личном чате.",
            show_alert=True,
        )
        return
    user_data = getattr(context, "user_data", None)
    if not isinstance(user_data, dict):
        user_data = {}
        setattr(context, "user_data", user_data)
    pending_invite = {"game_id": game_state.game_id}
    for existing_code, target in game_state.join_codes.items():
        if target == game_state.game_id:
            pending_invite["code"] = existing_code
            break
    user_data["pending_invite"] = pending_invite
    await query.answer("Открыл меню приглашений в личном чате.")


@command_entrypoint()
async def lobby_contact_handler(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    _normalise_thread_id(update)
    chat = update.effective_chat
    message = update.effective_message
    user = update.effective_user
    if chat is None or message is None or user is None:
        return
    if chat.type != ChatType.PRIVATE or message.contact is None:
        return
    user_data = getattr(context, "user_data", None)
    if isinstance(user_data, dict):
        pending = user_data.pop("pending_invite", None)
    else:
        pending = None
    if not isinstance(pending, dict):
        await message.reply_text(
            "Приглашение не найдено. Откройте меню лобби ещё раз.",
            reply_markup=ReplyKeyboardRemove(),
        )
        return
    game_id = pending.get("game_id")
    code_hint = pending.get("code")
    if not game_id:
        await message.reply_text(
            "Не удалось определить игру для приглашения. Попробуйте снова из лобби.",
            reply_markup=ReplyKeyboardRemove(),
        )
        return
    game_state = _load_state_by_game_id(game_id)
    if not game_state or game_state.status != "lobby":
        await message.reply_text(
            "Лобби больше недоступно. Создайте новое приглашение.",
            reply_markup=ReplyKeyboardRemove(),
        )
        return
    if user.id != game_state.host_id:
        await message.reply_text(
            "Только создатель комнаты может отправлять приглашения.",
            reply_markup=ReplyKeyboardRemove(),
        )
        return
    if len(game_state.players) >= MAX_LOBBY_PLAYERS:
        await message.reply_text(
            "Достигнут лимит игроков в этой комнате (6).",
            reply_markup=ReplyKeyboardRemove(),
        )
        return
    contact = message.contact
    target_user_id = contact.user_id
    if target_user_id == user.id:
        await message.reply_text(
            "Нельзя отправить приглашение самому себе.",
            reply_markup=ReplyKeyboardRemove(),
        )
        return
    if target_user_id and target_user_id in game_state.players:
        await message.reply_text(
            "Этот игрок уже участвует в игре.",
            reply_markup=ReplyKeyboardRemove(),
        )
        return
    join_code: str | None = None
    if isinstance(code_hint, str) and game_state.join_codes.get(code_hint) == game_state.game_id:
        join_code = code_hint
    if join_code is None:
        for existing_code, target in game_state.join_codes.items():
            if target == game_state.game_id:
                join_code = existing_code
                break
    generated_code = False
    if join_code is None:
        try:
            join_code = _assign_join_code(game_state)
            generated_code = True
        except RuntimeError:
            await message.reply_text(
                "Не удалось сгенерировать код присоединения. Попробуйте ещё раз позже.",
                reply_markup=ReplyKeyboardRemove(),
            )
            return
    needs_store = generated_code or state.join_codes.get(join_code) != game_state.game_id
    if needs_store:
        game_state.join_codes[join_code] = game_state.game_id
        _store_state(game_state)
    link = await _build_join_link(context, join_code)
    inviter_name = _user_display_name(user)
    lobby_theme = game_state.theme or "без темы"
    language = (game_state.language or "").upper()
    invite_lines = [
        f"{inviter_name} приглашает вас сыграть в кроссворд!",
    ]
    if language:
        invite_lines.append(f"Язык: {language}")
    invite_lines.append(f"Тема: {lobby_theme}")
    invite_lines.append(f"Код для присоединения: {join_code}")
    if link:
        invite_lines.append(f"Присоединяйтесь по ссылке: {link}")
    invite_text = "\n".join(invite_lines)
    contact_name_parts = [contact.first_name or "", contact.last_name or ""]
    contact_name = " ".join(part for part in contact_name_parts if part).strip()
    if not contact_name:
        contact_name = contact.phone_number or "игрок"
    sent_successfully = False
    error_message: str | None = None
    if target_user_id:
        try:
            await context.bot.send_message(
                chat_id=target_user_id,
                text=invite_text,
            )
            sent_successfully = True
        except Forbidden:
            error_message = (
                "Не удалось отправить приглашение: бот ещё не общался с этим пользователем."
            )
        except TelegramError:
            logger.exception(
                "Failed to deliver contact invite for game %s", game_state.game_id
            )
            error_message = "Возникла ошибка при отправке приглашения."
    else:
        error_message = (
            "У контакта нет Telegram-аккаунта. Передайте код вручную."
        )
    host_reply_lines = []
    if sent_successfully:
        host_reply_lines.append(f"Приглашение отправлено {contact_name}.")
    else:
        if not error_message:
            error_message = "Не удалось отправить приглашение."
        host_reply_lines.append(error_message)
        host_reply_lines.append("Передайте код вручную, чтобы игрок смог подключиться.")
    host_reply_lines.append(f"Код для подключения: {join_code}")
    if link:
        host_reply_lines.append(f"Ссылка: {link}")
    await message.reply_text(
        "\n".join(host_reply_lines),
        reply_markup=ReplyKeyboardRemove(),
    )


@command_entrypoint()
async def lobby_link_callback_handler(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    _normalise_thread_id(update)
    query = update.callback_query
    if query is None or not query.data:
        return
    if not query.data.startswith(LOBBY_LINK_CALLBACK_PREFIX):
        return
    game_id = query.data[len(LOBBY_LINK_CALLBACK_PREFIX) :]
    game_state = _load_state_by_game_id(game_id)
    if not game_state or game_state.status != "lobby":
        await query.answer("Лобби недоступно.", show_alert=True)
        return
    try:
        code = _assign_join_code(game_state)
    except RuntimeError:
        await query.answer("Не удалось создать новый код. Попробуйте ещё раз позже.", show_alert=True)
        return
    _store_state(game_state)
    link = await _build_join_link(context, code)
    user = update.effective_user
    if user is None:
        return
    dm_chat_id = _lookup_player_chat(user.id) or user.id
    parts = [f"Код для присоединения: {code}"]
    if link:
        parts.append(f"Ссылка: {link}")
    try:
        await context.bot.send_message(
            chat_id=dm_chat_id,
            text="\n".join(parts),
            reply_markup=ReplyKeyboardRemove(),
        )
    except Forbidden:
        await query.answer(
            "Не могу отправить ссылку — напишите боту в личном чате.",
            show_alert=True,
        )
        return
    await query.answer("Ссылка отправлена в личном чате.")


@command_entrypoint()
async def lobby_start_callback_handler(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    _normalise_thread_id(update)
    query = update.callback_query
    if query is None or not query.data:
        return
    data = query.data
    if data.startswith(LOBBY_WAIT_CALLBACK_PREFIX):
        await query.answer("Нужно минимум два игрока, чтобы начать игру.", show_alert=True)
        return
    if not data.startswith(LOBBY_START_CALLBACK_PREFIX):
        return
    game_id = data[len(LOBBY_START_CALLBACK_PREFIX) :]
    game_state = _load_state_by_game_id(game_id)
    if not game_state or game_state.status != "lobby":
        await query.answer("Игра уже запущена или недоступна.", show_alert=True)
        return
    query_answered = False
    if len(game_state.players) < 2:
        await query.answer("Нужно минимум два игрока, чтобы начать игру.", show_alert=True)
        await _update_lobby_message(context, game_state)
        return
    user = update.effective_user
    if not user or user.id != game_state.host_id:
        await query.answer("Только создатель комнаты может начинать игру.", show_alert=True)
        return
    puzzle = _load_puzzle_for_state(game_state)
    if puzzle is None:
        language = game_state.language
        theme = game_state.theme
        if not language or not theme:
            await query.answer("Сначала выберите язык и тему.", show_alert=True)
            return
        generation_task = state.lobby_generation_tasks.get(game_state.game_id)
        if generation_task and generation_task.done():
            state.lobby_generation_tasks.pop(game_state.game_id, None)
            generation_task = None
        if generation_task is None:
            generation_task = asyncio.create_task(
                _run_lobby_puzzle_generation(context, game_state.game_id, language, theme)
            )
            state.lobby_generation_tasks[game_state.game_id] = generation_task
        await query.answer("Кроссворд готовится, это может занять немного времени.")
        query_answered = True
        try:
            await generation_task
        except asyncio.CancelledError:
            logger.info("Lobby puzzle generation cancelled while starting game %s", game_id)
            return
        except Exception:
            logger.exception("Unexpected error while awaiting lobby generation for %s", game_id)
            await context.bot.send_message(
                chat_id=game_state.chat_id,
                text="Не удалось подготовить кроссворд. Попробуйте начать позже.",
            )
            return
        refreshed_state = _load_state_by_game_id(game_id)
        if refreshed_state is None or refreshed_state.puzzle_id == "":
            await context.bot.send_message(
                chat_id=game_state.chat_id,
                text="Кроссворд так и не был подготовлен. Попробуйте ещё раз позже.",
            )
            return
        game_state = refreshed_state
        puzzle = _load_puzzle_for_state(game_state)
        if puzzle is None:
            await context.bot.send_message(
                chat_id=game_state.chat_id,
                text="Не удалось загрузить кроссворд. Попробуйте позже.",
            )
            return
    players_sorted = sorted(
        game_state.players.values(), key=lambda player: player.joined_at
    )
    if len(players_sorted) < 2:
        await query.answer("Нужно минимум два игрока, чтобы начать игру.", show_alert=True)
        await _update_lobby_message(context, game_state)
        return
    game_state.turn_order = [player.user_id for player in players_sorted]
    game_state.turn_index = 0
    game_state.status = "running"
    game_state.active_slot_id = None
    game_state.started_at = time.time()
    game_state.last_update = time.time()
    for player in players_sorted:
        game_state.scoreboard[player.user_id] = 0
        player.answers_ok = 0
        player.answers_fail = 0
    _schedule_game_timers(context, game_state)
    _store_state(game_state)
    state.lobby_messages.pop(game_state.game_id, None)
    if not query_answered:
        await query.answer("Игра начинается!")
    await context.bot.send_message(
        chat_id=game_state.chat_id,
        text="Игра началась! Ходы идут по очереди.",
    )
    await _announce_turn(
        context,
        game_state,
        puzzle,
        prefix=f"Первым ходит {players_sorted[0].name}!",
    )


@command_entrypoint()
async def admin_test_game_callback_handler(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    _normalise_thread_id(update)
    query = update.callback_query
    if query is None or not query.data:
        return
    if not query.data.startswith(ADMIN_TEST_GAME_CALLBACK_PREFIX):
        return
    settings = state.settings
    if settings is None or settings.admin_id is None:
        await query.answer("Режим недоступен.", show_alert=True)
        return
    user = update.effective_user
    if user is None or user.id != settings.admin_id:
        await query.answer("Недостаточно прав.", show_alert=True)
        return
    payload = query.data[len(ADMIN_TEST_GAME_CALLBACK_PREFIX) :]
    target_chat_id: int | None = None
    if payload:
        with suppress(ValueError):
            target_chat_id = int(payload)
    if target_chat_id is None:
        chat = query.message.chat if query.message else update.effective_chat
        target_chat_id = chat.id if chat else None
    if target_chat_id is None:
        await query.answer()
        return
    base_state = _load_state_for_chat(target_chat_id)
    if base_state is None:
        await query.answer("Нет активной игры для теста.", show_alert=True)
        return
    if base_state.test_mode and base_state.status == "running":
        await query.answer("Тестовая игра уже запущена.", show_alert=True)
        return
    puzzle = _load_puzzle_for_state(base_state)
    if puzzle is None:
        await query.answer("Кроссворд ещё не готов.", show_alert=True)
        return
    cloned_puzzle, puzzle_id, component_ids = _clone_puzzle_for_test(puzzle)
    admin_game_id = f"admin:{target_chat_id}"
    existing = _load_state_by_game_id(admin_game_id)
    if existing is not None:
        _cleanup_game_state(existing)
    admin_id = settings.admin_id
    now = time.time()
    dm_chat_id = _lookup_player_chat(admin_id)
    chat = query.message.chat if query.message else update.effective_chat
    if chat and chat.type == ChatType.PRIVATE:
        dm_chat_id = chat.id
    admin_player = Player(
        user_id=admin_id,
        name=_user_display_name(user),
        dm_chat_id=dm_chat_id,
    )
    dummy_player = Player(user_id=DUMMY_USER_ID, name=DUMMY_NAME, is_bot=True)
    turn_order = [admin_id, DUMMY_USER_ID]
    if not ADMIN_FIRST:
        random.shuffle(turn_order)
    scoreboard = {admin_id: 0, DUMMY_USER_ID: 0}
    admin_state = GameState(
        chat_id=target_chat_id,
        puzzle_id=puzzle_id,
        puzzle_ids=component_ids,
        filled_cells={},
        solved_slots=set(),
        score=0,
        started_at=now,
        last_update=now,
        hinted_cells=set(),
        host_id=admin_id,
        game_id=admin_game_id,
        scoreboard=scoreboard,
        mode="turn_based",
        status="running",
        players={admin_id: admin_player, DUMMY_USER_ID: dummy_player},
        turn_order=turn_order,
        turn_index=0,
    )
    admin_state.test_mode = True
    admin_state.dummy_user_id = DUMMY_USER_ID
    admin_state.language = cloned_puzzle.language
    admin_state.theme = cloned_puzzle.theme
    admin_state.active_slot_id = None
    _register_player_chat(admin_id, dm_chat_id)
    set_chat_mode(context, MODE_IN_GAME)
    state.lobby_messages.pop(base_state.game_id, None)
    _store_state(admin_state)
    _schedule_game_timers(context, admin_state)
    _store_state(admin_state)
    await query.answer("Тестовая игра запущена!")
    intro_lines = [
        "[адм.] Тестовая игра 1×1 запущена!",
        f"Игроки: {_user_display_name(user)} и {DUMMY_NAME}.",
    ]
    first_player = admin_state.players.get(admin_state.turn_order[0])
    if first_player:
        intro_lines.append(f"Первым ходит {first_player.name}.")
    try:
        await context.bot.send_message(
            chat_id=target_chat_id,
            text="\n".join(intro_lines),
        )
    except Exception:  # noqa: BLE001
        logger.exception(
            "Failed to announce admin test game start for chat %s", target_chat_id
        )
    await _announce_turn(
        context,
        admin_state,
        cloned_puzzle,
        prefix=(
            f"Первым ходит {first_player.name}!" if first_player else "Игра начинается!"
        ),
    )


@command_entrypoint()
async def turn_select_callback_handler(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    _normalise_thread_id(update)
    query = update.callback_query
    if query is None or not query.data:
        return
    if not query.data.startswith(TURN_SELECT_CALLBACK_PREFIX):
        return
    game_id = query.data[len(TURN_SELECT_CALLBACK_PREFIX) :]
    game_state = _load_state_by_game_id(game_id)
    if not game_state or game_state.status != "running":
        await query.answer("Игра не активна.", show_alert=True)
        return
    if game_state.mode != "turn_based":
        await query.answer("Эта функция доступна только в мультиплеере.", show_alert=True)
        return
    current_player = _current_player(game_state)
    user = query.from_user
    if not current_player or not user or current_player.user_id != user.id:
        if current_player:
            await query.answer(
                f"Сейчас ход {current_player.name}.", show_alert=True
            )
        else:
            await query.answer("Сейчас нельзя выбирать подсказку.", show_alert=True)
        return
    puzzle = _load_puzzle_for_state(game_state)
    if puzzle is None:
        await query.answer("Кроссворд недоступен.", show_alert=True)
        return
    keyboard = _build_slot_keyboard(game_state, puzzle)
    await query.answer("Открываю список вопросов.")
    target_chat = current_player.dm_chat_id or game_state.chat_id
    try:
        await context.bot.send_message(
            chat_id=target_chat,
            text="Выберите слот для ответа:",
            reply_markup=keyboard,
        )
    except Exception:  # noqa: BLE001
        logger.exception(
            "Failed to send slot selection keyboard for game %s", game_state.game_id
        )


@command_entrypoint()
async def turn_slot_callback_handler(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    _normalise_thread_id(update)
    query = update.callback_query
    if query is None or not query.data:
        return
    if not query.data.startswith(TURN_SLOT_CALLBACK_PREFIX):
        return
    payload = query.data[len(TURN_SLOT_CALLBACK_PREFIX) :]
    if "|" not in payload:
        await query.answer()
        return
    game_id, slot_identifier = payload.split("|", 1)
    game_state = _load_state_by_game_id(game_id)
    if not game_state or game_state.status != "running":
        await query.answer("Игра завершена.", show_alert=True)
        return
    if game_state.mode != "turn_based":
        await query.answer("Комната не в пошаговом режиме.", show_alert=True)
        return
    current_player = _current_player(game_state)
    user = query.from_user
    if not current_player or not user or current_player.user_id != user.id:
        if current_player:
            await query.answer(
                f"Сейчас ход {current_player.name}.", show_alert=True
            )
        else:
            await query.answer("Сейчас нельзя выбрать слот.", show_alert=True)
        return
    puzzle = _load_puzzle_for_state(game_state)
    if puzzle is None:
        await query.answer("Кроссворд недоступен.", show_alert=True)
        return
    slot_ref = _find_slot_by_identifier(puzzle, slot_identifier)
    if slot_ref is None:
        await query.answer("Этот слот уже недоступен.", show_alert=True)
        return
    if _normalise_slot_id(slot_ref.public_id) in {
        _normalise_slot_id(entry) for entry in game_state.solved_slots
    }:
        await query.answer("Этот слот уже решён.", show_alert=True)
        return
    game_state.active_slot_id = _normalise_slot_id(slot_ref.public_id)
    game_state.last_update = time.time()
    _store_state(game_state)
    await query.answer("Слот выбран!", show_alert=False)
    clue = slot_ref.slot.clue or "(без подсказки)"
    announcement = (
        f"{current_player.name} отвечает на {slot_ref.public_id}: {clue}"
    )
    try:
        await context.bot.send_message(
            chat_id=game_state.chat_id,
            text=announcement,
        )
    except Exception:  # noqa: BLE001
        logger.exception("Failed to announce selected slot in game %s", game_state.game_id)
    if current_player.dm_chat_id:
        try:
            await context.bot.send_message(
                chat_id=current_player.dm_chat_id,
                text=f"Вы выбрали {slot_ref.public_id}. Подсказка: {clue}",
            )
        except Exception:  # noqa: BLE001
            logger.exception(
                "Failed to send DM confirmation for player %s", current_player.user_id
            )


@command_entrypoint(fallback=ConversationHandler.END)
async def cancel_new_game(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    _normalise_thread_id(update)
    set_chat_mode(context, MODE_IDLE)
    chat = update.effective_chat
    _clear_pending_language(context, chat)
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
    game_state = _load_state_for_chat(chat.id)
    if not game_state:
        await message.reply_text("Нет активного кроссворда. Используйте /new.")
        log_abort("missing_game_state")
        return
    puzzle = _load_puzzle_for_state(game_state)
    if puzzle is None:
        await message.reply_text("Не удалось загрузить кроссворд. Попробуйте начать заново.")
        log_abort("missing_puzzle")
        return

    in_turn_mode = game_state.mode == "turn_based"
    player_id: int | None = None
    current_player: Player | None = None
    if in_turn_mode:
        if game_state.status != "running":
            await message.reply_text("Игра ещё не запущена или уже завершена.")
            log_abort("turn_not_running")
            return
        player_id = _resolve_player_from_chat(game_state, chat, message)
        if player_id is None:
            await message.reply_text(
                "Не удалось определить игрока. Используйте личный чат или отметьте бота."
            )
            log_abort("player_not_identified")
            return
        current_player_id = _current_player_id(game_state)
        current_player = (
            game_state.players.get(current_player_id) if current_player_id is not None else None
        )
        if current_player_id is None or current_player is None:
            await message.reply_text("Сейчас нет активного игрока. Подождите объявления.")
            log_abort("current_player_missing")
            return
        if player_id != current_player_id:
            await message.reply_text(f"Сейчас ход {current_player.name}.")
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

        if in_turn_mode:
            expected_slot = game_state.active_slot_id
            if not expected_slot:
                await message.reply_text(
                    "Сначала выберите слот кнопкой «Выбрать подсказку»."
                )
                await refresh_clues_if_needed()
                log_abort("slot_not_selected", slot_identifier=public_id)
                return
            if public_id != expected_slot:
                await message.reply_text(
                    "Сначала выберите этот слот кнопкой «Выбрать подсказку»."
                )
                await refresh_clues_if_needed()
                log_abort("slot_not_selected", slot_identifier=public_id)
                return

        if _canonical_answer(candidate, puzzle.language) != _canonical_answer(
            slot.answer,
            puzzle.language,
        ):
            logger.info("Incorrect answer for slot %s", selected_slot_ref.public_id)
            if in_turn_mode:
                if current_player:
                    current_player.answers_fail += 1
                await message.reply_text("Ответ неверный. Ход переходит к следующему игроку.")
                await refresh_clues_if_needed()
                _cancel_turn_timers(game_state)
                _advance_turn(game_state)
                _store_state(game_state)
                prefix = (
                    f"{current_player.name} ошибся." if current_player else "Ответ неверный."
                )
                await _announce_turn(context, game_state, puzzle, prefix=prefix)
            else:
                await message.reply_text("Ответ неверный, попробуйте ещё раз.")
                await refresh_clues_if_needed()
            log_abort("answer_incorrect", slot_identifier=public_id)
            return

        game_state.score += slot.length
        if in_turn_mode and player_id is not None:
            _record_score(game_state, slot.length, user_id=player_id)
            if current_player:
                current_player.answers_ok += 1
        else:
            _record_score(game_state, slot.length)
        if in_turn_mode:
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

        if in_turn_mode:
            _cancel_turn_timers(game_state)
            if _all_slots_solved(puzzle, game_state):
                await _finish_game(
                    context,
                    game_state,
                    reason=(
                        f"{current_player.name} разгадал последний слот!"
                        if current_player
                        else "Все слова разгаданы!"
                    ),
                )
                return
            if chat.id != game_state.chat_id:
                try:
                    name = current_player.name if current_player else "Игрок"
                    await context.bot.send_message(
                        chat_id=game_state.chat_id,
                        text=f"{name} разгадал {selected_slot_ref.public_id}!",
                    )
                except Exception:  # noqa: BLE001
                    logger.exception(
                        "Failed to notify group about answer in game %s",
                        game_state.game_id,
                    )
            _advance_turn(game_state)
            _store_state(game_state)
            await _announce_turn(
                context,
                game_state,
                puzzle,
                prefix=(
                    f"{current_player.name} разгадал {selected_slot_ref.public_id}!"
                    if current_player
                    else f"Слот {selected_slot_ref.public_id} разгадан!"
                ),
            )
        else:
            if _all_slots_solved(puzzle, game_state):
                _cancel_reminder(context)
                set_chat_mode(context, MODE_IDLE)
                await message.reply_text(
                    "🎉 <b>Поздравляем!</b>\nВсе слова разгаданы! ✨",
                    parse_mode=constants.ParseMode.HTML,
                )
                await _send_completion_options(context, chat.id, message, puzzle)
            else:
                await refresh_clues_if_needed()


@command_entrypoint()
async def admin_menu_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    _normalise_thread_id(update)
    message = update.effective_message
    chat = update.effective_chat
    user = update.effective_user
    if message is None or chat is None or user is None:
        return
    settings = state.settings
    if settings is None or settings.admin_id is None:
        return
    if user.id != settings.admin_id:
        logger.debug("Ignoring /admin command from non-admin %s", user.id)
        return
    keyboard = InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton(
                    "[адм.] Тестовая игра 1×1",
                    callback_data=f"{ADMIN_TEST_GAME_CALLBACK_PREFIX}{chat.id}",
                )
            ]
        ]
    )
    await message.reply_text("Служебное меню:", reply_markup=keyboard)


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
    chat = update.effective_chat
    message = update.effective_message
    if chat is None or message is None:
        return
    if chat.type in GROUP_CHAT_TYPES:
        game_state = _load_state_for_chat(chat.id)
        if not game_state or game_state.mode != "turn_based":
            if not await _reject_group_chat(update):
                return
    else:
        if not await _reject_group_chat(update):
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
    pending_language = _get_pending_language(context, chat)
    if is_chat_mode_set(context) and current_mode != MODE_IN_GAME:
        if pending_language is not None:
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
    if not is_chat_mode_set(context) and pending_language is not None:
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
        game_state = _load_state_for_chat(chat.id)
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
    chat = update.effective_chat
    message = update.effective_message
    if chat is None or message is None:
        return
    if chat.type in GROUP_CHAT_TYPES:
        game_state = _load_state_for_chat(chat.id)
        if not game_state or game_state.mode != "turn_based":
            if not await _reject_group_chat(update):
                return
    else:
        if not await _reject_group_chat(update):
            return
        game_state = _load_state_for_chat(chat.id)
    if not game_state:
        await message.reply_text("Нет активной игры. Используйте /new.")
        return
    logger.debug("Chat %s requested /hint", chat.id)
    if is_chat_mode_set(context) and get_chat_mode(context) != MODE_IN_GAME:
        await message.reply_text("Нет активной игры. Используйте /new.")
        return

    puzzle = _load_puzzle_for_state(game_state)
    if puzzle is None:
        await message.reply_text("Не удалось загрузить кроссворд.")
        return

    with logging_context(puzzle_id=puzzle.id):
        in_turn_mode = game_state.mode == "turn_based"
        player_id: int | None = None
        current_player: Player | None = None
        if in_turn_mode:
            if game_state.status != "running":
                await message.reply_text("Игра ещё не запущена или завершена.")
                return
            player_id = _resolve_player_from_chat(game_state, chat, message)
            if player_id is None:
                await message.reply_text(
                    "Не удалось определить игрока. Используйте личный чат с ботом."
                )
                return
            current_player_id = _current_player_id(game_state)
            current_player = (
                game_state.players.get(current_player_id)
                if current_player_id is not None
                else None
            )
            if current_player_id is None or current_player is None:
                await message.reply_text("Сейчас нет активного игрока.")
                return
            if player_id != current_player_id:
                await message.reply_text(f"Сейчас ход {current_player.name}.")
                return
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

        normalised_public_id = _normalise_slot_id(slot_ref.public_id)
        if in_turn_mode and game_state.active_slot_id:
            if normalised_public_id != game_state.active_slot_id:
                await message.reply_text(
                    "Сначала выберите этот слот кнопкой «Выбрать подсказку»."
                )
                return

        result = _reveal_letter(
            game_state, slot_ref, slot_ref.slot.answer, user_id=player_id
        )
        if result is None:
            _record_hint_usage(game_state, slot_ref.public_id, user_id=player_id)
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

        if in_turn_mode and player_id is not None:
            _record_score(game_state, -HINT_PENALTY, user_id=player_id)
            game_state.last_update = time.time()
            _store_state(game_state)

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
async def finish_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    _normalise_thread_id(update)
    chat = update.effective_chat
    message = update.effective_message
    user = update.effective_user
    if chat is None or message is None or user is None:
        return
    game_state = _load_state_for_chat(chat.id)
    if not game_state or game_state.mode != "turn_based":
        await message.reply_text("В этом чате нет мультиплеерной игры.")
        return
    if user.id != game_state.host_id:
        await message.reply_text("Завершить игру может только хост.")
        return
    if game_state.status == "finished":
        await message.reply_text("Игра уже завершена.")
        return
    await _finish_game(context, game_state, reason="Игра завершена хостом.")


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
        _clear_pending_language(context, chat)
        context.chat_data[BUTTON_NEW_GAME_KEY] = {BUTTON_STEP_KEY: BUTTON_STEP_LANGUAGE}
        set_chat_mode(context, MODE_AWAIT_LANGUAGE)
        await context.bot.send_message(
            chat_id=chat.id,
            text="Выберите язык кроссворда (например: ru, en, it, es).",
        )
        logger.info("Prompted chat %s to start new puzzle flow", chat.id)
        return

    logger.debug("Unhandled completion callback payload: %s", data)


def configure_telegram_handlers(telegram_application: Application) -> None:
    conversation = ConversationHandler(
        entry_points=[CommandHandler(["new", "start"], start_new_game)],
        states={
            MENU_STATE: [
                CallbackQueryHandler(
                    new_game_menu_callback_handler,
                    pattern=fr"^{NEW_GAME_MENU_CALLBACK_PREFIX}.*$",
                ),
                CallbackQueryHandler(
                    new_game_menu_admin_proxy_handler,
                    pattern=fr"^{ADMIN_TEST_GAME_CALLBACK_PREFIX}.*$",
                    block=False,
                ),
            ],
            LANGUAGE_STATE: [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_language)],
            THEME_STATE: [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_theme)],
        },
        fallbacks=[CommandHandler("cancel", cancel_new_game)],
        name="new_game_conversation",
        block=False,
    )
    telegram_application.add_handler(conversation)
    telegram_application.add_handler(
        MessageHandler(filters.ALL, track_player_message),
        group=-1,
    )
    telegram_application.add_handler(
        CallbackQueryHandler(track_player_callback, pattern=".*", block=False),
        group=-1,
    )
    telegram_application.add_handler(
        MessageHandler(filters.Regex(ADMIN_COMMAND_PATTERN), admin_answer_request_handler)
    )
    telegram_application.add_handler(
        MessageHandler(
            filters.TEXT & ~filters.COMMAND,
            inline_answer_handler,
            block=False,
        )
    )
    telegram_application.add_handler(
        MessageHandler(
            filters.TEXT & ~filters.COMMAND,
            button_language_handler,
            block=False,
        )
    )
    telegram_application.add_handler(
        MessageHandler(
            filters.TEXT & ~filters.COMMAND,
            button_theme_handler,
            block=False,
        )
    )
    telegram_application.add_handler(
        MessageHandler(
            filters.TEXT & filters.REPLY & ~filters.COMMAND,
            join_name_response_handler,
            block=False,
        )
    )
    telegram_application.add_handler(
        MessageHandler(
            filters.CONTACT,
            lobby_contact_handler,
            block=False,
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
    telegram_application.add_handler(CommandHandler("join", join_command))
    telegram_application.add_handler(CommandHandler("admin", admin_menu_command))
    telegram_application.add_handler(
        CallbackQueryHandler(
            completion_callback_handler,
            pattern=fr"^{COMPLETION_CALLBACK_PREFIX}",
        )
    )
    telegram_application.add_handler(
        CallbackQueryHandler(
            lobby_invite_callback_handler,
            pattern=fr"^{LOBBY_INVITE_CALLBACK_PREFIX}.*",
            block=False,
        )
    )
    telegram_application.add_handler(
        CallbackQueryHandler(
            lobby_link_callback_handler,
            pattern=fr"^{LOBBY_LINK_CALLBACK_PREFIX}.*",
            block=False,
        )
    )
    telegram_application.add_handler(
        CallbackQueryHandler(
            lobby_start_callback_handler,
            pattern=fr"^{LOBBY_START_CALLBACK_PREFIX}.*|^{LOBBY_WAIT_CALLBACK_PREFIX}.*",
            block=False,
        )
    )
    telegram_application.add_handler(
        CallbackQueryHandler(
            turn_select_callback_handler,
            pattern=fr"^{TURN_SELECT_CALLBACK_PREFIX}.*",
            block=False,
        )
    )
    telegram_application.add_handler(
        CallbackQueryHandler(
            turn_slot_callback_handler,
            pattern=fr"^{TURN_SLOT_CALLBACK_PREFIX}.*",
            block=False,
        )
    )
    telegram_application.add_handler(
        CallbackQueryHandler(
            admin_test_game_callback_handler,
            pattern=fr"^{ADMIN_TEST_GAME_CALLBACK_PREFIX}.*",
            block=False,
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

