"""FastAPI application entrypoint for Telegram webhook processing."""

from __future__ import annotations

import asyncio
import os
import time
from uuid import uuid4
from contextlib import suppress
from functools import wraps
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from telegram import Update, constants
from telegram.constants import ChatType
from telegram.ext import (
    Application,
    CallbackContext,
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
    Direction,
    Puzzle,
    Slot,
    fill_puzzle_with_words,
    puzzle_from_dict,
    puzzle_to_dict,
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

    return Settings(
        telegram_bot_token=required_vars["TELEGRAM_BOT_TOKEN"],
        public_url=required_vars["PUBLIC_URL"].rstrip("/"),
        webhook_secret=required_vars["WEBHOOK_SECRET"],
        webhook_path=required_vars["WEBHOOK_PATH"] if required_vars["WEBHOOK_PATH"].startswith("/") else f"/{required_vars['WEBHOOK_PATH']}",
        webhook_check_interval=check_interval,
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
        self.active_states: dict[int, GameState] = {}


state = AppState()


def get_telegram_application() -> Application:
    if state.telegram_app is None:
        logger.error("Telegram application is not initialized")
        raise HTTPException(status_code=503, detail="Telegram application is not initialized")
    return state.telegram_app


def _cleanup_chat_resources(chat_id: int, puzzle_id: str | None = None) -> None:
    """Remove in-memory and persisted resources for the provided chat."""

    with logging_context(chat_id=chat_id, puzzle_id=puzzle_id):
        if chat_id in state.active_states:
            del state.active_states[chat_id]
        delete_state(chat_id)
        if puzzle_id:
            delete_puzzle(puzzle_id)
        logger.info("Cleaned up resources for chat %s", chat_id)


def _cleanup_game_state(game_state: GameState | None) -> None:
    if game_state is None:
        return
    _cleanup_chat_resources(game_state.chat_id, game_state.puzzle_id)


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

REMINDER_DELAY_SECONDS = 10 * 60

DEFAULT_PUZZLE_TEMPLATE = (
    "##.#...#.##",
    "#..##.##..#",
    "..#..#..#..",
    ".#...#...#.",
    "..##...##..",
    "###.....###",
    "..##...##..",
    ".#...#...#.",
    "..#..#..#..",
    "#..##.##..#",
    "##.#...#.##",
)


def _parse_block_template(template: Sequence[str]) -> set[tuple[int, int]]:
    positions: set[tuple[int, int]] = set()
    for row, row_value in enumerate(template):
        for col, char in enumerate(row_value):
            if char == "#":
                positions.add((row, col))
    return positions


DEFAULT_BLOCK_POSITIONS = _parse_block_template(DEFAULT_PUZZLE_TEMPLATE)


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


def _coord_key(row: int, col: int) -> str:
    return f"{row},{col}"


def _normalise_slot_id(slot_id: str) -> str:
    return slot_id.strip().upper()


def _find_slot(puzzle: Puzzle, slot_id: str) -> Optional[Slot]:
    normalised = _normalise_slot_id(slot_id)
    for slot in puzzle.slots:
        if slot.slot_id.upper() == normalised:
            return slot
    return None


def _canonical_answer(word: str, language: str) -> str:
    transformed = (word or "").strip().upper()
    if language.lower() == "ru":
        transformed = transformed.replace("Ё", "Е")
    return transformed


def _ensure_hint_set(game_state: GameState) -> set[str]:
    if game_state.hinted_cells is None:
        game_state.hinted_cells = set()
    return game_state.hinted_cells


def _store_state(game_state: GameState) -> None:
    with logging_context(chat_id=game_state.chat_id, puzzle_id=game_state.puzzle_id):
        state.active_states[game_state.chat_id] = game_state
        save_state(game_state)
        logger.info("Game state persisted for chat %s", game_state.chat_id)


def _load_state_for_chat(chat_id: int) -> Optional[GameState]:
    with logging_context(chat_id=chat_id):
        if chat_id in state.active_states:
            return state.active_states[chat_id]
        restored = load_state(chat_id)
        if restored is None:
            return None
        state.active_states[chat_id] = restored
        logger.info("Restored state from disk during command handling")
        return restored


def _load_puzzle_for_state(game_state: GameState) -> Optional[Puzzle]:
    with logging_context(chat_id=game_state.chat_id, puzzle_id=game_state.puzzle_id):
        payload = load_puzzle(game_state.puzzle_id)
        if payload is None:
            logger.error("Puzzle referenced by chat is missing")
            return None
        logger.debug("Loaded puzzle definition for rendering or clues")
        return puzzle_from_dict(dict(payload))


def _format_clue_section(slots: Iterable[Slot]) -> str:
    lines = []
    for slot in slots:
        clue_text = slot.clue or "(нет подсказки)"
        lines.append(f"{slot.slot_id} ({slot.length}): {clue_text}")
    return "\n".join(lines) if lines else "(подсказок нет)"


def _format_clues_message(puzzle: Puzzle) -> str:
    across = [slot for slot in puzzle.slots if slot.direction is Direction.ACROSS]
    down = [slot for slot in puzzle.slots if slot.direction is Direction.DOWN]
    across_text = _format_clue_section(across)
    down_text = _format_clue_section(down)
    return f"Across:\n{across_text}\n\nDown:\n{down_text}"


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


def _apply_answer_to_state(game_state: GameState, slot: Slot, answer: str) -> None:
    with logging_context(chat_id=game_state.chat_id, puzzle_id=game_state.puzzle_id):
        logger.debug("Applying answer for slot %s", slot.slot_id)
        keys: list[str] = []
        for index, (row, col) in enumerate(slot.coordinates()):
            key = _coord_key(row, col)
            keys.append(key)
            if index < len(answer):
                game_state.filled_cells[key] = answer[index]
        hint_set = _ensure_hint_set(game_state)
        for key in keys:
            hint_set.discard(key)
        game_state.solved_slots.add(slot.slot_id)
        game_state.last_update = time.time()
        _store_state(game_state)


def _reveal_letter(game_state: GameState, slot: Slot, answer: str) -> Optional[tuple[int, str]]:
    hint_set = _ensure_hint_set(game_state)
    for index, (row, col) in enumerate(slot.coordinates()):
        key = _coord_key(row, col)
        if key in game_state.filled_cells:
            continue
        if index >= len(answer):
            break
        letter = answer[index]
        game_state.filled_cells[key] = letter
        hint_set.add(key)
        game_state.hints_used += 1
        game_state.last_update = time.time()
        _store_state(game_state)
        return index, letter
    return None


def _all_slots_solved(puzzle: Puzzle, game_state: GameState) -> bool:
    solved = set(game_state.solved_slots)
    return all(slot.slot_id in solved for slot in puzzle.slots if slot.answer)


def _solve_remaining_slots(game_state: GameState, puzzle: Puzzle) -> list[tuple[str, str]]:
    solved_now: list[tuple[str, str]] = []
    hint_set = _ensure_hint_set(game_state)
    for slot in puzzle.slots:
        if not slot.answer:
            continue
        if slot.slot_id in game_state.solved_slots:
            continue
        answer = slot.answer
        for index, (row, col) in enumerate(slot.coordinates()):
            if index >= len(answer):
                break
            key = _coord_key(row, col)
            game_state.filled_cells[key] = answer[index]
            hint_set.discard(key)
        game_state.solved_slots.add(slot.slot_id)
        solved_now.append((slot.slot_id, answer))
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


def _assign_clues_to_slots(puzzle: Puzzle, clues: Sequence[WordClue]) -> None:
    language = puzzle.language
    clue_map: dict[str, str] = {}
    for clue in clues:
        clue_map[_canonical_answer(clue.word, language)] = clue.clue
    for slot in puzzle.slots:
        if not slot.answer:
            continue
        canonical = _canonical_answer(slot.answer, language)
        slot.clue = clue_map.get(canonical, f"Слово из {slot.length} букв")


def _generate_puzzle(chat_id: int, language: str, theme: str) -> tuple[Puzzle, GameState]:
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

        rows = len(DEFAULT_PUZZLE_TEMPLATE)
        cols = len(DEFAULT_PUZZLE_TEMPLATE[0]) if rows else 11

        for limit in range(max_attempt_words, min_attempt_words - 1, -1):
            candidate_words = [clue.word for clue in validated_clues[:limit]]
            puzzle_id = uuid4().hex
            puzzle = Puzzle.from_size(
                puzzle_id=puzzle_id,
                theme=theme,
                language=language,
                rows=rows,
                cols=cols,
                block_positions=DEFAULT_BLOCK_POSITIONS,
            )
            with logging_context(chat_id=chat_id, puzzle_id=puzzle.id):
                if fill_puzzle_with_words(puzzle, candidate_words):
                    logger.info("Filled puzzle grid using %s candidate words", limit)
                    _assign_clues_to_slots(puzzle, validated_clues)
                    save_puzzle(puzzle.id, puzzle_to_dict(puzzle))
                    now = time.time()
                    game_state = GameState(
                        chat_id=chat_id,
                        puzzle_id=puzzle.id,
                        filled_cells={},
                        solved_slots=set(),
                        score=0,
                        hints_used=0,
                        started_at=now,
                        last_update=now,
                        hinted_cells=set(),
                    )
                    _store_state(game_state)
                    logger.info("Generated puzzle ready for delivery")
                    return puzzle, game_state
                logger.debug("Attempt with %s words failed to fill puzzle", limit)

        raise RuntimeError("Не удалось сформировать кроссворд из сгенерированных слов")


# ---------------------------------------------------------------------------
# Telegram command handlers
# ---------------------------------------------------------------------------


@command_entrypoint(fallback=ConversationHandler.END)
async def start_new_game(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    _normalise_thread_id(update)
    if not await _reject_group_chat(update):
        return ConversationHandler.END
    logger.debug("Chat %s initiated /new", update.effective_chat.id if update.effective_chat else "<unknown>")
    context.user_data["new_game_language"] = None
    if update.effective_message:
        await update.effective_message.reply_text(
            "Выберите язык кроссворда (например: ru, en, it, es).",
        )
    return LANGUAGE_STATE


@command_entrypoint(fallback=ConversationHandler.END)
async def handle_language(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    _normalise_thread_id(update)
    if not await _reject_group_chat(update):
        return ConversationHandler.END
    message = update.effective_message
    if message is None or not message.text:
        return LANGUAGE_STATE
    language = message.text.strip().lower()
    if not language or not language.isalpha():
        await message.reply_text("Пожалуйста, введите язык одним словом, например ru.")
        return LANGUAGE_STATE
    logger.debug("Chat %s selected language %s", update.effective_chat.id if update.effective_chat else "<unknown>", language)
    context.user_data["new_game_language"] = language
    await message.reply_text("Отлично! Теперь укажите тему кроссворда.")
    return THEME_STATE


@command_entrypoint(fallback=ConversationHandler.END)
async def handle_theme(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    _normalise_thread_id(update)
    if not await _reject_group_chat(update):
        return ConversationHandler.END
    message = update.effective_message
    if message is None or not message.text:
        return THEME_STATE
    language = context.user_data.get("new_game_language")
    if not language:
        await message.reply_text("Сначала выберите язык через команду /new.")
        return ConversationHandler.END
    theme = message.text.strip()
    if not theme:
        await message.reply_text("Введите тему, например: Древний Рим.")
        return THEME_STATE

    chat = update.effective_chat
    if chat is None:
        return ConversationHandler.END

    logger.info("Chat %s requested theme '%s'", chat.id, theme)
    await message.reply_text("Готовлю кроссворд, это может занять немного времени...")
    loop = asyncio.get_running_loop()
    try:
        puzzle, game_state = await loop.run_in_executor(
            None, _generate_puzzle, chat.id, language, theme
        )
    except Exception:  # noqa: BLE001
        logger.exception("Failed to generate puzzle for chat %s", chat.id)
        _cleanup_chat_resources(chat.id)
        await message.reply_text(
            "Сейчас не получилось подготовить кроссворд. Попробуйте выполнить /new чуть позже."
        )
        context.user_data.pop("new_game_language", None)
        return ConversationHandler.END

    context.user_data.pop("new_game_language", None)
    _cancel_reminder(context)

    image_path = None
    try:
        with logging_context(puzzle_id=puzzle.id):
            image_path = render_puzzle(puzzle, game_state)
            await context.bot.send_chat_action(
                chat_id=chat.id, action=constants.ChatAction.UPLOAD_PHOTO
            )
            with open(image_path, "rb") as photo:
                await message.reply_photo(
                    photo=photo,
                    caption=(
                        f"Кроссворд готов!\nЯзык: {puzzle.language.upper()}\nТема: {puzzle.theme}"
                    ),
                )
            await message.reply_text(_format_clues_message(puzzle))
            logger.info("Delivered freshly generated puzzle to chat")
    except Exception:  # noqa: BLE001
        logger.exception("Failed to deliver puzzle to chat %s", chat.id)
        _cleanup_game_state(game_state)
        if image_path is not None:
            with suppress(OSError):
                image_path.unlink(missing_ok=True)
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

    return ConversationHandler.END


@command_entrypoint(fallback=ConversationHandler.END)
async def cancel_new_game(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    _normalise_thread_id(update)
    context.user_data.pop("new_game_language", None)
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
        await message.reply_text(_format_clues_message(puzzle))


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
    logger.debug("Chat %s answering slot %s", chat.id, slot_id)
    raw_answer = " ".join(context.args[1:])

    game_state = _load_state_for_chat(chat.id)
    if not game_state:
        await message.reply_text("Нет активного кроссворда. Используйте /new.")
        return
    puzzle = _load_puzzle_for_state(game_state)
    if puzzle is None:
        await message.reply_text("Не удалось загрузить кроссворд. Попробуйте начать заново.")
        return
    with logging_context(puzzle_id=puzzle.id):
        slot = _find_slot(puzzle, slot_id)
        if slot is None:
            logger.warning("Answer received for missing slot %s", slot_id)
            await message.reply_text(f"Слот {slot_id} не найден.")
            return
        if slot.slot_id in game_state.solved_slots:
            await message.reply_text("Этот слот уже решён.")
            return
        if not slot.answer:
            await message.reply_text("Для этого слота не задан ответ.")
            return

        try:
            validated = validate_word_list(
                puzzle.language,
                [WordClue(word=raw_answer, clue="")],
                deduplicate=False,
            )
        except WordValidationError as exc:
            logger.warning("Rejected answer for slot %s due to validation: %s", slot.slot_id, exc)
            await message.reply_text(f"Слово не прошло проверку: {exc}")
            return
        except Exception as exc:  # noqa: BLE001
            logger.exception("Unexpected error validating answer for slot %s", slot.slot_id)
            await message.reply_text("Не удалось проверить слово. Попробуйте позже.")
            return

        if not validated:
            logger.info("Answer for slot %s failed language rules", slot.slot_id)
            await message.reply_text("Слово не соответствует правилам языка.")
            return

        candidate = validated[0].word
        if _canonical_answer(candidate, puzzle.language) != _canonical_answer(slot.answer, puzzle.language):
            logger.info("Incorrect answer for slot %s", slot.slot_id)
            await message.reply_text("Ответ неверный, попробуйте ещё раз.")
            return

        game_state.score += slot.length
        _apply_answer_to_state(game_state, slot, candidate)
        logger.info("Accepted answer for slot %s", slot.slot_id)

        try:
            image_path = render_puzzle(puzzle, game_state)
            await context.bot.send_chat_action(
                chat_id=chat.id, action=constants.ChatAction.UPLOAD_PHOTO
            )
            with open(image_path, "rb") as photo:
                await message.reply_photo(photo=photo, caption=f"Верно! {slot.slot_id}")
        except Exception:  # noqa: BLE001
            logger.exception("Failed to render updated grid after correct answer")
            await message.reply_text(
                "Ответ принят, но не удалось обновить изображение. Попробуйте команду /state позже."
            )

        if _all_slots_solved(puzzle, game_state):
            _cancel_reminder(context)
            await message.reply_text("Поздравляем! Все слова разгаданы.")


@command_entrypoint()
async def hint_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    _normalise_thread_id(update)
    if not await _reject_group_chat(update):
        return
    chat = update.effective_chat
    message = update.effective_message
    if chat is None or message is None:
        return
    logger.debug("Chat %s requested /hint", chat.id)

    game_state = _load_state_for_chat(chat.id)
    if not game_state:
        await message.reply_text("Нет активной игры. Используйте /new.")
        return
    puzzle = _load_puzzle_for_state(game_state)
    if puzzle is None:
        await message.reply_text("Не удалось загрузить кроссворд.")
        return

    with logging_context(puzzle_id=puzzle.id):
        slot: Optional[Slot] = None
        if context.args:
            slot = _find_slot(puzzle, context.args[0])
            if slot is None:
                await message.reply_text(f"Слот {context.args[0]} не найден.")
                return
        else:
            for candidate in puzzle.slots:
                if candidate.slot_id in game_state.solved_slots:
                    continue
                if not candidate.answer:
                    continue
                slot = candidate
                break
            if slot is None:
                await message.reply_text("Нет слотов для подсказки.")
                return

        if not slot.answer:
            await message.reply_text("Для этого слота нет ответа.")
            return

        result = _reveal_letter(game_state, slot, slot.answer)
        if result is None:
            game_state.hints_used += 1
            game_state.last_update = time.time()
            _store_state(game_state)
            reply_text = (
                f"Все буквы в {slot.slot_id} уже открыты. Подсказка: {slot.clue or 'нет'}"
            )
            logger.info("Hint requested for already revealed slot %s", slot.slot_id)
        else:
            position, letter = result
            reply_text = (
                f"Открыта буква №{position + 1} в {slot.slot_id}: {letter}\n"
                f"Подсказка: {slot.clue or 'нет'}"
            )
            logger.info("Revealed letter %s at position %s for slot %s", letter, position + 1, slot.slot_id)

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
        logger.info("Revealed remaining slots via /solve (%s entries)", len(solved_now))


def configure_telegram_handlers(telegram_application: Application) -> None:
    conversation = ConversationHandler(
        entry_points=[CommandHandler("new", start_new_game)],
        states={
            LANGUAGE_STATE: [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_language)],
            THEME_STATE: [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_theme)],
        },
        fallbacks=[CommandHandler("cancel", cancel_new_game)],
        name="new_game_conversation",
    )
    telegram_application.add_handler(conversation)
    telegram_application.add_handler(CommandHandler("clues", send_clues))
    telegram_application.add_handler(CommandHandler("state", send_state_image))
    telegram_application.add_handler(CommandHandler("answer", answer_command))
    telegram_application.add_handler(CommandHandler(["hint", "open"], hint_command))
    telegram_application.add_handler(CommandHandler("solve", solve_command))
    telegram_application.add_handler(CommandHandler("cancel", cancel_new_game))


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
            if info.url != expected_url or current_secret != settings.webhook_secret:
                logger.warning("Webhook mismatch detected. Expected url=%s secret token=%s", expected_url, settings.webhook_secret)
                await application.bot.set_webhook(
                    url=expected_url,
                    secret_token=settings.webhook_secret,
                    allowed_updates=[],
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
            expired = prune_expired_states(app_state.active_states)
            if expired:
                logger.info("Removed %s expired game states", len(expired))
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

    state.active_states = load_all_states()
    if state.active_states:
        logger.info("Restored %s active game states", len(state.active_states))
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

    state.telegram_app = telegram_application

    register_webhook_route(settings.webhook_path)

    expected_url = f"{settings.public_url}{settings.webhook_path}"
    logger.debug("Ensuring webhook is configured for %s", expected_url)
    await telegram_application.bot.set_webhook(
        url=expected_url,
        secret_token=settings.webhook_secret,
        allowed_updates=[],
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

        if state.telegram_app.is_running:
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
        allowed_updates=[],
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

