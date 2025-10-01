import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from telegram import InlineKeyboardMarkup
from telegram.constants import ChatType
from telegram.ext import ConversationHandler

import app
from app import (
    HINT_PENALTY,
    LANGUAGE_STATE,
    LOBBY_START_CALLBACK_PREFIX,
    LOBBY_WAIT_CALLBACK_PREFIX,
    MAX_LOBBY_PLAYERS,
    MENU_STATE,
    NEW_GAME_MODE_GROUP,
    NEW_GAME_MODE_SOLO,
    Settings,
    finish_command,
    hint_command,
    join_command,
    lobby_link_callback_handler,
    lobby_start_callback_handler,
    new_game_menu_admin_proxy_handler,
    new_game_menu_callback_handler,
    start_new_game,
    state,
)
from utils.crossword import Direction, Puzzle, Slot
from utils.storage import GameState, Player


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
def fresh_state():
    original_settings = state.settings
    state.active_games.clear()
    state.chat_to_game.clear()
    state.player_chats.clear()
    state.join_codes.clear()
    state.lobby_messages.clear()
    state.generating_chats.clear()
    state.scheduled_jobs.clear()
    state.chat_threads.clear()
    yield
    state.active_games.clear()
    state.chat_to_game.clear()
    state.player_chats.clear()
    state.join_codes.clear()
    state.lobby_messages.clear()
    state.generating_chats.clear()
    state.scheduled_jobs.clear()
    state.chat_threads.clear()
    state.settings = original_settings


class DummyJob:
    def __init__(self, chat_id: int, name: str) -> None:
        self.chat_id = chat_id
        self.name = name
        self.cancelled = False

    def schedule_removal(self) -> None:
        self.cancelled = True


class DummyJobQueue:
    def __init__(self) -> None:
        self.submitted: list[tuple] = []

    def run_once(self, callback, when, *, chat_id: int, name: str, data=None):  # noqa: ANN001 - signature mimics library
        job = DummyJob(chat_id, name)
        self.submitted.append((callback, when, chat_id, name, data))
        return job


def _make_turn_puzzle() -> Puzzle:
    puzzle = Puzzle.from_size("puzzle", "Тема", "ru", 5, 5)
    puzzle.slots = [
        Slot(
            slot_id="A1",
            direction=Direction.ACROSS,
            number=1,
            start_row=0,
            start_col=0,
            length=3,
            clue="Столица Италии",
            answer="рим",
        ),
        Slot(
            slot_id="D1",
            direction=Direction.DOWN,
            number=1,
            start_row=0,
            start_col=0,
            length=3,
            clue="Река",
            answer="дон",
        ),
    ]
    return puzzle


def _make_turn_state(chat_id: int, puzzle: Puzzle) -> GameState:
    now = time.time()
    player_one = Player(user_id=1, name="Игрок 1", dm_chat_id=101)
    player_two = Player(user_id=2, name="Игрок 2", dm_chat_id=102)
    return GameState(
        chat_id=chat_id,
        puzzle_id=puzzle.id,
        filled_cells={},
        solved_slots=set(),
        score=0,
        started_at=now,
        last_update=now,
        hinted_cells=set(),
        host_id=1,
        game_id=str(chat_id),
        scoreboard={1: 0, 2: 0},
        mode="turn_based",
        status="running",
        players={1: player_one, 2: player_two},
        turn_order=[1, 2],
        turn_index=0,
        active_slot_id="A1",
    )


def _make_group_update(chat_id: int, user_id: int):
    chat = SimpleNamespace(id=chat_id, type=ChatType.GROUP)
    message = SimpleNamespace(
        message_thread_id=None,
        message_id=42,
        reply_text=AsyncMock(),
        reply_photo=AsyncMock(),
        from_user=SimpleNamespace(id=user_id),
    )
    update = SimpleNamespace(
        effective_chat=chat,
        effective_message=message,
        effective_user=message.from_user,
        callback_query=None,
    )
    return update, chat, message


@pytest.mark.anyio
async def test_start_new_group_game_creates_lobby(monkeypatch, fresh_state):
    chat_id = -1001
    host_user = SimpleNamespace(id=10, full_name="Ведущий", username="host")
    chat = SimpleNamespace(id=chat_id, type=ChatType.GROUP)
    message = SimpleNamespace(message_thread_id=None, reply_text=AsyncMock())
    update = SimpleNamespace(
        effective_chat=chat,
        effective_message=message,
        effective_user=host_user,
    )
    context = SimpleNamespace(chat_data={}, user_data={}, bot=SimpleNamespace())

    monkeypatch.setattr(app, "_load_state_for_chat", lambda _: None)
    cleanup_called = []
    monkeypatch.setattr(app, "_cleanup_game_state", lambda *_: cleanup_called.append(True))
    stored_states: list[GameState] = []

    def fake_store(game_state: GameState) -> None:
        stored_states.append(game_state)
        state.active_games[game_state.game_id] = game_state
        state.chat_to_game[game_state.chat_id] = game_state.game_id

    monkeypatch.setattr(app, "_store_state", fake_store)

    result = await app._start_new_group_game(update, context)

    assert result == LANGUAGE_STATE
    assert stored_states, "Game state should be stored"
    game_state = stored_states[-1]
    assert game_state.status == "lobby"
    assert game_state.mode == "turn_based"
    assert list(game_state.players) == [host_user.id]
    assert game_state.scoreboard.get(host_user.id) == 0
    assert context.chat_data.get("new_game_language") is None
    message.reply_text.assert_awaited()


def test_assign_join_code_avoids_collisions(monkeypatch, fresh_state):
    game_state = GameState(
        chat_id=-200,
        puzzle_id="p",
        filled_cells={},
        solved_slots=set(),
        score=0,
        started_at=time.time(),
        last_update=time.time(),
        scoreboard={},
        players={},
        status="lobby",
        mode="turn_based",
    )
    state.join_codes["ABC123"] = "other"
    sequence = iter("ABC123DEF456")
    monkeypatch.setattr(app.secrets, "choice", lambda _: next(sequence))

    code = app._assign_join_code(game_state)

    assert code == "DEF456"
    assert game_state.join_codes[code] == game_state.game_id


@pytest.mark.anyio
async def test_join_command_invokes_process(monkeypatch, fresh_state):
    process_mock = AsyncMock()
    monkeypatch.setattr(app, "_process_join_code", process_mock)
    chat = SimpleNamespace(id=123, type=ChatType.PRIVATE)
    message = SimpleNamespace(reply_text=AsyncMock(), message_thread_id=None)
    user = SimpleNamespace(id=55)
    update = SimpleNamespace(
        effective_chat=chat,
        effective_message=message,
        effective_user=user,
    )
    context = SimpleNamespace(args=["XYZ789"], bot=SimpleNamespace(), user_data={}, chat_data={})

    await join_command(update, context)

    process_mock.assert_awaited_once_with(update, context, "XYZ789")


@pytest.mark.anyio
async def test_start_command_with_join_code(monkeypatch, fresh_state):
    process_mock = AsyncMock()
    monkeypatch.setattr(app, "_process_join_code", process_mock)
    chat = SimpleNamespace(id=999, type=ChatType.PRIVATE)
    message = SimpleNamespace(message_thread_id=None)
    user = SimpleNamespace(id=77)
    update = SimpleNamespace(
        effective_chat=chat,
        effective_message=message,
        effective_user=user,
    )
    context = SimpleNamespace(args=["join_ABCDEF"], bot=SimpleNamespace(), chat_data={}, user_data={})

    result = await start_new_game(update, context)

    assert result == ConversationHandler.END
    process_mock.assert_awaited_once_with(update, context, "ABCDEF")


@pytest.mark.anyio
async def test_start_new_game_shows_menu_private(monkeypatch, fresh_state):
    chat = SimpleNamespace(id=101, type=ChatType.PRIVATE)
    message = SimpleNamespace(message_thread_id=None, reply_text=AsyncMock())
    user = SimpleNamespace(id=55)
    update = SimpleNamespace(
        effective_chat=chat,
        effective_message=message,
        effective_user=user,
    )
    context = SimpleNamespace(args=[], chat_data={}, user_data={}, bot=SimpleNamespace())

    result = await start_new_game(update, context)

    assert result == MENU_STATE
    message.reply_text.assert_awaited()
    call = message.reply_text.await_args
    assert "Выберите, как хотите играть" in call.args[0]
    markup = call.kwargs["reply_markup"]
    assert isinstance(markup, InlineKeyboardMarkup)
    assert markup.inline_keyboard[0][0].callback_data == NEW_GAME_MODE_SOLO
    assert markup.inline_keyboard[1][0].callback_data == NEW_GAME_MODE_GROUP


@pytest.mark.anyio
async def test_start_new_game_adds_admin_button_for_admin(monkeypatch, fresh_state):
    chat = SimpleNamespace(id=303, type=ChatType.PRIVATE)
    message = SimpleNamespace(message_thread_id=None, reply_text=AsyncMock())
    user = SimpleNamespace(id=999)
    update = SimpleNamespace(
        effective_chat=chat,
        effective_message=message,
        effective_user=user,
    )
    state.settings = Settings(
        telegram_bot_token="token",
        public_url="https://example.com",
        webhook_secret="secret",
        admin_id=user.id,
    )
    context = SimpleNamespace(args=[], chat_data={}, user_data={}, bot=SimpleNamespace())

    result = await start_new_game(update, context)

    assert result == MENU_STATE
    call = message.reply_text.await_args
    markup = call.kwargs["reply_markup"]
    assert isinstance(markup, InlineKeyboardMarkup)
    assert len(markup.inline_keyboard) == 3
    assert (
        markup.inline_keyboard[-1][0].callback_data
        == f"{app.ADMIN_TEST_GAME_CALLBACK_PREFIX}{chat.id}"
    )
    assert "[адм.] Тестовая сессия" in call.args[0]


@pytest.mark.anyio
async def test_new_game_menu_solo_starts_private_flow(monkeypatch, fresh_state):
    chat = SimpleNamespace(id=404, type=ChatType.PRIVATE)
    message = SimpleNamespace(message_thread_id=None)
    job = DummyJob(chat.id, "reminder")
    context = SimpleNamespace(
        chat_data={
            "reminder_job": job,
            app.GENERATION_NOTICE_KEY: {"active": True},
            "lobby_message_id": 123,
        },
        user_data={
            app.BUTTON_NEW_GAME_KEY: {"step": "language"},
            "new_game_language": "ru",
            "pending_join": object(),
        },
        bot=SimpleNamespace(),
    )
    query = SimpleNamespace(
        data=NEW_GAME_MODE_SOLO,
        answer=AsyncMock(),
        message=message,
    )
    update = SimpleNamespace(
        effective_chat=chat,
        effective_message=message,
        effective_user=SimpleNamespace(id=1),
        callback_query=query,
    )
    private_mock = AsyncMock(return_value=LANGUAGE_STATE)
    monkeypatch.setattr(app, "_start_new_private_game", private_mock)

    result = await new_game_menu_callback_handler(update, context)

    assert result == LANGUAGE_STATE
    private_mock.assert_awaited_once_with(update, context)
    query.answer.assert_awaited_once()
    assert job.cancelled is True
    assert "reminder_job" not in context.chat_data
    assert app.BUTTON_NEW_GAME_KEY not in context.user_data
    assert app.GENERATION_NOTICE_KEY not in context.chat_data
    assert "lobby_message_id" not in context.chat_data
    assert "new_game_language" not in context.user_data
    assert "pending_join" not in context.user_data


@pytest.mark.anyio
async def test_new_game_menu_group_starts_group_flow(monkeypatch, fresh_state):
    chat = SimpleNamespace(id=-505, type=ChatType.GROUP)
    message = SimpleNamespace(message_thread_id=None)
    job = DummyJob(chat.id, "reminder")
    context = SimpleNamespace(
        chat_data={
            "reminder_job": job,
            app.GENERATION_NOTICE_KEY: {"active": True},
            app.BUTTON_NEW_GAME_KEY: {"step": "language"},
        },
        user_data={"pending_join": object()},
        bot=SimpleNamespace(),
    )
    query = SimpleNamespace(
        data=NEW_GAME_MODE_GROUP,
        answer=AsyncMock(),
        message=message,
    )
    update = SimpleNamespace(
        effective_chat=chat,
        effective_message=message,
        effective_user=SimpleNamespace(id=2),
        callback_query=query,
    )
    group_mock = AsyncMock(return_value=ConversationHandler.END)
    monkeypatch.setattr(app, "_start_new_group_game", group_mock)

    result = await new_game_menu_callback_handler(update, context)

    assert result == ConversationHandler.END
    group_mock.assert_awaited_once_with(update, context)
    query.answer.assert_awaited_once()
    assert job.cancelled is True
    assert "reminder_job" not in context.chat_data
    assert app.BUTTON_NEW_GAME_KEY not in context.chat_data
    assert app.GENERATION_NOTICE_KEY not in context.chat_data
    assert "pending_join" not in context.user_data


@pytest.mark.anyio
async def test_new_game_menu_group_requires_group_chat(monkeypatch, fresh_state):
    chat = SimpleNamespace(id=808, type=ChatType.PRIVATE)
    message = SimpleNamespace(message_thread_id=None)
    context = SimpleNamespace(chat_data={}, user_data={}, bot=SimpleNamespace())
    query = SimpleNamespace(
        data=NEW_GAME_MODE_GROUP,
        answer=AsyncMock(),
        message=message,
    )
    update = SimpleNamespace(
        effective_chat=chat,
        effective_message=message,
        effective_user=SimpleNamespace(id=3),
        callback_query=query,
    )
    group_mock = AsyncMock()
    monkeypatch.setattr(app, "_start_new_group_game", group_mock)

    result = await new_game_menu_callback_handler(update, context)

    assert result == MENU_STATE
    group_mock.assert_not_called()
    query.answer.assert_awaited_once()
    call = query.answer.await_args
    assert call.kwargs.get("show_alert") is True


@pytest.mark.anyio
async def test_new_game_menu_admin_proxy_clears_state(monkeypatch, fresh_state):
    chat = SimpleNamespace(id=909, type=ChatType.PRIVATE)
    message = SimpleNamespace(message_thread_id=None)
    job = DummyJob(chat.id, "reminder")
    context = SimpleNamespace(
        chat_data={
            "reminder_job": job,
            app.GENERATION_NOTICE_KEY: {"active": True},
        },
        user_data={
            app.BUTTON_NEW_GAME_KEY: {"step": "language"},
            "pending_join": object(),
            "new_game_language": "en",
        },
        bot=SimpleNamespace(),
    )
    query = SimpleNamespace(
        data=f"{app.ADMIN_TEST_GAME_CALLBACK_PREFIX}{chat.id}",
        answer=AsyncMock(),
        message=message,
    )
    update = SimpleNamespace(
        effective_chat=chat,
        effective_message=message,
        effective_user=SimpleNamespace(id=4),
        callback_query=query,
    )

    result = await new_game_menu_admin_proxy_handler(update, context)

    assert result == ConversationHandler.END
    assert job.cancelled is True
    assert "reminder_job" not in context.chat_data
    assert app.BUTTON_NEW_GAME_KEY not in context.user_data
    assert app.GENERATION_NOTICE_KEY not in context.chat_data
    assert "pending_join" not in context.user_data
    assert "new_game_language" not in context.user_data


@pytest.mark.anyio
async def test_admin_proxy_restarts_after_finished_game(monkeypatch, fresh_state):
    chat = SimpleNamespace(id=1111, type=ChatType.PRIVATE)
    message = SimpleNamespace(message_thread_id=None, reply_text=AsyncMock())
    query = SimpleNamespace(
        data=f"{app.ADMIN_TEST_GAME_CALLBACK_PREFIX}{chat.id}",
        answer=AsyncMock(),
        message=message,
    )
    update = SimpleNamespace(
        effective_chat=chat,
        effective_message=message,
        callback_query=query,
    )
    context = SimpleNamespace(chat_data={}, user_data={}, bot=SimpleNamespace())

    puzzle = _make_turn_puzzle()
    base_state = _make_turn_state(chat.id, puzzle)
    base_state.status = "finished"
    state.active_games[base_state.game_id] = base_state
    state.chat_to_game[chat.id] = base_state.game_id

    start_group_mock = AsyncMock(return_value=LANGUAGE_STATE)
    monkeypatch.setattr(app, "_start_new_group_game", start_group_mock)

    result = await new_game_menu_admin_proxy_handler(update, context)

    assert result == LANGUAGE_STATE
    start_group_mock.assert_awaited_once_with(update, context)
    assert context.chat_data.get(app.PENDING_ADMIN_TEST_KEY) == chat.id
    assert base_state.game_id not in state.active_games
    assert chat.id not in state.chat_to_game
    query.answer.assert_awaited_once()


def test_lobby_keyboard_start_activation(fresh_state):
    puzzle = _make_turn_puzzle()
    game_state = _make_turn_state(-400, puzzle)
    game_state.status = "lobby"
    game_state.players = {1: game_state.players[1]}
    game_state.scoreboard = {1: 0}
    game_state.turn_order = []

    keyboard = app._build_lobby_keyboard(game_state)
    start_data = keyboard.inline_keyboard[-1][0].callback_data
    assert start_data.startswith(LOBBY_WAIT_CALLBACK_PREFIX)

    # Add second player to enable start
    new_player = Player(user_id=3, name="Игрок 3")
    game_state.players[new_player.user_id] = new_player
    game_state.scoreboard[new_player.user_id] = 0

    keyboard = app._build_lobby_keyboard(game_state)
    start_data = keyboard.inline_keyboard[-1][0].callback_data
    assert start_data.startswith(LOBBY_START_CALLBACK_PREFIX)


@pytest.mark.anyio
async def test_join_code_limits_players(monkeypatch, fresh_state):
    puzzle = _make_turn_puzzle()
    game_state = _make_turn_state(-500, puzzle)
    game_state.status = "lobby"
    game_state.players = {
        idx: Player(user_id=idx, name=f"P{idx}") for idx in range(1, MAX_LOBBY_PLAYERS + 1)
    }
    state.join_codes["ROOM01"] = game_state.game_id

    def load_by_game_id(game_id: str):
        return game_state if game_id == game_state.game_id else None

    monkeypatch.setattr(app, "_load_state_by_game_id", load_by_game_id)

    chat = SimpleNamespace(id=700, type=ChatType.PRIVATE)
    message = SimpleNamespace(reply_text=AsyncMock(), message_thread_id=None)
    user = SimpleNamespace(id=999)
    update = SimpleNamespace(
        effective_chat=chat,
        effective_message=message,
        effective_user=user,
    )
    context = SimpleNamespace(bot=SimpleNamespace(send_message=AsyncMock()), user_data={}, chat_data={})

    await app._process_join_code(update, context, "room01")

    message.reply_text.assert_awaited()
    assert "лимит" in message.reply_text.await_args.args[0].lower()


@pytest.mark.anyio
async def test_lobby_start_callback_starts_game(monkeypatch, fresh_state):
    puzzle = _make_turn_puzzle()
    game_state = _make_turn_state(-550, puzzle)
    game_state.status = "lobby"
    state.active_games[game_state.game_id] = game_state
    state.chat_to_game[game_state.chat_id] = game_state.game_id

    monkeypatch.setattr(app, "_load_state_by_game_id", lambda _: game_state)
    monkeypatch.setattr(app, "_load_puzzle_for_state", lambda _: puzzle)
    schedule_mock = MagicMock()
    monkeypatch.setattr(app, "_schedule_game_timers", schedule_mock)

    stored_states: list[GameState] = []

    def fake_store(gs: GameState) -> None:
        stored_states.append(gs)
        state.active_games[gs.game_id] = gs

    monkeypatch.setattr(app, "_store_state", fake_store)
    announce_mock = AsyncMock()
    monkeypatch.setattr(app, "_announce_turn", announce_mock)

    query_message = SimpleNamespace(message_thread_id=None)
    query = SimpleNamespace(
        data=f"{LOBBY_START_CALLBACK_PREFIX}{game_state.game_id}",
        answer=AsyncMock(),
        message=query_message,
    )
    effective_chat = SimpleNamespace(id=game_state.chat_id, type=ChatType.GROUP)
    update = SimpleNamespace(
        callback_query=query,
        effective_user=SimpleNamespace(id=1),
        effective_chat=effective_chat,
        effective_message=query_message,
    )
    context = SimpleNamespace(bot=SimpleNamespace(send_message=AsyncMock()), job_queue=DummyJobQueue())

    await lobby_start_callback_handler(update, context)

    assert game_state.status == "running"
    assert game_state.turn_order == [1, 2]
    assert all(score == 0 for score in game_state.scoreboard.values())
    schedule_mock.assert_called_once()
    announce_mock.assert_awaited()
    query.answer.assert_awaited_with("Игра начинается!")


@pytest.mark.anyio
async def test_lobby_link_callback_sends_code(monkeypatch, fresh_state):
    puzzle = _make_turn_puzzle()
    game_state = _make_turn_state(-560, puzzle)
    game_state.status = "lobby"
    state.active_games[game_state.game_id] = game_state
    state.chat_to_game[game_state.chat_id] = game_state.game_id

    monkeypatch.setattr(app, "_load_state_by_game_id", lambda _: game_state)
    def fake_assign(gs: GameState) -> str:
        gs.join_codes["ABCDEF"] = gs.game_id
        return "ABCDEF"

    assign_mock = MagicMock(side_effect=fake_assign)
    monkeypatch.setattr(app, "_assign_join_code", assign_mock)
    store_mock = MagicMock()
    monkeypatch.setattr(app, "_store_state", store_mock)
    monkeypatch.setattr(app, "_build_join_link", AsyncMock(return_value="https://t.me/bot?start=join_ABCDEF"))

    state.player_chats[1] = 500
    query_message = SimpleNamespace(
        chat=SimpleNamespace(id=game_state.chat_id, type=ChatType.GROUP),
        message_thread_id=None,
    )
    query = SimpleNamespace(
        data=f"{app.LOBBY_LINK_CALLBACK_PREFIX}{game_state.game_id}",
        answer=AsyncMock(),
        message=query_message,
    )
    update = SimpleNamespace(
        callback_query=query,
        effective_user=SimpleNamespace(id=1),
        effective_chat=query_message.chat,
        effective_message=query_message,
    )
    context = SimpleNamespace(bot=SimpleNamespace(send_message=AsyncMock()))

    await lobby_link_callback_handler(update, context)

    assign_mock.assert_called_once_with(game_state)
    store_mock.assert_called_once_with(game_state)
    context.bot.send_message.assert_awaited()
    query.answer.assert_awaited()


@pytest.mark.anyio
async def test_turn_based_rejects_non_current_answer(monkeypatch, fresh_state):
    puzzle = _make_turn_puzzle()
    game_state = _make_turn_state(-600, puzzle)
    state.active_games[game_state.game_id] = game_state
    state.chat_to_game[game_state.chat_id] = game_state.game_id

    monkeypatch.setattr(app, "_load_state_for_chat", lambda _: game_state)
    monkeypatch.setattr(app, "_load_puzzle_for_state", lambda _: puzzle)
    monkeypatch.setattr(app, "_store_state", lambda _s: None)
    monkeypatch.setattr(app, "_send_clues_update", AsyncMock())

    update, chat, message = _make_group_update(game_state.chat_id, user_id=2)
    context = SimpleNamespace(
        bot=SimpleNamespace(send_chat_action=AsyncMock(), send_message=AsyncMock()),
        job_queue=DummyJobQueue(),
    )

    await app._handle_answer_submission(context, chat, message, "A1", "рим")

    message.reply_text.assert_awaited()
    assert "ход" in message.reply_text.await_args.args[0]
    assert game_state.scoreboard[1] == 0
    assert game_state.turn_index == 0
    message.reply_photo.assert_not_awaited()


@pytest.mark.anyio
async def test_turn_based_answer_advances_turn(monkeypatch, tmp_path, fresh_state):
    puzzle = _make_turn_puzzle()
    game_state = _make_turn_state(-700, puzzle)
    state.active_games[game_state.game_id] = game_state
    state.chat_to_game[game_state.chat_id] = game_state.game_id

    monkeypatch.setattr(app, "_load_state_for_chat", lambda _: game_state)
    monkeypatch.setattr(app, "_load_puzzle_for_state", lambda _: puzzle)

    stored_states: list[GameState] = []

    def fake_store(gs: GameState) -> None:
        stored_states.append(gs)
        state.active_games[gs.game_id] = gs
        state.chat_to_game[gs.chat_id] = gs.game_id

    monkeypatch.setattr(app, "_store_state", fake_store)
    monkeypatch.setattr(app, "_send_clues_update", AsyncMock())

    image_path = tmp_path / "grid.png"
    image_path.write_bytes(b"png")
    monkeypatch.setattr(app, "render_puzzle", lambda _p, _s: str(image_path))

    job_queue = DummyJobQueue()
    bot = SimpleNamespace(send_chat_action=AsyncMock(), send_message=AsyncMock())
    context = SimpleNamespace(bot=bot, job_queue=job_queue)

    game_state.thread_id = 321
    await app._announce_turn(context, game_state, puzzle)
    group_call = next(
        call
        for call in bot.send_message.await_args_list
        if call.kwargs.get("chat_id") == game_state.chat_id
    )
    assert group_call.kwargs.get("message_thread_id") == 321
    bot.send_message.reset_mock()
    assert len(job_queue.submitted) >= 1
    initial_warn_name = game_state.turn_warn_job_id
    initial_timeout_name = game_state.turn_timer_job_id
    initial_warn_job = state.scheduled_jobs.get(initial_warn_name)
    initial_timeout_job = state.scheduled_jobs.get(initial_timeout_name)

    update, chat, message = _make_group_update(game_state.chat_id, user_id=1)
    message.reply_photo = AsyncMock()

    await app._handle_answer_submission(context, chat, message, "A1", "рим")

    assert game_state.scoreboard[1] == app.SCORE_PER_WORD
    assert game_state.score == app.SCORE_PER_WORD
    assert game_state.turn_index == 1
    assert game_state.players[1].answers_ok == 1
    assert initial_warn_job.cancelled is True
    assert initial_timeout_job.cancelled is True
    assert game_state.turn_timer_job_id in state.scheduled_jobs
    assert state.scheduled_jobs[game_state.turn_timer_job_id] is not initial_timeout_job
    assert message.reply_photo.await_count == 1
    bot.send_message.assert_awaited()


@pytest.mark.anyio
async def test_dm_only_game_notifications_send_once(monkeypatch, fresh_state):
    puzzle = _make_turn_puzzle()
    chat_id = 555
    game_state = _make_turn_state(chat_id, puzzle)
    for player in game_state.players.values():
        player.dm_chat_id = chat_id
    state.active_games[game_state.game_id] = game_state
    state.chat_to_game[chat_id] = game_state.game_id

    monkeypatch.setattr(app, "_store_state", lambda _gs: None)
    monkeypatch.setattr(
        app, "_load_state_by_game_id", lambda gid: game_state if gid == game_state.game_id else None
    )
    monkeypatch.setattr(app, "_load_puzzle_for_state", lambda _gs: puzzle)

    announce_mock = AsyncMock()
    context = SimpleNamespace(
        bot=SimpleNamespace(send_message=announce_mock), job_queue=DummyJobQueue()
    )

    await app._announce_turn(context, game_state, puzzle)

    assert announce_mock.await_count == 1
    assert announce_mock.await_args.kwargs["chat_id"] == chat_id
    assert "message_thread_id" not in announce_mock.await_args.kwargs

    warning_mock = AsyncMock()
    job_name = "turn-warn-test"
    state.scheduled_jobs[job_name] = object()
    job = SimpleNamespace(
        name=job_name,
        data={"game_id": game_state.game_id, "player_id": game_state.turn_order[game_state.turn_index]},
    )
    warning_context = SimpleNamespace(bot=SimpleNamespace(send_message=warning_mock), job=job)

    await app._turn_warning_job(warning_context)

    assert warning_mock.await_count == 1
    assert warning_mock.await_args.kwargs["chat_id"] == chat_id

    slot_mock = AsyncMock()
    query = SimpleNamespace(
        data=f"{app.TURN_SLOT_CALLBACK_PREFIX}{game_state.game_id}|A1",
        answer=AsyncMock(),
        from_user=SimpleNamespace(id=game_state.turn_order[game_state.turn_index]),
        message=SimpleNamespace(chat=SimpleNamespace(id=chat_id, type=ChatType.PRIVATE)),
    )
    update = SimpleNamespace(
        callback_query=query,
        effective_chat=SimpleNamespace(id=chat_id, type=ChatType.PRIVATE),
        effective_message=SimpleNamespace(message_thread_id=None),
        effective_user=query.from_user,
    )
    slot_context = SimpleNamespace(bot=SimpleNamespace(send_message=slot_mock))

    await app.turn_slot_callback_handler(update, slot_context)

    assert slot_mock.await_count == 1
    assert slot_mock.await_args.kwargs["chat_id"] == chat_id


@pytest.mark.anyio
async def test_turn_based_hint_limits_players(monkeypatch, fresh_state):
    puzzle = _make_turn_puzzle()
    game_state = _make_turn_state(-800, puzzle)
    state.active_games[game_state.game_id] = game_state
    state.chat_to_game[game_state.chat_id] = game_state.game_id

    monkeypatch.setattr(app, "_load_state_for_chat", lambda _: game_state)
    monkeypatch.setattr(app, "_load_puzzle_for_state", lambda _: puzzle)

    update, chat, message = _make_group_update(game_state.chat_id, user_id=2)
    context = SimpleNamespace(bot=SimpleNamespace(send_chat_action=AsyncMock()), args=[], job_queue=DummyJobQueue())

    await hint_command(update, context)

    message.reply_text.assert_awaited()
    assert "ход" in message.reply_text.await_args.args[0]
    assert game_state.scoreboard[1] == 0


@pytest.mark.anyio
async def test_turn_based_hint_penalises_current_player(monkeypatch, tmp_path, fresh_state):
    puzzle = _make_turn_puzzle()
    game_state = _make_turn_state(-900, puzzle)
    state.active_games[game_state.game_id] = game_state
    state.chat_to_game[game_state.chat_id] = game_state.game_id

    monkeypatch.setattr(app, "_load_state_for_chat", lambda _: game_state)
    monkeypatch.setattr(app, "_load_puzzle_for_state", lambda _: puzzle)

    image_path = tmp_path / "hint.png"
    image_path.write_bytes(b"png")
    monkeypatch.setattr(app, "render_puzzle", lambda _p, _s: str(image_path))

    update, chat, message = _make_group_update(game_state.chat_id, user_id=1)
    context = SimpleNamespace(bot=SimpleNamespace(send_chat_action=AsyncMock()), args=[], job_queue=DummyJobQueue())

    await hint_command(update, context)

    assert game_state.scoreboard[1] == -HINT_PENALTY
    assert game_state.score == -HINT_PENALTY
    assert game_state.hints_used
    message.reply_photo.assert_awaited()


@pytest.mark.anyio
async def test_finish_command_triggers_finish(monkeypatch, fresh_state):
    puzzle = _make_turn_puzzle()
    game_state = _make_turn_state(-910, puzzle)
    state.active_games[game_state.game_id] = game_state
    state.chat_to_game[game_state.chat_id] = game_state.game_id

    monkeypatch.setattr(app, "_load_state_for_chat", lambda _: game_state)
    finish_mock = AsyncMock()
    monkeypatch.setattr(app, "_finish_game", finish_mock)

    update, chat, message = _make_group_update(game_state.chat_id, user_id=1)
    context = SimpleNamespace(bot=SimpleNamespace(), args=[], job_queue=DummyJobQueue())

    await finish_command(update, context)

    finish_mock.assert_awaited_once()
    assert (
        finish_mock.await_args.kwargs.get("reason")
        == "Игроки решили завершить игру. 🤝"
    )


@pytest.mark.anyio
async def test_auto_finish_after_last_slot(monkeypatch, tmp_path, fresh_state):
    puzzle = _make_turn_puzzle()
    puzzle.slots = puzzle.slots[:1]
    game_state = _make_turn_state(-920, puzzle)
    game_state.turn_order = [1]
    game_state.players = {1: game_state.players[1]}
    game_state.scoreboard = {1: 0}
    state.active_games[game_state.game_id] = game_state
    state.chat_to_game[game_state.chat_id] = game_state.game_id

    monkeypatch.setattr(app, "_load_state_for_chat", lambda _: game_state)
    monkeypatch.setattr(app, "_load_puzzle_for_state", lambda _: puzzle)
    finish_mock = AsyncMock()
    monkeypatch.setattr(app, "_finish_game", finish_mock)
    monkeypatch.setattr(app, "_send_clues_update", AsyncMock())

    image_path = tmp_path / "last.png"
    image_path.write_bytes(b"png")
    monkeypatch.setattr(app, "render_puzzle", lambda _p, _s: str(image_path))

    job_queue = DummyJobQueue()
    bot = SimpleNamespace(send_chat_action=AsyncMock(), send_message=AsyncMock())
    context = SimpleNamespace(bot=bot, job_queue=job_queue)

    update, chat, message = _make_group_update(game_state.chat_id, user_id=1)
    message.reply_photo = AsyncMock()

    await app._handle_answer_submission(context, chat, message, "A1", "рим")

    finish_mock.assert_awaited_once()
    assert "последний" in finish_mock.await_args.kwargs.get("reason", "")


def test_format_leaderboard_orders_players(fresh_state):
    puzzle = _make_turn_puzzle()
    game_state = _make_turn_state(-930, puzzle)
    game_state.scoreboard = {1: 3, 2: 5, 3: 5}
    extra = Player(user_id=3, name="Игрок 3")
    game_state.players[3] = extra
    game_state.players[1].answers_ok = 2
    game_state.players[2].answers_ok = 2
    extra.answers_ok = 3
    app._record_hint_usage(game_state, "A1", user_id=2)
    app._record_hint_usage(game_state, "A1", user_id=3)
    app._record_hint_usage(game_state, "A1", user_id=3)

    text = app._format_leaderboard(game_state)

    lines = text.split("<br/>")
    assert lines[0].startswith("1. <b>Игрок 3")
    assert lines[1].startswith("2. <b>Игрок 2")
    assert "💡 2" in lines[0]


@pytest.mark.anyio
async def test_admin_menu_requires_admin(monkeypatch, fresh_state):
    state.settings = SimpleNamespace(admin_id=500)
    chat = SimpleNamespace(id=50, type=ChatType.PRIVATE)
    non_admin = SimpleNamespace(id=123)
    message = SimpleNamespace(reply_text=AsyncMock(), message_thread_id=None)
    update = SimpleNamespace(
        effective_chat=chat,
        effective_message=message,
        effective_user=non_admin,
    )
    context = SimpleNamespace(bot=SimpleNamespace())

    await app.admin_menu_command(update, context)
    message.reply_text.assert_not_awaited()

    admin = SimpleNamespace(id=500)
    update.effective_user = admin
    await app.admin_menu_command(update, context)
    message.reply_text.assert_awaited()


@pytest.mark.anyio
async def test_admin_test_game_creates_room(monkeypatch, fresh_state):
    state.settings = SimpleNamespace(admin_id=700)
    puzzle = _make_turn_puzzle()
    base_state = _make_turn_state(-940, puzzle)
    base_state.status = "running"
    state.active_games[base_state.game_id] = base_state
    state.chat_to_game[base_state.chat_id] = base_state.game_id
    state.lobby_messages[base_state.game_id] = (base_state.chat_id, 111)

    monkeypatch.setattr(app, "_load_state_for_chat", lambda _: base_state)
    monkeypatch.setattr(app, "_load_puzzle_for_state", lambda _: puzzle)
    monkeypatch.setattr(app, "_clone_puzzle_for_test", lambda _p: (puzzle, "clone", None))

    existing_admin_state = GameState(
        chat_id=base_state.chat_id,
        puzzle_id="old",
        filled_cells={},
        solved_slots=set(),
        score=0,
        started_at=time.time(),
        last_update=time.time(),
        scoreboard={},
    )

    def fake_load_by_id(game_id: str):
        if game_id == "admin:-940":
            return existing_admin_state
        return None

    cleanup_mock = MagicMock()
    monkeypatch.setattr(app, "_cleanup_game_state", cleanup_mock)
    monkeypatch.setattr(app, "_load_state_by_game_id", fake_load_by_id)

    stored_states: list[GameState] = []

    def fake_store(gs: GameState) -> None:
        stored_states.append(gs)
        state.active_games[gs.game_id] = gs
        state.chat_to_game[gs.chat_id] = gs.game_id

    monkeypatch.setattr(app, "_store_state", fake_store)

    job_queue = DummyJobQueue()
    bot = SimpleNamespace(send_message=AsyncMock())
    context = SimpleNamespace(bot=bot, job_queue=job_queue)
    query_message = SimpleNamespace(
        chat=SimpleNamespace(id=base_state.chat_id, type=ChatType.GROUP),
        message_thread_id=None,
        reply_text=AsyncMock(),
    )
    query = SimpleNamespace(
        data=f"{app.ADMIN_TEST_GAME_CALLBACK_PREFIX}{base_state.chat_id}",
        answer=AsyncMock(),
        message=query_message,
    )
    update = SimpleNamespace(
        callback_query=query,
        effective_user=SimpleNamespace(id=700, full_name="Админ", username="admin"),
        effective_chat=query_message.chat,
        effective_message=query_message,
    )

    await app.admin_test_game_callback_handler(update, context)

    cleanup_mock.assert_called_once_with(existing_admin_state)
    assert base_state.game_id not in state.lobby_messages
    assert any(gs.game_id.startswith("admin:") for gs in stored_states)
    admin_state = next(gs for gs in stored_states if gs.game_id.startswith("admin:"))
    assert admin_state.test_mode is True
    assert admin_state.dummy_user_id == app.DUMMY_USER_ID
    assert admin_state.players.get(app.DUMMY_USER_ID)
    query.answer.assert_awaited()


@pytest.mark.anyio
async def test_admin_test_game_recovers_from_stale_mapping(monkeypatch, fresh_state):
    state.settings = SimpleNamespace(admin_id=700)
    puzzle = _make_turn_puzzle()
    base_state = _make_turn_state(-941, puzzle)
    base_state.status = "lobby"
    stale_game_id = f"admin:{base_state.chat_id}"
    state.chat_to_game[base_state.chat_id] = stale_game_id

    load_calls: list[str | int] = []

    def fake_load(identifier):  # noqa: ANN001 - mimics utils.storage.load_state
        load_calls.append(identifier)
        if identifier == stale_game_id:
            return None
        if identifier == base_state.chat_id or str(identifier) == str(base_state.chat_id):
            return base_state
        return None

    monkeypatch.setattr(app, "load_state", fake_load)
    monkeypatch.setattr(app, "_load_puzzle_for_state", lambda _: puzzle)
    monkeypatch.setattr(app, "_clone_puzzle_for_test", lambda _p: (puzzle, "clone", None))
    monkeypatch.setattr(app, "_load_state_by_game_id", lambda _: None)
    cleanup_mock = MagicMock()
    monkeypatch.setattr(app, "_cleanup_game_state", cleanup_mock)
    monkeypatch.setattr(app, "_schedule_game_timers", lambda *args, **kwargs: None)
    announce_mock = AsyncMock()
    monkeypatch.setattr(app, "_announce_turn", announce_mock)

    stored_states: list[GameState] = []

    def fake_store(gs: GameState) -> None:
        stored_states.append(gs)
        state.active_games[gs.game_id] = gs
        state.chat_to_game[gs.chat_id] = gs.game_id

    monkeypatch.setattr(app, "_store_state", fake_store)

    job_queue = DummyJobQueue()
    bot = SimpleNamespace(send_message=AsyncMock())
    context = SimpleNamespace(bot=bot, job_queue=job_queue, chat_data={})
    query_message = SimpleNamespace(
        chat=SimpleNamespace(id=base_state.chat_id, type=ChatType.GROUP),
        message_thread_id=None,
        reply_text=AsyncMock(),
    )
    query = SimpleNamespace(
        data=f"{app.ADMIN_TEST_GAME_CALLBACK_PREFIX}{base_state.chat_id}",
        answer=AsyncMock(),
        message=query_message,
    )
    update = SimpleNamespace(
        callback_query=query,
        effective_user=SimpleNamespace(id=700, full_name="Админ", username="admin"),
        effective_chat=query_message.chat,
        effective_message=query_message,
    )

    await app.admin_test_game_callback_handler(update, context)

    assert load_calls and load_calls[0] == stale_game_id
    assert len(load_calls) >= 2 and str(load_calls[1]) == str(base_state.chat_id)
    assert state.active_games.get(base_state.game_id) is base_state
    assert stored_states and stored_states[-1].game_id.startswith("admin:")
    announce_mock.assert_awaited()
    query.answer.assert_awaited_with("Тестовая игра запущена!")
    cleanup_mock.assert_not_called()


@pytest.mark.anyio
async def test_dummy_turn_job_success(monkeypatch, tmp_path, fresh_state, caplog):
    puzzle = _make_turn_puzzle()
    puzzle.slots = puzzle.slots[:1]
    game_state = _make_turn_state(-950, puzzle)
    game_state.test_mode = True
    game_state.dummy_user_id = app.DUMMY_USER_ID
    dummy_player = Player(user_id=app.DUMMY_USER_ID, name="Dummy", is_bot=True)
    game_state.players[app.DUMMY_USER_ID] = dummy_player
    game_state.turn_order = [app.DUMMY_USER_ID]
    game_state.turn_index = 0
    game_state.active_slot_id = None
    game_state.scoreboard = {app.DUMMY_USER_ID: 0}
    state.active_games[game_state.game_id] = game_state
    state.chat_to_game[game_state.chat_id] = game_state.game_id

    monkeypatch.setattr(app, "_load_state_by_game_id", lambda _: game_state)
    monkeypatch.setattr(app, "_load_puzzle_for_state", lambda _: puzzle)
    finish_mock = AsyncMock()
    monkeypatch.setattr(app, "_finish_game", finish_mock)
    monkeypatch.setattr(app.random, "random", lambda: 0.1)

    stored_states: list[GameState] = []

    def fake_store(gs: GameState) -> None:
        stored_states.append(gs)
        state.active_games[gs.game_id] = gs

    monkeypatch.setattr(app, "_store_state", fake_store)
    monkeypatch.setattr(app, "_announce_turn", AsyncMock())

    job_name = f"dummy-turn-{game_state.game_id}"
    game_state.dummy_job_id = job_name
    state.scheduled_jobs[job_name] = DummyJob(game_state.chat_id, job_name)
    game_state.dummy_turn_started_at = time.time() - 0.5

    job = SimpleNamespace(name=job_name, data={"game_id": game_state.game_id, "planned_delay": 0.5})
    context = SimpleNamespace(job=job, bot=SimpleNamespace(send_message=AsyncMock()))

    caplog.set_level("INFO")
    await app._dummy_turn_job(context)

    assert game_state.scoreboard[app.DUMMY_USER_ID] == app.SCORE_PER_WORD
    assert game_state.dummy_successes == 1
    assert dummy_player.answers_ok == 1
    finish_mock.assert_awaited()
    assert any("Dummy turn" in record.message for record in caplog.records)
    assert job_name not in state.scheduled_jobs
