import asyncio
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from telegram import InlineKeyboardMarkup, constants
from telegram.constants import ChatType
from telegram.error import BadRequest
from telegram.ext import ConversationHandler

import app
from app import (
    ANSWER_INSTRUCTIONS_TEXT,
    HINT_PENALTY,
    LANGUAGE_STATE,
    LOBBY_START_CALLBACK_PREFIX,
    LOBBY_WAIT_CALLBACK_PREFIX,
    MAX_LOBBY_PLAYERS,
    MENU_STATE,
    NEW_GAME_MODE_GROUP,
    NEW_GAME_MODE_SOLO,
    THEME_STATE,
    Settings,
    finish_command,
    hint_command,
    inline_answer_handler,
    join_command,
    lobby_link_callback_handler,
    lobby_start_callback_handler,
    lobby_start_button_handler,
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
    state.dm_chat_to_game.clear()
    state.join_codes.clear()
    state.lobby_messages.clear()
    state.lobby_host_invites.clear()
    state.generating_chats.clear()
    state.lobby_generation_tasks.clear()
    state.scheduled_jobs.clear()
    state.chat_threads.clear()
    yield
    state.active_games.clear()
    state.chat_to_game.clear()
    state.player_chats.clear()
    state.dm_chat_to_game.clear()
    state.join_codes.clear()
    state.lobby_messages.clear()
    state.lobby_host_invites.clear()
    state.generating_chats.clear()
    state.lobby_generation_tasks.clear()
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


class FalseyJobQueue(DummyJobQueue):
    def __bool__(self) -> bool:
        return False


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
        active_slot_id=None,
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


def test_iter_player_dm_chats_includes_group_when_host_missing_dm():
    game_state = GameState(
        chat_id=-500,
        puzzle_id="demo",
        host_id=1,
        players={
            1: Player(user_id=1, name="Host", dm_chat_id=None),
            2: Player(user_id=2, name="Player", dm_chat_id=202),
        },
        scoreboard={},
        mode="turn_based",
        status="lobby",
    )

    chats = app._iter_player_dm_chats(game_state)

    assert (2, 202) in chats
    assert (None, -500) in chats


def test_iter_player_dm_chats_skips_group_once_host_has_dm():
    game_state = GameState(
        chat_id=-501,
        puzzle_id="demo",
        host_id=1,
        players={
            1: Player(user_id=1, name="Host", dm_chat_id=101),
            2: Player(user_id=2, name="Player", dm_chat_id=202),
        },
        scoreboard={},
        mode="turn_based",
        status="lobby",
    )

    chats = app._iter_player_dm_chats(game_state)

    assert (None, -501) not in chats
    assert chats.count((1, 101)) == 1
    assert chats.count((2, 202)) == 1


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


@pytest.mark.anyio
async def test_publish_lobby_message_private_records_dm_messages(fresh_state):
    puzzle = _make_turn_puzzle()
    game_state = _make_turn_state(555, puzzle)
    game_state.status = "lobby"
    game_state.mode = "turn_based"

    first_response = SimpleNamespace(message_id=401)
    second_response = SimpleNamespace(message_id=402)
    bot = SimpleNamespace(
        send_message=AsyncMock(side_effect=[first_response, second_response])
    )
    context = SimpleNamespace(bot=bot)

    await app._publish_lobby_message(context, game_state)

    host_chat_id = game_state.players[game_state.host_id].dm_chat_id
    assert bot.send_message.await_count == len(game_state.players)
    first_call, second_call = bot.send_message.await_args_list
    assert first_call.kwargs["chat_id"] == host_chat_id
    assert second_call.kwargs["chat_id"] != host_chat_id
    assert state.lobby_host_invites[game_state.game_id] == (
        host_chat_id,
        first_response.message_id,
    )
    assert state.lobby_messages[game_state.game_id] == {
        second_call.kwargs["chat_id"]: second_response.message_id
    }


@pytest.mark.anyio
async def test_generation_notice_broadcasts_to_private_players(fresh_state):
    puzzle = _make_turn_puzzle()
    game_state = _make_turn_state(777, puzzle)
    game_state.status = "lobby"
    bot = SimpleNamespace(send_message=AsyncMock())
    context = SimpleNamespace(
        bot=bot,
        application=SimpleNamespace(chat_data={}),
        job_queue=None,
    )
    host_chat_id = next(player.dm_chat_id for player in game_state.players.values())
    host_message = SimpleNamespace(
        chat=SimpleNamespace(id=host_chat_id),
        reply_text=AsyncMock(),
    )

    await app._send_generation_notice_to_game(
        context,
        game_state,
        "Тестовое уведомление",
        message=host_message,
    )

    host_message.reply_text.assert_awaited_once()
    bot.send_message.assert_awaited()
    sent_chat_ids = {call.kwargs["chat_id"] for call in bot.send_message.await_args_list}
    expected_receivers = {
        player.dm_chat_id
        for player in game_state.players.values()
        if player.dm_chat_id != host_chat_id
    }
    assert sent_chat_ids == expected_receivers


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
async def test_new_game_menu_group_allows_private_chat(monkeypatch, fresh_state):
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
    group_mock = AsyncMock(return_value=LANGUAGE_STATE)
    monkeypatch.setattr(app, "_start_new_group_game", group_mock)

    result = await new_game_menu_callback_handler(update, context)

    assert result == LANGUAGE_STATE
    group_mock.assert_awaited_once_with(update, context)
    query.answer.assert_awaited_once()
    call = query.answer.await_args
    assert call.kwargs.get("show_alert") is not True


@pytest.mark.anyio
async def test_private_multiplayer_flow_from_dm(monkeypatch, fresh_state):
    chat = SimpleNamespace(id=606, type=ChatType.PRIVATE)
    host_user = SimpleNamespace(id=42, full_name="Хост", username="host")
    start_message = SimpleNamespace(message_thread_id=None, reply_text=AsyncMock())
    update_start = SimpleNamespace(
        effective_chat=chat,
        effective_message=start_message,
        effective_user=host_user,
    )
    send_message_mock = AsyncMock(return_value=SimpleNamespace(message_id=777))
    context = SimpleNamespace(
        chat_data={},
        user_data={},
        bot=SimpleNamespace(send_message=send_message_mock),
    )

    monkeypatch.setattr(app, "uuid4", lambda: SimpleNamespace(hex="roomdm001"))
    monkeypatch.setattr(app, "save_state", lambda *_: None)
    monkeypatch.setattr(app, "load_state", lambda *_: None)

    result = await app._start_new_group_game(update_start, context)

    assert result == LANGUAGE_STATE
    start_message.reply_text.assert_awaited_once()
    game_id = state.chat_to_game[chat.id]
    assert game_id == "roomdm001"
    assert state.dm_chat_to_game[chat.id] == game_id
    assert state.player_chats[host_user.id] == chat.id
    game_state = state.active_games[game_id]
    assert game_state.host_id == host_user.id
    assert game_state.players[host_user.id].dm_chat_id == chat.id

    language_message = SimpleNamespace(
        message_thread_id=None,
        text="ru",
        reply_text=AsyncMock(),
    )
    update_language = SimpleNamespace(
        effective_chat=chat,
        effective_message=language_message,
        effective_user=host_user,
    )

    lang_result = await app.handle_language(update_language, context)

    assert lang_result == THEME_STATE
    assert state.active_games[game_id].language == "ru"
    language_message.reply_text.assert_awaited_once()

    theme_message = SimpleNamespace(
        message_thread_id=None,
        text="История",
        reply_text=AsyncMock(),
    )
    update_theme = SimpleNamespace(
        effective_chat=chat,
        effective_message=theme_message,
        effective_user=host_user,
    )

    generation_calls: list[tuple] = []

    async def fake_run(context_arg, game_id_arg, language_arg, theme_arg):
        generation_calls.append((context_arg, game_id_arg, language_arg, theme_arg))
        state.lobby_generation_tasks.pop(game_id_arg, None)

    monkeypatch.setattr(app, "_run_lobby_puzzle_generation", fake_run)
    run_generate_mock = AsyncMock()
    monkeypatch.setattr(app, "_run_generate_puzzle", run_generate_mock)

    theme_result = await app.handle_theme(update_theme, context)

    assert theme_result == ConversationHandler.END
    await asyncio.sleep(0)
    assert generation_calls == [(context, game_id, "ru", "История")]
    run_generate_mock.assert_not_awaited()
    theme_message.reply_text.assert_awaited()
    assert game_state.theme == "История"
    assert state.lobby_host_invites[game_id][0] == chat.id
    assert send_message_mock.await_count >= 1
    assert state.lobby_generation_tasks.get(game_id) is None


@pytest.mark.anyio
async def test_handle_theme_admin_test_launch(monkeypatch, fresh_state):
    puzzle = _make_turn_puzzle()
    chat_id = -204
    base_state = _make_turn_state(chat_id, puzzle)
    admin_id = base_state.host_id
    admin_player = base_state.players[admin_id]
    base_state.status = "lobby"
    base_state.language = "ru"
    base_state.theme = None
    base_state.puzzle_id = "previous"
    base_state.puzzle_ids = None
    base_state.players = {admin_id: admin_player}
    base_state.scoreboard = {}
    base_state.turn_order = []
    base_state.turn_index = 0
    state.active_games[base_state.game_id] = base_state
    state.chat_to_game[chat_id] = base_state.game_id
    state.lobby_messages[base_state.game_id] = {chat_id: 42}
    state.settings = SimpleNamespace(admin_id=admin_id)

    generated_state = GameState(
        chat_id=base_state.chat_id,
        puzzle_id=puzzle.id,
        game_id=base_state.game_id,
        hinted_cells=set(),
        players={admin_id: admin_player},
        scoreboard={admin_id: 0},
        host_id=admin_id,
        mode="turn_based",
        status="lobby",
    )

    run_generate_mock = AsyncMock(return_value=(puzzle, generated_state))
    monkeypatch.setattr(app, "_run_generate_puzzle", run_generate_mock)
    monkeypatch.setattr(app, "_load_state_for_chat", lambda _: base_state)
    monkeypatch.setattr(app, "delete_puzzle", lambda _pid: None)
    monkeypatch.setattr(app, "_clone_puzzle_for_test", lambda _p: (puzzle, puzzle.id, None))
    monkeypatch.setattr(app, "_load_state_by_game_id", lambda _gid: None)
    monkeypatch.setattr(app, "_register_player_chat", lambda *args, **kwargs: None)
    monkeypatch.setattr(app, "set_chat_mode", lambda *args, **kwargs: None)
    monkeypatch.setattr(app, "_schedule_game_timers", lambda *args, **kwargs: None)
    deliver_mock = AsyncMock()
    monkeypatch.setattr(app, "_deliver_puzzle_via_bot", deliver_mock)
    monkeypatch.setattr(app, "_iter_player_dm_chats", lambda _gs: [])
    monkeypatch.setattr(app, "_broadcast_photo_to_players", AsyncMock())
    monkeypatch.setattr(app, "_broadcast_to_players", AsyncMock())
    announce_mock = AsyncMock()
    monkeypatch.setattr(app, "_announce_turn", announce_mock)

    stored_states: list[GameState] = []

    def fake_store(game_state: GameState) -> None:
        stored_states.append(game_state)
        state.active_games[game_state.game_id] = game_state

    monkeypatch.setattr(app, "_store_state", fake_store)

    message = SimpleNamespace(
        message_thread_id=None,
        text="Тестовая тема",
        reply_text=AsyncMock(),
    )
    chat = SimpleNamespace(id=chat_id, type=ChatType.GROUP)
    user = SimpleNamespace(id=admin_id, full_name="Админ", username="admin")
    update = SimpleNamespace(
        effective_chat=chat,
        effective_message=message,
        effective_user=user,
    )
    context = SimpleNamespace(
        bot=SimpleNamespace(send_message=AsyncMock()),
        chat_data={app.PENDING_ADMIN_TEST_KEY: chat_id},
        user_data={},
    )

    result = await app.handle_theme(update, context)

    assert result == ConversationHandler.END
    run_generate_mock.assert_awaited_once()
    assert message.reply_text.await_count == 1
    assert "Подбираю" in message.reply_text.await_args_list[0].args[0]
    send_calls = context.bot.send_message.await_args_list
    assert len(send_calls) == 1
    assert "Тестовая игра 1×1" in send_calls[0].kwargs["text"]
    assert send_calls[0].kwargs["chat_id"] == chat_id
    assert app.PENDING_ADMIN_TEST_KEY not in context.chat_data
    announce_mock.assert_awaited()
    assert stored_states, "Admin state should be stored"


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


def test_lobby_keyboard_excludes_admin_button_after_entry(fresh_state):
    puzzle = _make_turn_puzzle()
    game_state = _make_turn_state(-401, puzzle)
    game_state.status = "lobby"
    state.settings = SimpleNamespace(admin_id=game_state.host_id)

    keyboard = app._build_lobby_keyboard(game_state)

    admin_callbacks = [
        button.callback_data
        for row in keyboard.inline_keyboard
        for button in row
        if button.callback_data
    ]
    assert all(
        not callback.startswith(app.ADMIN_TEST_GAME_CALLBACK_PREFIX)
        for callback in admin_callbacks
    )


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
async def test_join_name_accepts_plain_text_with_pending(monkeypatch, fresh_state):
    puzzle = _make_turn_puzzle()
    game_state = _make_turn_state(-510, puzzle)
    game_state.status = "lobby"
    host = game_state.players[game_state.host_id]
    host.dm_chat_id = 1001
    pending_game_id = "roompending001"

    def fake_load_state(game_id: str):
        return game_state if game_id == pending_game_id else None

    monkeypatch.setattr(app, "_load_state_by_game_id", fake_load_state)

    registrations: list[tuple[int, int]] = []

    def fake_register_player_chat(user_id: int, dm_chat_id: int) -> None:
        registrations.append((user_id, dm_chat_id))

    monkeypatch.setattr(app, "_register_player_chat", fake_register_player_chat)

    added_players: list[Player] = []

    def fake_ensure_player_entry(
        state_obj: GameState, tg_user, name: str, dm_chat_id: int
    ) -> Player:
        player = Player(user_id=tg_user.id, name=name, dm_chat_id=dm_chat_id)
        state_obj.players[player.user_id] = player
        added_players.append(player)
        return player

    monkeypatch.setattr(app, "_ensure_player_entry", fake_ensure_player_entry)

    stored_states: list[GameState] = []

    def fake_store_state(state_obj: GameState) -> None:
        stored_states.append(state_obj)

    monkeypatch.setattr(app, "_store_state", fake_store_state)

    broadcasts: list[tuple] = []

    async def fake_broadcast(context_obj, state_obj: GameState, text: str) -> None:
        broadcasts.append((context_obj, state_obj, text))

    monkeypatch.setattr(app, "_broadcast_to_players", fake_broadcast)

    lobby_updates: list[tuple] = []

    async def fake_update_lobby(context_obj, state_obj: GameState) -> None:
        lobby_updates.append((context_obj, state_obj))

    monkeypatch.setattr(app, "_update_lobby_message", fake_update_lobby)

    chat = SimpleNamespace(id=3030, type=ChatType.PRIVATE)
    message = SimpleNamespace(
        message_thread_id=None,
        text="Новый игрок",
        reply_text=AsyncMock(),
        reply_to_message=None,
    )
    user = SimpleNamespace(id=9999, full_name="Новый игрок")
    update = SimpleNamespace(
        effective_chat=chat,
        effective_message=message,
        effective_user=user,
    )
    context = SimpleNamespace(
        bot=SimpleNamespace(id=123456),
        chat_data={},
        user_data={"pending_join": {"game_id": pending_game_id, "code": "ROOM01"}},
    )

    await app.join_name_response_handler(update, context)

    assert context.user_data.get("player_name") == "Новый игрок"
    assert "pending_join" not in context.user_data
    assert added_players and added_players[0].name == "Новый игрок"
    assert registrations == [(user.id, chat.id)]
    assert stored_states == [game_state]
    assert broadcasts and broadcasts[0][1] is game_state
    assert "Новый игрок" in broadcasts[0][2]
    assert lobby_updates == [(context, game_state)]
    message.reply_text.assert_awaited_once()
    assert game_state.players[user.id].dm_chat_id == chat.id


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
async def test_lobby_start_callback_private_broadcasts_start(monkeypatch, fresh_state):
    puzzle = _make_turn_puzzle()
    game_state = _make_turn_state(888, puzzle)
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

    send_message_mock = AsyncMock()
    context = SimpleNamespace(bot=SimpleNamespace(send_message=send_message_mock), job_queue=DummyJobQueue())
    query_message = SimpleNamespace(
        chat=SimpleNamespace(id=game_state.chat_id, type=ChatType.PRIVATE),
        message_thread_id=None,
    )
    query = SimpleNamespace(
        data=f"{LOBBY_START_CALLBACK_PREFIX}{game_state.game_id}",
        answer=AsyncMock(),
        message=query_message,
    )
    update = SimpleNamespace(
        callback_query=query,
        effective_user=SimpleNamespace(id=1),
        effective_chat=query_message.chat,
        effective_message=query_message,
    )

    await lobby_start_callback_handler(update, context)

    expected_dm_chats = {player.dm_chat_id for player in game_state.players.values()}
    sent_chats = {call.kwargs["chat_id"] for call in send_message_mock.await_args_list}
    assert sent_chats == expected_dm_chats
    assert send_message_mock.await_count == len(expected_dm_chats)


@pytest.mark.anyio
async def test_lobby_start_button_triggers_start_without_inline_error(monkeypatch, fresh_state):
    puzzle = _make_turn_puzzle()
    game_state = _make_turn_state(-575, puzzle)
    game_state.status = "lobby"

    state.active_games[game_state.game_id] = game_state
    state.chat_to_game[game_state.chat_id] = game_state.game_id
    host_dm_chat = game_state.players[game_state.host_id].dm_chat_id
    state.dm_chat_to_game[host_dm_chat] = game_state.game_id

    chat = SimpleNamespace(id=host_dm_chat, type=ChatType.PRIVATE)
    message = SimpleNamespace(
        text=app.LOBBY_START_BUTTON_TEXT,
        message_thread_id=None,
        reply_text=AsyncMock(),
        message_id=77,
    )
    user = SimpleNamespace(id=game_state.host_id, full_name="Хост", username="host")
    update = SimpleNamespace(
        effective_chat=chat,
        effective_message=message,
        effective_user=user,
    )
    context = SimpleNamespace(user_data={}, chat_data={}, bot=SimpleNamespace(), job_queue=None)

    submission_mock = AsyncMock()
    monkeypatch.setattr(app, "_handle_answer_submission", submission_mock)

    def fake_load_state(chat_id: int):
        if chat_id in {game_state.chat_id, host_dm_chat}:
            return game_state
        return None

    monkeypatch.setattr(app, "_load_state_for_chat", fake_load_state)
    process_mock = AsyncMock()
    monkeypatch.setattr(app, "_process_lobby_start", process_mock)

    await inline_answer_handler(update, context)

    submission_mock.assert_not_awaited()
    message.reply_text.assert_not_awaited()

    await lobby_start_button_handler(update, context)

    process_mock.assert_awaited_once()
    args, kwargs = process_mock.await_args
    assert args == (context, game_state, user)
    assert kwargs == {"trigger_message": message}


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
async def test_lobby_contact_handler_uses_reply_keyboard(monkeypatch, fresh_state):
    host_id = 321
    game_id = "lobby321"
    request_id = 987
    chat = SimpleNamespace(id=999, type=ChatType.PRIVATE)
    message = SimpleNamespace(
        message_thread_id=None,
        reply_text=AsyncMock(),
        user_shared=SimpleNamespace(user_id=654, request_id=request_id),
        users_shared=None,
        contact=None,
    )
    update = SimpleNamespace(
        effective_chat=chat,
        effective_message=message,
        effective_user=SimpleNamespace(id=host_id, full_name="Хост", username="host"),
    )
    bot = SimpleNamespace(
        get_chat=AsyncMock(return_value=SimpleNamespace(full_name="Игрок")),
        send_message=AsyncMock(),
    )
    application = SimpleNamespace(user_data={})
    context = SimpleNamespace(
        application=application,
        bot=bot,
        bot_data={},
        chat_data={},
    )
    user_store = app._ensure_user_store_for(context, host_id)
    user_store["pending_invite"] = {
        "request_id": request_id,
        "game_id": game_id,
        "code": None,
    }
    game_state = GameState(
        chat_id=-500,
        puzzle_id="puzzle",
        host_id=host_id,
        game_id=game_id,
        mode="turn_based",
        status="lobby",
        players={host_id: Player(user_id=host_id, name="Хост", dm_chat_id=chat.id)},
    )
    state.active_games[game_id] = game_state
    state.chat_to_game[game_state.chat_id] = game_id

    def fake_assign(gs: GameState) -> str:
        gs.join_codes["XYZ123"] = gs.game_id
        return "XYZ123"

    monkeypatch.setattr(app, "_assign_join_code", fake_assign)
    monkeypatch.setattr(app, "_store_state", lambda gs: state.active_games.__setitem__(gs.game_id, gs))
    monkeypatch.setattr(app, "_build_join_link", AsyncMock(return_value="https://t.me/bot?start=join_XYZ123"))

    await app.lobby_contact_handler(update, context)

    bot.send_message.assert_awaited_once()
    host_call = message.reply_text.await_args
    assert "Код для подключения: XYZ123" in host_call.args[0]
    assert host_call.kwargs.get("reply_markup") is not None


@pytest.mark.anyio
async def test_lobby_contact_handler_explains_bad_request(monkeypatch, fresh_state):
    host_id = 111
    game_id = "lobby111"
    request_id = 222
    chat = SimpleNamespace(id=555, type=ChatType.PRIVATE)
    message = SimpleNamespace(
        message_thread_id=None,
        reply_text=AsyncMock(),
        user_shared=SimpleNamespace(user_id=777, request_id=request_id),
        users_shared=None,
        contact=None,
    )
    update = SimpleNamespace(
        effective_chat=chat,
        effective_message=message,
        effective_user=SimpleNamespace(id=host_id, full_name="Организатор"),
    )
    bot = SimpleNamespace(
        get_chat=AsyncMock(return_value=SimpleNamespace(full_name="Участник")),
        send_message=AsyncMock(side_effect=BadRequest("Chat not found")),
    )
    application = SimpleNamespace(user_data={})
    context = SimpleNamespace(
        application=application,
        bot=bot,
        bot_data={},
        chat_data={},
    )
    user_store = app._ensure_user_store_for(context, host_id)
    user_store["pending_invite"] = {
        "request_id": request_id,
        "game_id": game_id,
        "code": None,
    }
    game_state = GameState(
        chat_id=-900,
        puzzle_id="puzzle",
        host_id=host_id,
        game_id=game_id,
        mode="turn_based",
        status="lobby",
        players={host_id: Player(user_id=host_id, name="Организатор", dm_chat_id=chat.id)},
    )
    state.active_games[game_id] = game_state
    state.chat_to_game[game_state.chat_id] = game_id

    def fake_assign(gs: GameState) -> str:
        gs.join_codes["JOIN42"] = gs.game_id
        return "JOIN42"

    monkeypatch.setattr(app, "_assign_join_code", fake_assign)
    monkeypatch.setattr(app, "_store_state", lambda gs: state.active_games.__setitem__(gs.game_id, gs))
    monkeypatch.setattr(app, "_build_join_link", AsyncMock(return_value=None))

    await app.lobby_contact_handler(update, context)

    bot.send_message.assert_awaited_once()
    host_call = message.reply_text.await_args
    host_message = host_call.args[0]
    assert "бот не может отправить приглашение" in host_message.lower()
    assert "Код для подключения: JOIN42" in host_message
    assert host_call.kwargs.get("reply_markup") is not None


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
async def test_correct_answer_sends_clues_before_turn(monkeypatch, tmp_path, fresh_state):
    puzzle = _make_turn_puzzle()
    game_state = _make_turn_state(-702, puzzle)
    state.active_games[game_state.game_id] = game_state
    state.chat_to_game[game_state.chat_id] = game_state.game_id

    monkeypatch.setattr(app, "_load_state_for_chat", lambda _: game_state)
    monkeypatch.setattr(app, "_load_puzzle_for_state", lambda _: puzzle)
    monkeypatch.setattr(app, "_broadcast_photo_to_players", AsyncMock())

    stored_states: list[GameState] = []

    def fake_store(gs: GameState) -> None:
        stored_states.append(gs)
        state.active_games[gs.game_id] = gs
        state.chat_to_game[gs.chat_id] = gs.game_id

    monkeypatch.setattr(app, "_store_state", fake_store)

    broadcast_mock = AsyncMock(
        side_effect=[app.BroadcastResult(successful_chats=set()) for _ in range(3)]
    )
    monkeypatch.setattr(app, "_broadcast_to_players", broadcast_mock)

    image_path = tmp_path / "grid.png"
    image_path.write_bytes(b"png")
    monkeypatch.setattr(app, "render_puzzle", lambda _p, _s: str(image_path))

    bot = SimpleNamespace(
        send_chat_action=AsyncMock(),
        send_photo=AsyncMock(),
        send_message=AsyncMock(),
    )
    context = SimpleNamespace(bot=bot, job_queue=DummyJobQueue())

    chat = SimpleNamespace(id=game_state.chat_id, type=ChatType.GROUP)
    message = SimpleNamespace(
        reply_text=AsyncMock(),
        reply_photo=AsyncMock(),
        from_user=SimpleNamespace(id=1, full_name="Игрок 1"),
    )

    await app._handle_answer_submission(context, chat, message, "A1", "рим")

    assert broadcast_mock.await_count == 3
    first_call = broadcast_mock.await_args_list[0]
    second_call = broadcast_mock.await_args_list[1]
    third_call = broadcast_mock.await_args_list[2]

    updated_clues = app._format_clues_message(puzzle, game_state)
    assert first_call.args[2] == updated_clues
    assert first_call.kwargs.get("parse_mode") == constants.ParseMode.HTML
    assert second_call.args[2] == ANSWER_INSTRUCTIONS_TEXT
    assert "Ход игрока" in third_call.args[2]

    send_calls = bot.send_message.await_args_list
    assert len(send_calls) >= 3
    first_kwargs = send_calls[0].kwargs
    assert first_kwargs.get("text") == updated_clues
    assert first_kwargs.get("parse_mode") == constants.ParseMode.HTML
    assert first_kwargs.get("reply_markup") is None
    assert send_calls[1].kwargs.get("text") == ANSWER_INSTRUCTIONS_TEXT
    assert send_calls[1].kwargs.get("parse_mode") is None
    assert "Ход игрока" in send_calls[2].kwargs.get("text", "")

    assert stored_states, "Game state should be stored after correct answer"

@pytest.mark.anyio
async def test_turn_based_rejects_rapid_second_answer(monkeypatch, tmp_path, fresh_state):
    puzzle = _make_turn_puzzle()
    game_state = _make_turn_state(-701, puzzle)
    state.active_games[game_state.game_id] = game_state
    state.chat_to_game[game_state.chat_id] = game_state.game_id

    monkeypatch.setattr(app, "_load_state_for_chat", lambda _: game_state)
    monkeypatch.setattr(app, "_load_puzzle_for_state", lambda _: puzzle)

    def fake_store(gs: GameState) -> None:
        state.active_games[gs.game_id] = gs
        state.chat_to_game[gs.chat_id] = gs.game_id

    monkeypatch.setattr(app, "_store_state", fake_store)
    monkeypatch.setattr(app, "_send_clues_update", AsyncMock())
    announce_mock = AsyncMock()
    monkeypatch.setattr(app, "_broadcast_photo_to_players", AsyncMock())
    monkeypatch.setattr(app, "_announce_turn", announce_mock)

    image_path = tmp_path / "grid.png"
    image_path.write_bytes(b"png")
    monkeypatch.setattr(app, "render_puzzle", lambda _p, _s: str(image_path))

    job_queue = DummyJobQueue()
    bot = SimpleNamespace(send_chat_action=AsyncMock(), send_message=AsyncMock())
    context = SimpleNamespace(bot=bot, job_queue=job_queue)

    _update1, chat, message1 = _make_group_update(game_state.chat_id, user_id=1)

    start_photo = asyncio.Event()
    finish_photo = asyncio.Event()

    async def delayed_reply_photo(*args, **kwargs):  # noqa: ANN001 - signature matches AsyncMock usage
        start_photo.set()
        await finish_photo.wait()

    message1.reply_photo = AsyncMock(side_effect=delayed_reply_photo)

    first_task = asyncio.create_task(
        app._handle_answer_submission(context, chat, message1, "A1", "рим")
    )

    await start_photo.wait()
    await asyncio.sleep(0)
    assert game_state.turn_index == 1

    game_state.active_slot_id = "D1"

    _, _, message2 = _make_group_update(game_state.chat_id, user_id=1)
    message2.reply_photo = AsyncMock()
    message2.reply_text = AsyncMock()

    await app._handle_answer_submission(context, chat, message2, "D1", "дон")

    message2.reply_text.assert_awaited()
    assert "Сейчас ход" in message2.reply_text.await_args.args[0]
    message2.reply_photo.assert_not_awaited()

    game_state.active_slot_id = None

    finish_photo.set()
    await first_task

    announce_mock.assert_awaited()


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

    assert announce_mock.await_count == 3
    first_call = announce_mock.await_args_list[0]
    second_call = announce_mock.await_args_list[1]
    third_call = announce_mock.await_args_list[2]
    assert first_call.kwargs.get("text") == app._format_clues_message(puzzle, game_state)
    assert first_call.kwargs.get("parse_mode") == constants.ParseMode.HTML
    assert second_call.kwargs.get("text") == ANSWER_INSTRUCTIONS_TEXT
    assert second_call.kwargs.get("parse_mode") is None
    announce_kwargs = third_call.kwargs
    assert announce_kwargs["chat_id"] == chat_id
    assert "message_thread_id" not in announce_kwargs
    text = announce_kwargs.get("text", "")
    assert "Отправьте ответ прямо в чат" in text
    assert "A1 - париж" in text
    assert "/answer" not in text
    assert announce_kwargs.get("reply_markup") is None

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


@pytest.mark.anyio
async def test_turn_based_answer_without_keyboard(monkeypatch, tmp_path, fresh_state):
    puzzle = _make_turn_puzzle()
    game_state = _make_turn_state(-612, puzzle)
    state.active_games[game_state.game_id] = game_state
    state.chat_to_game[game_state.chat_id] = game_state.game_id

    monkeypatch.setattr(app, "_load_state_for_chat", lambda _: game_state)
    monkeypatch.setattr(app, "_load_puzzle_for_state", lambda _: puzzle)
    monkeypatch.setattr(app, "_store_state", lambda _gs: None)
    monkeypatch.setattr(app, "_send_clues_update", AsyncMock())
    broadcast_mock = AsyncMock()
    monkeypatch.setattr(app, "_broadcast_photo_to_players", broadcast_mock)
    announce_mock = AsyncMock()
    monkeypatch.setattr(app, "_announce_turn", announce_mock)

    image_path = tmp_path / "answer.png"
    image_path.write_bytes(b"png")
    monkeypatch.setattr(app, "render_puzzle", lambda _p, _s: str(image_path))

    context = SimpleNamespace(
        bot=SimpleNamespace(send_chat_action=AsyncMock()),
        job_queue=DummyJobQueue(),
    )

    _update, chat, message = _make_group_update(game_state.chat_id, user_id=1)
    message.reply_photo = AsyncMock()
    message.reply_text = AsyncMock()

    await app._handle_answer_submission(context, chat, message, "A1", "рим")

    assert message.reply_text.await_count == 0
    assert message.reply_photo.await_count == 1
    assert game_state.solved_slots == {"A1"}
    assert game_state.scoreboard[1] == app.SCORE_PER_WORD
    assert game_state.active_slot_id is None
    broadcast_mock.assert_awaited()
    announce_mock.assert_awaited()


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
async def test_turn_based_hint_explicit_slot(monkeypatch, tmp_path, fresh_state):
    puzzle = _make_turn_puzzle()
    game_state = _make_turn_state(-950, puzzle)
    state.active_games[game_state.game_id] = game_state
    state.chat_to_game[game_state.chat_id] = game_state.game_id

    monkeypatch.setattr(app, "_load_state_for_chat", lambda _: game_state)
    monkeypatch.setattr(app, "_load_puzzle_for_state", lambda _: puzzle)
    monkeypatch.setattr(app, "_store_state", lambda _gs: None)

    image_path = tmp_path / "hint.png"
    image_path.write_bytes(b"png")
    monkeypatch.setattr(app, "render_puzzle", lambda _p, _s: str(image_path))

    update, chat, message = _make_group_update(game_state.chat_id, user_id=1)
    message.reply_photo = AsyncMock()
    message.reply_text = AsyncMock()

    context = SimpleNamespace(
        bot=SimpleNamespace(send_chat_action=AsyncMock()),
        args=["A1"],
        job_queue=DummyJobQueue(),
    )

    await hint_command(update, context)

    assert message.reply_text.await_count == 0
    assert message.reply_photo.await_count == 1
    assert game_state.active_slot_id == "A1"
    assert game_state.scoreboard[1] == -HINT_PENALTY


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

    lines = text.splitlines()
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
    state.lobby_messages[base_state.game_id] = {base_state.chat_id: 111}

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
    main_chat_texts = [
        call.kwargs.get("text", "")
        for call in bot.send_message.await_args_list
        if call.kwargs.get("chat_id") == base_state.chat_id
    ]
    assert sum(text.count("Первым ходит") for text in main_chat_texts) == 1
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
    main_chat_texts = [
        call.kwargs.get("text", "")
        for call in bot.send_message.await_args_list
        if call.kwargs.get("chat_id") == base_state.chat_id
    ]
    assert all("Первым ходит" not in text for text in main_chat_texts)
    announce_mock.assert_awaited()
    prefix = announce_mock.await_args.kwargs.get("prefix")
    assert prefix and prefix.count("Первым ходит") == 1
    query.answer.assert_awaited_with("Тестовая игра запущена!")
    cleanup_mock.assert_not_called()


def test_schedule_dummy_turn_clamps_minimum_delay(monkeypatch, fresh_state):
    puzzle = _make_turn_puzzle()
    puzzle.slots = puzzle.slots[:1]
    game_state = _make_turn_state(-999, puzzle)
    game_state.test_mode = True
    game_state.dummy_user_id = app.DUMMY_USER_ID
    dummy_player = Player(user_id=app.DUMMY_USER_ID, name="Dummy", is_bot=True)
    game_state.players[app.DUMMY_USER_ID] = dummy_player
    game_state.turn_order = [app.DUMMY_USER_ID]
    game_state.turn_index = 0
    game_state.active_slot_id = None
    state.active_games[game_state.game_id] = game_state
    state.chat_to_game[game_state.chat_id] = game_state.game_id

    job_queue = DummyJobQueue()
    context = SimpleNamespace(job_queue=job_queue)

    monkeypatch.setattr(app, "DUMMY_DELAY_RANGE", (1.0, 3.0))
    monkeypatch.setattr(app.random, "uniform", lambda *_: 2.0)

    app._schedule_dummy_turn(context, game_state, puzzle)

    assert len(job_queue.submitted) == 1
    _, when, _, _, data = job_queue.submitted[0]
    assert when >= app.MIN_DUMMY_DELAY
    assert data["planned_delay"] >= app.MIN_DUMMY_DELAY
    assert game_state.dummy_planned_delay >= app.MIN_DUMMY_DELAY
    assert state.scheduled_jobs[game_state.dummy_job_id].name == game_state.dummy_job_id


def test_schedule_dummy_turn_with_falsey_job_queue(monkeypatch, fresh_state):
    puzzle = _make_turn_puzzle()
    puzzle.slots = puzzle.slots[:1]
    game_state = _make_turn_state(-998, puzzle)
    game_state.test_mode = True
    game_state.dummy_user_id = app.DUMMY_USER_ID
    dummy_player = Player(user_id=app.DUMMY_USER_ID, name="Dummy", is_bot=True)
    game_state.players = {app.DUMMY_USER_ID: dummy_player}
    game_state.turn_order = [app.DUMMY_USER_ID]
    game_state.turn_index = 0
    game_state.active_slot_id = None

    job_queue = FalseyJobQueue()
    context = SimpleNamespace(job_queue=job_queue)

    monkeypatch.setattr(app.random, "uniform", lambda *_: app.MIN_DUMMY_DELAY + 1)

    app._schedule_dummy_turn(context, game_state, puzzle)

    assert len(job_queue.submitted) == 1
    _, _, _, name, _ = job_queue.submitted[0]
    assert name == game_state.dummy_job_id


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
    image_path = tmp_path / "dummy.png"

    def fake_render(*_args, **_kwargs):
        image_path.write_bytes(b"fake-image")
        return image_path

    monkeypatch.setattr(app, "render_puzzle", fake_render)
    broadcast_photo_mock = AsyncMock()
    monkeypatch.setattr(app, "_broadcast_photo_to_players", broadcast_photo_mock)

    job_name = f"dummy-turn-{game_state.game_id}"
    game_state.dummy_job_id = job_name
    state.scheduled_jobs[job_name] = DummyJob(game_state.chat_id, job_name)
    game_state.dummy_turn_started_at = time.time() - 0.5

    job = SimpleNamespace(name=job_name, data={"game_id": game_state.game_id, "planned_delay": 0.5})
    context = SimpleNamespace(
        job=job,
        bot=SimpleNamespace(
            send_message=AsyncMock(),
            send_photo=AsyncMock(),
        ),
    )

    caplog.set_level("INFO")
    await app._dummy_turn_job(context)

    assert game_state.scoreboard[app.DUMMY_USER_ID] == app.SCORE_PER_WORD
    assert game_state.dummy_successes == 1
    assert dummy_player.answers_ok == 1
    finish_mock.assert_awaited()
    assert any("Dummy turn" in record.message for record in caplog.records)
    assert job_name not in state.scheduled_jobs
    broadcast_photo_mock.assert_awaited()
    expected_caption = (
        f"Верно! 🤖 Dummy - A1: РИМ (+{app.SCORE_PER_WORD} очков)"
    )
    assert broadcast_photo_mock.await_args.kwargs.get("caption") == expected_caption
    context.bot.send_photo.assert_awaited()
    assert context.bot.send_photo.await_args.kwargs.get("caption") == expected_caption
    sent_texts = [
        call.kwargs.get("text")
        for call in context.bot.send_message.await_args_list
    ]
    assert all("/answer" not in text for text in sent_texts if isinstance(text, str))


@pytest.mark.anyio
async def test_dummy_turn_job_admin_test_mirrors_primary_chat(
    monkeypatch, tmp_path, fresh_state
):
    puzzle = _make_turn_puzzle()
    puzzle.slots = puzzle.slots[:1]
    game_state = _make_turn_state(-951, puzzle)
    host_player = game_state.players[1]
    game_state.players = {1: host_player}
    game_state.game_id = f"admin:{game_state.chat_id}"
    game_state.test_mode = True
    game_state.dummy_user_id = app.DUMMY_USER_ID
    dummy_player = Player(
        user_id=app.DUMMY_USER_ID,
        name="Dummy",
        is_bot=True,
        dm_chat_id=None,
    )
    game_state.players[app.DUMMY_USER_ID] = dummy_player
    game_state.turn_order = [app.DUMMY_USER_ID, 1]
    game_state.turn_index = 0
    game_state.active_slot_id = None
    game_state.scoreboard = {1: 0, app.DUMMY_USER_ID: 0}
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

    image_path = tmp_path / "dummy_admin.png"

    def fake_render(*_args, **_kwargs):
        image_path.write_bytes(b"fake-image")
        return image_path

    monkeypatch.setattr(app, "render_puzzle", fake_render)
    monkeypatch.setattr(
        app,
        "_broadcast_photo_to_players",
        AsyncMock(),
    )

    job_name = f"dummy-turn-{game_state.game_id}"
    game_state.dummy_job_id = job_name
    state.scheduled_jobs[job_name] = DummyJob(game_state.chat_id, job_name)
    game_state.dummy_turn_started_at = time.time() - 0.5

    job = SimpleNamespace(
        name=job_name,
        data={"game_id": game_state.game_id, "planned_delay": 0.5},
    )
    context = SimpleNamespace(
        job=job,
        bot=SimpleNamespace(
            send_message=AsyncMock(),
            send_photo=AsyncMock(),
        ),
    )

    await app._dummy_turn_job(context)

    main_chat_messages = [
        call.kwargs["text"]
        for call in context.bot.send_message.await_args_list
        if call.kwargs.get("chat_id") == game_state.chat_id
    ]
    assert any("отвечает на" in text for text in main_chat_messages)
    assert all("разгадал" not in text for text in main_chat_messages)


@pytest.mark.anyio
async def test_dummy_turn_job_falls_back_to_primary_on_dm_failure(
    monkeypatch, fresh_state
):
    puzzle = _make_turn_puzzle()
    dummy_player = Player(
        user_id=-2,
        name="Dummy",
        dm_chat_id=909,
        is_bot=True,
    )
    human_player = Player(user_id=1, name="Админ", dm_chat_id=808)
    game_state = GameState(
        chat_id=-5005,
        puzzle_id=puzzle.id,
        hinted_cells=set(),
        score=0,
        started_at=time.time(),
        last_update=time.time(),
        host_id=human_player.user_id,
        game_id="admin:-5005",
        mode="turn_based",
        status="running",
        players={human_player.user_id: human_player, dummy_player.user_id: dummy_player},
        turn_order=[human_player.user_id, dummy_player.user_id],
        turn_index=1,
        scoreboard={human_player.user_id: 0, dummy_player.user_id: 0},
        test_mode=True,
        dummy_user_id=dummy_player.user_id,
    )
    state.active_games[game_state.game_id] = game_state
    state.chat_to_game[game_state.chat_id] = game_state.game_id

    monkeypatch.setattr(app, "_load_state_by_game_id", lambda _: game_state)
    monkeypatch.setattr(app, "_load_puzzle_for_state", lambda *_: puzzle)
    stored_states: list[GameState] = []
    monkeypatch.setattr(app, "_store_state", lambda gs: stored_states.append(gs))
    monkeypatch.setattr(app, "_finish_game", AsyncMock())
    monkeypatch.setattr(app, "_announce_turn", AsyncMock())
    monkeypatch.setattr(app.random, "random", lambda: 1.0)
    monkeypatch.setattr(
        app, "_generate_dummy_incorrect_answer", lambda *_, **__: "ошибка"
    )

    attempted: list[int] = []
    delivered: list[int] = []

    messages_by_chat: dict[int, list[str]] = {}

    async def fake_send_message(*args, **kwargs):
        chat_id = kwargs.get("chat_id")
        if chat_id is None and args:
            chat_id = args[0]
        attempted.append(chat_id)
        if chat_id == dummy_player.dm_chat_id:
            raise RuntimeError("dm failure")
        delivered.append(chat_id)
        text = kwargs.get("text")
        if isinstance(text, str):
            messages_by_chat.setdefault(chat_id, []).append(text)
        return SimpleNamespace(message_id=700 + len(delivered))

    bot = SimpleNamespace(
        send_message=AsyncMock(side_effect=fake_send_message),
        send_photo=AsyncMock(),
    )
    job = SimpleNamespace(
        name="dummy-turn-test",
        data={"game_id": game_state.game_id, "planned_delay": 0.5},
    )
    state.scheduled_jobs[job.name] = job
    context = SimpleNamespace(job=job, bot=bot)

    await app._dummy_turn_job(context)

    assert attempted.count(dummy_player.dm_chat_id) == 2
    assert delivered.count(game_state.chat_id) == 2
    assert delivered.count(human_player.dm_chat_id) == 2
    for texts in messages_by_chat.values():
        for text in texts:
            assert "/answer" not in text
    failure_messages = messages_by_chat.get(game_state.chat_id, [])
    assert any("ошибся" in text and "ОШИБКА" in text for text in failure_messages)


@pytest.mark.anyio
async def test_announce_turn_dm_failure_uses_primary_chat(monkeypatch, fresh_state):
    puzzle = _make_turn_puzzle()
    player = Player(user_id=1, name="Игрок", dm_chat_id=404)
    game_state = GameState(
        chat_id=-7007,
        puzzle_id=puzzle.id,
        hinted_cells=set(),
        score=0,
        started_at=time.time(),
        last_update=time.time(),
        host_id=player.user_id,
        game_id="turns:-7007",
        mode="turn_based",
        status="running",
        players={player.user_id: player},
        turn_order=[player.user_id],
        turn_index=0,
        scoreboard={player.user_id: 0},
    )

    monkeypatch.setattr(app, "_store_state", lambda *_: None)

    attempted: list[int] = []
    delivered: list[int] = []

    async def fake_send_message(*args, **kwargs):
        chat_id = kwargs.get("chat_id")
        if chat_id is None and args:
            chat_id = args[0]
        attempted.append(chat_id)
        if chat_id == player.dm_chat_id:
            raise RuntimeError("dm failure")
        delivered.append(chat_id)
        return SimpleNamespace(message_id=900 + len(delivered))

    bot = SimpleNamespace(send_message=AsyncMock(side_effect=fake_send_message))
    context = SimpleNamespace(bot=bot, job_queue=None)

    await app._announce_turn(context, game_state, puzzle)

    assert attempted == [
        player.dm_chat_id,
        player.dm_chat_id,
        game_state.chat_id,
        game_state.chat_id,
        player.dm_chat_id,
        game_state.chat_id,
    ]
    assert delivered == [
        game_state.chat_id,
        game_state.chat_id,
        game_state.chat_id,
    ]

