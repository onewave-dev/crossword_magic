"""Tests ensuring replacement candidates respect intersection heuristics."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from app import MAX_REPLACEMENT_REQUESTS, _generate_puzzle
from utils.fill_in_generator import DisconnectedWordError, FillInGenerationError
from utils.llm_generator import WordClue


def test_replacement_prefers_intersecting_words(monkeypatch) -> None:
    """Ensure replacement selection skips candidates without shared letters."""

    base_clues = [
        WordClue(word="AAAA", clue="first"),
        WordClue(word="BBBB", clue="second"),
        WordClue(word="METAL", clue="third"),
        WordClue(word="STONE", clue="fourth"),
        WordClue(word="GLASS", clue="fifth"),
        WordClue(word="PAPER", clue="sixth"),
        WordClue(word="ROCK", clue="seventh"),
        WordClue(word="STEEL", clue="eighth"),
        WordClue(word="BRASS", clue="ninth"),
        WordClue(word="COPPER", clue="tenth"),
    ]
    replacement_candidates = [
        WordClue(word="ZZZZ", clue="no overlap 1"),
        WordClue(word="YYYY", clue="no overlap 2"),
        WordClue(word="XXXX", clue="no overlap 3"),
        WordClue(word="BOLT", clue="shares letters"),
        WordClue(word="VVVV", clue="no overlap 4"),
        WordClue(word="MMMM", clue="no overlap 5"),
    ]

    def fake_generate_clues(
        theme: str,
        language: str,
        *,
        min_results: int = 10,
        max_results: int = 40,
    ):
        if "вместо" in theme:
            call_state["theme"] = theme
            call_state["replacement_args"] = (min_results, max_results)
            return replacement_candidates
        call_state["base_args"] = (min_results, max_results)
        return base_clues

    def fake_validate_word_list(language: str, clues, deduplicate: bool = True):
        return list(clues)

    call_state: dict[str, object] = {
        "count": 0,
        "words": None,
        "theme": None,
        "replacement_args": None,
        "base_args": None,
    }

    def fake_generate_fill_in_puzzle(puzzle_id, theme, language, words, max_size=15):
        call_state["count"] += 1
        if call_state["count"] == 1:
            raise DisconnectedWordError("AAAA")
        call_state["words"] = list(words)
        return SimpleNamespace(
            id=puzzle_id,
            language=language,
            theme=theme,
            slots=[],
        )

    monkeypatch.setattr("app.generate_clues", fake_generate_clues)
    monkeypatch.setattr("app.validate_word_list", fake_validate_word_list)
    monkeypatch.setattr("app.generate_fill_in_puzzle", fake_generate_fill_in_puzzle)
    monkeypatch.setattr("app._assign_clues_to_slots", lambda *args, **kwargs: None)
    monkeypatch.setattr("app.save_puzzle", lambda *args, **kwargs: None)
    monkeypatch.setattr("app.puzzle_to_dict", lambda puzzle: {})
    monkeypatch.setattr("app._store_state", lambda *args, **kwargs: None)

    puzzle, game_state = _generate_puzzle(chat_id=1, language="en", theme="Space")

    assert call_state["count"] == 2
    assert call_state["words"][0] == "BOLT"
    assert "ZZZZ" not in call_state["words"]
    assert len(call_state["words"]) == len(base_clues)
    assert call_state["theme"] is not None
    assert "Подбери 6-8 новых слов" in call_state["theme"]
    assert "Избегай слов:" in call_state["theme"]
    assert "Каждое слово должно содержать хотя бы одну букву" in call_state["theme"]
    assert call_state["replacement_args"] == (6, 8)
    assert call_state["base_args"] == (10, 40)
    assert puzzle.language == "en"
    assert game_state.chat_id == 1


def test_generation_descends_to_smaller_word_sets(monkeypatch) -> None:
    """Ensure generation loop reduces word count before failing entirely."""

    base_clues = [
        WordClue(word=f"WORD{i:02}", clue=f"clue {i}") for i in range(1, 16)
    ]
    attempts: list[int] = []
    requested_sizes: list[int] = []

    def fake_generate_clues(
        theme: str,
        language: str,
        *,
        min_results: int = 10,
        max_results: int = 40,
    ):
        assert (min_results, max_results) == (10, 40)
        return base_clues

    def fake_validate_word_list(language: str, clues, deduplicate: bool = True):
        return list(clues)

    def fake_build_word_components(clues, language):
        return [list(clues)]

    def fake_select_connected_clue_set(components, language, size):
        requested_sizes.append(size)
        if size > len(base_clues) or size <= 0:
            return None
        return list(base_clues[:size])

    def fake_generate_fill_in_puzzle(puzzle_id, theme, language, words, max_size=15):
        attempts.append(len(list(words)))
        if len(words) > 10:
            raise FillInGenerationError("too many words")
        return SimpleNamespace(
            id=puzzle_id,
            language=language,
            theme=theme,
            slots=[],
        )

    monkeypatch.setattr("app.generate_clues", fake_generate_clues)
    monkeypatch.setattr("app.validate_word_list", fake_validate_word_list)
    monkeypatch.setattr("app._build_word_components", fake_build_word_components)
    monkeypatch.setattr("app._select_connected_clue_set", fake_select_connected_clue_set)
    monkeypatch.setattr("app.generate_fill_in_puzzle", fake_generate_fill_in_puzzle)
    monkeypatch.setattr("app._assign_clues_to_slots", lambda *args, **kwargs: None)
    monkeypatch.setattr("app.save_puzzle", lambda *args, **kwargs: None)
    monkeypatch.setattr("app.puzzle_to_dict", lambda puzzle: {})
    monkeypatch.setattr("app._store_state", lambda *args, **kwargs: None)

    puzzle, state = _generate_puzzle(chat_id=99, language="en", theme="Test")

    assert isinstance(puzzle, SimpleNamespace)
    assert state.chat_id == 99
    assert attempts[0] == len(base_clues)
    assert min(attempts) == 10
    assert min(attempts) < len(base_clues)
    assert min(requested_sizes) <= 10


def test_replacement_prefers_highest_scoring_candidate(monkeypatch) -> None:
    """Select the replacement with the strongest letter connectivity."""

    base_clues = [
        WordClue(word="BRIDGE", clue="structure"),
        WordClue(word="BRICK", clue="material"),
        WordClue(word="MORTAR", clue="binder"),
        WordClue(word="COLUMN", clue="support"),
        WordClue(word="BEAM", clue="frame"),
        WordClue(word="ROOF", clue="top"),
        WordClue(word="WINDOW", clue="opening"),
        WordClue(word="DOOR", clue="entry"),
        WordClue(word="STAIRS", clue="steps"),
        WordClue(word="LADDER", clue="access"),
    ]
    replacement_candidates = [
        WordClue(word="ROAD", clue="path"),
        WordClue(word="BOARD", clue="panel"),
        WordClue(word="TILE", clue="cover"),
        WordClue(word="GATE", clue="entry"),
        WordClue(word="RIDGE", clue="crest"),
        WordClue(word="WALL", clue="divider"),
    ]

    call_state: dict[str, object] = {"attempts": 0, "final_words": None}

    def fake_generate_clues(
        theme: str,
        language: str,
        *,
        min_results: int = 10,
        max_results: int = 40,
    ):
        if "вместо" in theme:
            assert (min_results, max_results) == (6, 8)
            return replacement_candidates
        assert (min_results, max_results) == (10, 40)
        return base_clues

    def fake_validate_word_list(language: str, clues, deduplicate: bool = True):
        return list(clues)

    def fake_generate_fill_in_puzzle(puzzle_id, theme, language, words, max_size=15):
        call_state["attempts"] = call_state.get("attempts", 0) + 1
        if call_state["attempts"] == 1:
            raise DisconnectedWordError("BRIDGE")
        call_state["final_words"] = list(words)
        return SimpleNamespace(
            id=puzzle_id,
            language=language,
            theme=theme,
            slots=[],
        )

    monkeypatch.setattr("app.generate_clues", fake_generate_clues)
    monkeypatch.setattr("app.validate_word_list", fake_validate_word_list)
    monkeypatch.setattr("app.generate_fill_in_puzzle", fake_generate_fill_in_puzzle)
    monkeypatch.setattr("app._assign_clues_to_slots", lambda *args, **kwargs: None)
    monkeypatch.setattr("app.save_puzzle", lambda *args, **kwargs: None)
    monkeypatch.setattr("app.puzzle_to_dict", lambda puzzle: {})
    monkeypatch.setattr("app._store_state", lambda *args, **kwargs: None)

    _generate_puzzle(chat_id=1, language="en", theme="Space")

    assert call_state["attempts"] == 2
    assert isinstance(call_state["final_words"], list)
    assert "BOARD" in call_state["final_words"]
    assert "ROAD" not in call_state["final_words"]
    assert len(call_state["final_words"]) == len(base_clues)


def test_rejected_word_can_return_in_followup_attempt(monkeypatch) -> None:
    """Allow previously rejected word to reappear after attempt reset."""

    base_clues = [
        WordClue(word="ALPHA", clue="first"),
        WordClue(word="BETA", clue="second"),
        WordClue(word="GAMMA", clue="third"),
        WordClue(word="DELTA", clue="fourth"),
        WordClue(word="EPSILON", clue="fifth"),
        WordClue(word="ZETA", clue="sixth"),
        WordClue(word="ETA", clue="seventh"),
        WordClue(word="THETA", clue="eighth"),
        WordClue(word="IOTA", clue="ninth"),
        WordClue(word="KAPPA", clue="tenth"),
    ]

    call_state: dict[str, object] = {
        "replacement_calls": 0,
        "generate_calls": [],
        "final_words": None,
        "selection_sizes": [],
    }

    def fake_generate_clues(
        theme: str,
        language: str,
        *,
        min_results: int = 10,
        max_results: int = 40,
    ):
        if "вместо" in theme:
            call_state["replacement_calls"] = int(call_state["replacement_calls"]) + 1
            if call_state["replacement_calls"] == 1:
                return [
                    WordClue(word="BETA", clue="duplicate"),
                    WordClue(word="OMEGA", clue="alt"),
                ]
            return [WordClue(word="BETA", clue="retry")]
        return base_clues

    def fake_validate_word_list(language: str, clues, deduplicate: bool = True):
        return list(clues)

    def fake_build_word_components(clues, language):
        return [list(clues)]

    def fake_select_connected_clue_set(components, language, size):
        call_state["selection_sizes"].append(size)
        if size > len(base_clues) or size <= 0:
            return None
        if len(call_state["selection_sizes"]) == 1:
            return list(base_clues[:size])
        subset = [base_clues[0]]
        idx = 2
        while len(subset) < size and idx < len(base_clues):
            subset.append(base_clues[idx])
            idx += 1
        return subset

    def fake_generate_fill_in_puzzle(puzzle_id, theme, language, words, max_size=15):
        call_state["generate_calls"].append(list(words))
        call_number = len(call_state["generate_calls"])
        if call_number == 1:
            raise DisconnectedWordError("ALPHA")
        if call_number == 2:
            raise FillInGenerationError("attempt failed")
        if call_number == 3:
            raise DisconnectedWordError("ALPHA")
        call_state["final_words"] = list(words)
        return SimpleNamespace(
            id=puzzle_id,
            language=language,
            theme=theme,
            slots=[],
        )

    monkeypatch.setattr("app.generate_clues", fake_generate_clues)
    monkeypatch.setattr("app.validate_word_list", fake_validate_word_list)
    monkeypatch.setattr("app._build_word_components", fake_build_word_components)
    monkeypatch.setattr("app._select_connected_clue_set", fake_select_connected_clue_set)
    monkeypatch.setattr("app.generate_fill_in_puzzle", fake_generate_fill_in_puzzle)
    monkeypatch.setattr("app._assign_clues_to_slots", lambda *args, **kwargs: None)
    monkeypatch.setattr("app.save_puzzle", lambda *args, **kwargs: None)
    monkeypatch.setattr("app.puzzle_to_dict", lambda puzzle: {})
    monkeypatch.setattr("app._store_state", lambda *args, **kwargs: None)

    puzzle, state = _generate_puzzle(chat_id=5, language="en", theme="Retry")

    assert puzzle.language == "en"
    assert state.chat_id == 5
    assert call_state["replacement_calls"] == 2
    assert len(call_state["generate_calls"]) >= 4
    assert any("OMEGA" in words for words in call_state["generate_calls"][:2])
    assert "BETA" not in call_state["generate_calls"][2]
    assert isinstance(call_state["final_words"], list)
    assert "BETA" in call_state["final_words"]


def test_replacement_attempts_trigger_regeneration(monkeypatch) -> None:
    """Regenerate the clue list after exhausting replacement attempts."""

    base_clues = [
        WordClue(word="AAAA", clue="first"),
        WordClue(word="BBBB", clue="second"),
        WordClue(word="CCCCC", clue="third"),
        WordClue(word="DDDDD", clue="fourth"),
        WordClue(word="EEEEE", clue="fifth"),
        WordClue(word="FFFFF", clue="sixth"),
        WordClue(word="GGGGG", clue="seventh"),
        WordClue(word="HHHHH", clue="eighth"),
        WordClue(word="IIIII", clue="ninth"),
        WordClue(word="JJJJJ", clue="tenth"),
    ]
    refreshed_clues = [
        WordClue(word=f"NEW{i}", clue=f"clue {i}") for i in range(1, 11)
    ]

    call_counts = {"base": 0, "replacement": 0, "generate": 0}
    final_words: list[list[str]] = []

    def fake_generate_clues(
        theme: str,
        language: str,
        *,
        min_results: int = 10,
        max_results: int = 40,
    ):
        if "вместо" in theme:
            assert (min_results, max_results) == (6, 8)
            call_counts["replacement"] += 1
            return [WordClue(word="AAAA", clue="duplicate")]
        call_counts["base"] += 1
        assert (min_results, max_results) == (10, 40)
        if call_counts["base"] == 1:
            return base_clues
        assert call_counts["base"] == 2
        return refreshed_clues

    def fake_validate_word_list(language: str, clues, deduplicate: bool = True):
        return list(clues)

    def fake_build_word_components(clues, language):
        return [list(clues)]

    def fake_select_connected_clue_set(components, language, size):
        if not components:
            return None
        return list(components[0][:size])

    def fake_generate_fill_in_puzzle(puzzle_id, theme, language, words, max_size=15):
        call_counts["generate"] += 1
        if call_counts["generate"] == 1:
            raise DisconnectedWordError("AAAA")
        words_list = list(words)
        final_words.append(words_list)
        return SimpleNamespace(
            id=puzzle_id,
            language=language,
            theme=theme,
            slots=[],
        )

    monkeypatch.setattr("app.generate_clues", fake_generate_clues)
    monkeypatch.setattr("app.validate_word_list", fake_validate_word_list)
    monkeypatch.setattr("app._build_word_components", fake_build_word_components)
    monkeypatch.setattr("app._select_connected_clue_set", fake_select_connected_clue_set)
    monkeypatch.setattr("app.generate_fill_in_puzzle", fake_generate_fill_in_puzzle)
    monkeypatch.setattr("app._assign_clues_to_slots", lambda *args, **kwargs: None)
    monkeypatch.setattr("app.save_puzzle", lambda *args, **kwargs: None)
    monkeypatch.setattr("app.puzzle_to_dict", lambda puzzle: {})
    monkeypatch.setattr("app._store_state", lambda *args, **kwargs: None)

    puzzle, state = _generate_puzzle(chat_id=1, language="en", theme="Space")

    assert puzzle.language == "en"
    assert state.chat_id == 1
    assert call_counts["base"] == 2
    assert call_counts["replacement"] == MAX_REPLACEMENT_REQUESTS
    assert call_counts["generate"] >= 2
    assert final_words, "Expected successful generation after refresh"
    assert set(final_words[-1]) == {clue.word for clue in refreshed_clues}
