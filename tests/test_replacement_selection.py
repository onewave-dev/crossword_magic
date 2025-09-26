"""Tests ensuring replacement candidates respect intersection heuristics."""

from __future__ import annotations

from types import SimpleNamespace

from app import _generate_puzzle
from utils.fill_in_generator import DisconnectedWordError
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
        WordClue(word="ZZZZ", clue="no overlap"),
        WordClue(word="BOLT", clue="shares letters"),
    ]

    def fake_generate_clues(theme: str, language: str):
        if "вместо" in theme:
            return replacement_candidates
        return base_clues

    def fake_validate_word_list(language: str, clues, deduplicate: bool = True):
        return list(clues)

    call_state: dict[str, object] = {"count": 0, "words": None}

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
    assert puzzle.language == "en"
    assert game_state.chat_id == 1
