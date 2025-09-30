import pytest

from app import _canonical_letter_set, _generate_puzzle
from utils.crossword import Puzzle
from utils.llm_generator import WordClue


def _connected(words: list[str], language: str) -> bool:
    if not words:
        return False
    letter_sets = [_canonical_letter_set(word, language) for word in words]
    graph: dict[int, set[int]] = {idx: set() for idx in range(len(words))}
    for i in range(len(words)):
        for j in range(i + 1, len(words)):
            if letter_sets[i] & letter_sets[j]:
                graph[i].add(j)
                graph[j].add(i)
    visited: set[int] = set()
    stack = [0]
    while stack:
        node = stack.pop()
        if node in visited:
            continue
        visited.add(node)
        stack.extend(graph[node] - visited)
    return len(visited) == len(words)


def test_generate_puzzle_uses_connected_subset(monkeypatch: pytest.MonkeyPatch) -> None:
    def make_connected_word(index: int) -> WordClue:
        pattern = format(index, "05b")
        letters = ["A" if bit == "1" else "B" for bit in pattern]
        return WordClue(word="AB" + "".join(letters), clue=f"Signal {index}")

    connected_cluster = [make_connected_word(idx) for idx in range(30)]
    disconnected_order = [
        WordClue(word="DOG", clue="Canine"),
        WordClue(word="WOLF", clue="Wild canine"),
        WordClue(word="LYNX", clue="Cat"),
        *connected_cluster,
    ]

    target_component = {clue.word for clue in connected_cluster}
    captured_words: list[str] = []

    def fake_generate_clues(
        theme: str,
        language: str,
        *,
        min_results: int = 10,
        max_results: int = 40,
    ) -> list[WordClue]:
        assert (min_results, max_results) == (10, 40)
        return disconnected_order

    def fake_validate_word_list(
        language: str, clues, deduplicate: bool = True
    ) -> list[WordClue]:
        return list(disconnected_order)

    def fake_generate_fill_in_puzzle(
        *,
        puzzle_id: str,
        theme: str,
        language: str,
        words: list[str],
        max_size: int,
    ) -> Puzzle:
        captured_words[:] = list(words)
        return Puzzle.from_size(puzzle_id, theme, language, 5, 5)

    monkeypatch.setattr("app.generate_clues", fake_generate_clues)
    monkeypatch.setattr("app.validate_word_list", fake_validate_word_list)
    monkeypatch.setattr("app.generate_fill_in_puzzle", fake_generate_fill_in_puzzle)
    monkeypatch.setattr("app.save_puzzle", lambda *args, **kwargs: None)
    monkeypatch.setattr("app.save_state", lambda *args, **kwargs: None)
    monkeypatch.setattr("app._store_state", lambda game_state: None)

    puzzle, game_state = _generate_puzzle(chat_id=123, language="en", theme="test")

    assert puzzle.language == "en"
    assert captured_words
    assert set(captured_words) == target_component
    assert _connected(captured_words, "en"), "Selected words must form a connected set"
    assert game_state.puzzle_id == puzzle.id
