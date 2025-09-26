from app import _build_word_components, _resolve_slot
from utils.crossword import CompositeComponent, CompositePuzzle
from utils.fill_in_generator import generate_fill_in_puzzle
from utils.llm_generator import WordClue


def test_build_word_components_detects_disconnected_groups():
    clues = [
        WordClue(word="cat", clue=""),
        WordClue(word="tack", clue=""),
        WordClue(word="dog", clue=""),
        WordClue(word="good", clue=""),
    ]

    components = _build_word_components(clues, "en")
    assert len(components) == 2
    words_by_component = [sorted(clue.word for clue in group) for group in components]
    assert sorted(words_by_component) == [["cat", "tack"], ["dog", "good"]]


def test_resolve_slot_requires_component_suffix():
    puzzle1 = generate_fill_in_puzzle("p1", "theme", "en", ["alpha"])
    puzzle2 = generate_fill_in_puzzle("p2", "theme", "en", ["omega"])
    composite = CompositePuzzle(
        id="comp",
        theme="theme",
        language="en",
        components=[
            CompositeComponent(index=1, puzzle=puzzle1, row_offset=0, col_offset=0),
            CompositeComponent(index=2, puzzle=puzzle2, row_offset=puzzle1.size_rows + 1, col_offset=0),
        ],
        gap_cells=1,
    )

    slot_ref, message = _resolve_slot(composite, "A1")
    assert slot_ref is None
    assert message is not None
    assert "A1-1" in message and "A1-2" in message

    slot_ref, message = _resolve_slot(composite, "A1-2")
    assert message is None
    assert slot_ref is not None
    assert slot_ref.component_index == 2
    assert slot_ref.slot.slot_id == "A1"
