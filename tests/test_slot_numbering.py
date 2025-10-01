from app import _apply_slot_mapping_to_state
from utils.crossword import Direction, Puzzle, renumber_slots
from utils.storage import GameState


def test_down_slots_numbered_by_row_then_column() -> None:
    puzzle = Puzzle.from_size(
        puzzle_id="p1",
        theme="test",
        language="en",
        rows=3,
        cols=3,
        block_positions=[(0, 1)],
    )

    down_slots = [slot for slot in puzzle.slots if slot.direction is Direction.DOWN]
    assert [(slot.slot_id, slot.start_row, slot.start_col) for slot in down_slots] == [
        ("D1", 0, 0),
        ("D2", 0, 2),
        ("D3", 1, 1),
    ]


def test_renumber_slots_returns_mapping_for_updated_identifiers() -> None:
    puzzle = Puzzle.from_size(
        puzzle_id="p2",
        theme="test",
        language="en",
        rows=3,
        cols=3,
        block_positions=[(0, 1)],
    )

    # Simulate legacy numbering for the first down slot
    down_slot = next(slot for slot in puzzle.slots if slot.direction is Direction.DOWN)
    down_slot.slot_id = "D99"
    down_slot.number = 99

    mapping = renumber_slots(puzzle)

    assert mapping == {"D99": "D1"}
    assert down_slot.slot_id == "D1"


def test_apply_slot_mapping_to_state_updates_identifiers() -> None:
    state = GameState(
        chat_id=1,
        puzzle_id="p",
        solved_slots={"D99"},
        hints_used={"D99": {123: 2}},
        active_slot_id="d99",
    )

    _apply_slot_mapping_to_state(state, {"D99": "D1"})

    assert state.solved_slots == {"D1"}
    assert state.active_slot_id == "D1"
    assert state.hints_used == {"D1": {123: 2}}
