"""Utilities for maintaining integrity of a player's ship cells before rendering."""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from typing import Any, MutableSequence


logger = logging.getLogger(__name__)


CellMatrix = MutableSequence[MutableSequence[int]]
OwnerMatrix = MutableSequence[MutableSequence[Any]]


def ensure_player_ship_cells(
    *,
    match: Any,
    player_key: Any,
    view_board: CellMatrix,
    view_owners: OwnerMatrix,
    expected_ship_cells: int,
    iter_reference_ship_cells: Callable[[Any], Iterable[tuple[int, int, int]]],
    get_cell_state: Callable[[Any], int],
) -> bool:
    """Ensure the rendered board contains exactly ``expected_ship_cells`` for the player.

    The function mirrors the safety net used by the production router. It first restores
    missing cells from the historical snapshot/reference, then falls back to the live
    board if necessary. If, after both recovery passes, the number of ship cells still
    does not match the expectation, the render should be aborted.

    Parameters
    ----------
    match:
        Live game match object providing the authoritative player grids via
        ``match.boards[player_key].grid``.
    player_key:
        Identifier of the player whose ships must be present on the rendered view.
    view_board / view_owners:
        Mutable matrices representing the pending render (values are mutated in place).
    expected_ship_cells:
        The expected number of ship cells that must be visible for the player.
    iter_reference_ship_cells:
        Callable returning an iterable of ``(row, col, state)`` triples from the
        reference snapshot/history.
    get_cell_state:
        Callable that converts a live grid cell value to the canonical state code
        (1, 3 or 4 for ships).

    Returns
    -------
    bool
        ``True`` when the expected number of cells is achieved, ``False`` otherwise.
    """

    def _player_ship_cells_count() -> int:
        return sum(
            1
            for rr, (row_states, row_owners) in enumerate(zip(view_board, view_owners))
            for cc, (state, owner) in enumerate(zip(row_states, row_owners))
            if owner == player_key and state in {1, 3, 4}
        )

    current_ship_cells = _player_ship_cells_count()
    if current_ship_cells < expected_ship_cells:
        missing = expected_ship_cells - current_ship_cells

        # 1) Try restoring from snapshot/history first.
        restored_from_ref: list[tuple[int, int, int]] = []
        for rr, cc, state in iter_reference_ship_cells(player_key):
            if view_owners[rr][cc] == player_key and view_board[rr][cc] in {1, 3, 4}:
                continue
            view_board[rr][cc] = state
            view_owners[rr][cc] = player_key
            restored_from_ref.append((rr, cc, state))
            missing -= 1
            if missing == 0:
                break
        if restored_from_ref:
            logger.warning(
                "Restored %d ship cells for %s from reference snapshot/history: %s",
                len(restored_from_ref),
                player_key,
                restored_from_ref,
            )

        # 2) If still missing cells, fill from the live player board.
        if missing > 0:
            own_live = match.boards[player_key].grid
            live_added: list[tuple[int, int, int]] = []
            for rr in range(15):
                for cc in range(15):
                    st = get_cell_state(own_live[rr][cc])
                    if st not in {1, 3, 4}:
                        continue
                    if (
                        view_owners[rr][cc] == player_key
                        and view_board[rr][cc] in {1, 3, 4}
                    ):
                        continue
                    view_board[rr][cc] = st
                    view_owners[rr][cc] = player_key
                    live_added.append((rr, cc, st))
                    missing -= 1
                    if missing == 0:
                        break
                if missing == 0:
                    break
            if live_added:
                logger.warning(
                    "Restored %d ship cells for %s from live grid: %s",
                    len(live_added),
                    player_key,
                    live_added,
                )

        # 3) Control: after both sources we must have exactly the expected number.
        current_ship_cells = _player_ship_cells_count()
        if current_ship_cells != expected_ship_cells:
            logger.error(
                "Unable to reach exactly %d ship cells for %s before rendering: got %d",
                expected_ship_cells,
                player_key,
                current_ship_cells,
            )
            return False
    elif current_ship_cells > expected_ship_cells:
        logger.error(
            "Too many ship cells for %s before rendering: %d (expected %d)",
            player_key,
            current_ship_cells,
            expected_ship_cells,
        )
        return False

    return True

