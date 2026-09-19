"""Board encoding for the xiangqi policy-value net.

15 planes of 10x9: 7 red piece types, 7 black piece types, 1 color plane
(1.0 = red to move). The board is left-right symmetric -> 2x augmentation.
"""
from __future__ import annotations

import numpy as np

from .board import Board, COLS, ROWS, EMPTY, BLACK

N_PLANES = 15


def encode(board: Board) -> np.ndarray:
    x = np.zeros((N_PLANES, ROWS, COLS), dtype=np.float32)
    for i, p in enumerate(board.cells):
        if p == EMPTY:
            continue
        r, c = RC_LOCAL(i)
        if p < 10:
            x[p - 1][r][c] = 1.0
        else:
            x[6 + (p - 10)][r][c] = 1.0
    if board.to_move != BLACK:          # red to move
        x[14][:] = 1.0
    return x


def RC_LOCAL(i: int) -> tuple[int, int]:
    return (i // COLS, i % COLS)


def flip_lr(x: np.ndarray, p_from: np.ndarray, p_to: np.ndarray):
    """Horizontal mirror (xiangqi has no other symmetry)."""
    return np.ascontiguousarray(x[..., ::-1]), p_from[::-1].copy(), p_to[::-1].copy()
