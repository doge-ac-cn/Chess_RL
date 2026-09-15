"""Gomoku board core: rules, legality, win detection, candidate/threat queries.

Free-style rules (no renju forbidden moves): 5 or more in a row wins,
board full = draw. Cells: 0 empty, 1 black, 2 white. Black moves first.
"""
from __future__ import annotations

import numpy as np

EMPTY, BLACK, WHITE = 0, 1, 2
DIRS = ((0, 1), (1, 0), (1, 1), (1, -1))
DRAW = -1
ONGOING = 0


def other(p: int) -> int:
    return BLACK + WHITE - p


class Board:
    __slots__ = ("n", "cells", "to_move", "last", "history")

    def __init__(self, n: int = 15):
        self.n = n
        self.cells = np.zeros((n, n), dtype=np.int8)
        self.to_move = BLACK
        self.last = None          # (r, c) of last placed stone
        self.history = []

    def clone(self) -> "Board":
        b = Board.__new__(Board)
        b.n = self.n
        b.cells = self.cells.copy()
        b.to_move = self.to_move
        b.last = self.last
        b.history = list(self.history)
        return b

    def reset(self) -> None:
        self.cells[:] = EMPTY
        self.to_move = BLACK
        self.last = None
        self.history.clear()

    def legal(self, r: int, c: int) -> bool:
        return 0 <= r < self.n and 0 <= c < self.n and self.cells[r, c] == EMPTY

    def play(self, r: int, c: int) -> None:
        if not self.legal(r, c):
            raise ValueError(f"illegal move ({r},{c})")
        self.cells[r, c] = self.to_move
        self.history.append((r, c))
        self.last = (r, c)
        self.to_move = other(self.to_move)

    def undo(self) -> None:
        r, c = self.history.pop()
        self.cells[r, c] = EMPTY
        self.to_move = other(self.to_move)
        self.last = self.history[-1] if self.history else None

    def is_full(self) -> bool:
        return bool((self.cells == EMPTY).sum() == 0)

    def winner_after(self, r: int, c: int) -> int:
        """Side that just completed >=5 in a row by placing at (r,c); 0 if none."""
        p = int(self.cells[r, c])
        if p == EMPTY:
            return EMPTY
        n = self.n
        for dr, dc in DIRS:
            cnt = 1
            for s in (1, -1):
                rr, cc = r + dr * s, c + dc * s
                while 0 <= rr < n and 0 <= cc < n and self.cells[rr, cc] == p:
                    cnt += 1
                    rr += dr * s
                    cc += dc * s
            if cnt >= 5:
                return p
        return EMPTY

    def status(self) -> int:
        """WINNING side, DRAW, or ONGOING."""
        if self.last is not None:
            w = self.winner_after(*self.last)
            if w:
                return w
        if (self.cells == EMPTY).sum() == 0:
            return DRAW
        return ONGOING

    def key(self) -> bytes:
        return self.cells.tobytes() + bytes((self.to_move,))

    # ---------------- move candidate / threat queries ----------------

    def candidates(self, radius: int = 2) -> list[tuple[int, int]]:
        """Empty cells within Chebyshev `radius` of any stone; center if board empty."""
        n = self.n
        empt = np.argwhere(self.cells == EMPTY)
        if len(empt) == 0:
            return []
        if self.history:
            stones = np.argwhere(self.cells != EMPTY)
            dist = np.abs(empt[:, None, :] - stones[None, :, :]).max(axis=2).min(axis=1)
            keep = empt[dist <= radius]
            return [(int(r), int(c)) for r, c in keep]
        m = n // 2
        return [(m, m)]

    def five_points(self, p: int) -> list[tuple[int, int]]:
        """Empty cells where placing p immediately makes >=5 (winning points)."""
        pts = []
        for r, c in self.candidates(radius=1):
            n = self.n
            for dr, dc in DIRS:
                cnt = 1
                for s in (1, -1):
                    rr, cc = r + dr * s, c + dc * s
                    while 0 <= rr < n and 0 <= cc < n and self.cells[rr, cc] == p:
                        cnt += 1
                        rr += dr * s
                        cc += dc * s
                if cnt >= 5:
                    pts.append((r, c))
                    break
        return pts

    def four_completions(self, r: int, c: int, p: int) -> set[tuple[int, int]]:
        """If p played at (r,c): empty points that would complete a five
        (i.e. the forcing points of the four(s) created). >=2 points means
        an unstoppable four (straight four or double four)."""
        n = self.n
        self.cells[r, c] = p
        pts: set[tuple[int, int]] = set()
        for dr, dc in DIRS:
            for k in range(-4, 1):
                sr, sc = r + k * dr, c + k * dc
                mine, empty = 0, None
                ok = True
                for t in range(5):
                    rr, cc = sr + t * dr, sc + t * dc
                    if not (0 <= rr < n and 0 <= cc < n):
                        ok = False
                        break
                    v = self.cells[rr, cc]
                    if v == other(p):
                        ok = False
                        break
                    if v == p:
                        mine += 1
                    else:
                        empty = (rr, cc)
                if ok and mine == 4 and empty is not None:
                    pts.add(empty)
        self.cells[r, c] = EMPTY
        return pts


# ---------------- encoding / symmetry ----------------

def encode(board: Board) -> np.ndarray:
    """(4, n, n) float32 planes: my stones, opp stones, last move, black-to-move."""
    n = board.n
    x = np.zeros((4, n, n), dtype=np.float32)
    me, opp = board.to_move, other(board.to_move)
    x[0] = board.cells == me
    x[1] = board.cells == opp
    if board.last is not None:
        x[2][board.last] = 1.0
    x[3] = 1.0 if me == BLACK else 0.0
    return x


def apply_symmetry(x: np.ndarray, pi: np.ndarray, k: int, flip: bool):
    """Apply one of the 8 dihedral transforms to input planes and policy grid."""
    x2 = np.rot90(x, k, axes=(1, 2))
    p2 = np.rot90(pi, k)
    if flip:
        x2 = np.flip(x2, axis=2)
        p2 = np.flip(p2, axis=1)
    return np.ascontiguousarray(x2), np.ascontiguousarray(p2)
