"""Go rules engine (Chinese area scoring, simple ko).

Board sizes: any (9/13/19). Cells: 0 empty, 1 black, 2 white. Pass = None.
Move: (r, c) tuple. Rules: capture on placement, suicide forbidden, simple ko
(point that may not be immediately replayed), two consecutive passes end the
game, Chinese area scoring (stones + surrounded empty regions) + 7.5 komi.
"""
from __future__ import annotations

EMPTY, BLACK, WHITE = 0, 1, 2
KOMI = 7.5
PASS = None


def other(c: int) -> int:
    return BLACK + WHITE - c


class Board:
    __slots__ = ("n", "cells", "to_move", "history", "ko_point", "passes")

    def __init__(self, n: int = 9):
        self.n = n
        self.cells = [EMPTY] * (n * n)
        self.to_move = BLACK
        self.history = []
        self.ko_point: int | None = None
        self.passes = 0

    def clone(self) -> "Board":
        b = Board.__new__(Board)
        b.n = self.n
        b.cells = self.cells[:]
        b.to_move = self.to_move
        b.history = list(self.history)
        b.ko_point = self.ko_point
        b.passes = self.passes
        return b

    def _neighbors(self, i: int):
        r, c = divmod(i, self.n)
        if r > 0: yield i - self.n
        if r < self.n - 1: yield i + self.n
        if c > 0: yield i - 1
        if c < self.n - 1: yield i + 1

    def _group_and_liberties(self, i: int) -> tuple[set[int], set[int]]:
        color = self.cells[i]
        stack, group, libs = [i], {i}, set()
        while stack:
            j = stack.pop()
            for nb in self._neighbors(j):
                v = self.cells[nb]
                if v == EMPTY:
                    libs.add(nb)
                elif v == color and nb not in group:
                    group.add(nb)
                    stack.append(nb)
        return group, libs

    def play(self, mv) -> None:
        """Place at (r,c) or PASS. Raises ValueError on illegal moves."""
        if mv is PASS:
            self.history.append((None, self.to_move, 0))
            self.passes += 1
            self.ko_point = None
            self.to_move = other(self.to_move)
            return
        r, c = mv
        i = r * self.n + c
        if self.cells[i] != EMPTY:
            raise ValueError(f"occupied: {mv}")
        if self.ko_point == i:
            raise ValueError(f"ko recapture forbidden: {mv}")

        me = self.to_move
        opp = other(me)
        self.cells[i] = me

        # remove opponent groups without liberties
        captured = 0
        cap_last = -1
        for nb in self._neighbors(i):
            if self.cells[nb] == opp:
                group, libs = self._group_and_liberties(nb)
                if not libs:
                    for j in group:
                        self.cells[j] = EMPTY
                    captured += len(group)
                    cap_last = next(iter(group))
        # undo-shape for ko detection: remember if we captured exactly one
        single_capture = captured == 1

        # suicide check
        group, libs = self._group_and_liberties(i)
        if not libs:
            self.cells[i] = EMPTY
            raise ValueError(f"suicide: {mv}")

        # simple ko: single stone captured by a single stone with one liberty
        if single_capture and len(group) == 1 and len(libs) == 1:
            self.ko_point = cap_last
        else:
            self.ko_point = None

        self.history.append(((r, c), me, captured))
        self.passes = 0
        self.to_move = opp

    def legal_moves(self) -> list[tuple[int, int]]:
        out = []
        for i, v in enumerate(self.cells):
            if v != EMPTY or (self.ko_point == i):
                continue
            # fast suicide check by simulation on a copy
            r, c = divmod(i, self.n)
            b = self.clone()
            try:
                b.play((r, c))
                out.append((r, c))
            except ValueError:
                pass
        return out

    def is_over(self) -> bool:
        return self.passes >= 2

    def score(self) -> tuple[float, float]:
        """Chinese area scoring -> (black, white) including komi for white."""
        black = sum(1 for v in self.cells if v == BLACK)
        white = sum(1 for v in self.cells if v == WHITE)
        seen = set()
        for i, v in enumerate(self.cells):
            if v != EMPTY or i in seen:
                continue
            region, border = {i}, set()
            stack = [i]
            while stack:
                j = stack.pop()
                for nb in self._neighbors(j):
                    if self.cells[nb] == EMPTY:
                        if nb not in region:
                            region.add(nb)
                            stack.append(nb)
                    else:
                        border.add(self.cells[nb])
            seen |= region
            if border == {BLACK}:
                black += len(region)
            elif border == {WHITE}:
                white += len(region)
        return float(black), white + KOMI
