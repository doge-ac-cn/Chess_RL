"""Baseline players for xiangqi: random + material/position heuristic teacher."""
from __future__ import annotations

import random

from .board import (Board, IDX, RC, RED, BLACK, EMPTY, other, TYPE, SIDE,
                    K, A, B, N, R, C, P)

# material values (centipawn-ish)
VALUE = {K: 100000, R: 900, C: 450, N: 400, B: 200, A: 200, P: 100}

# simple piece-square bonus for advancing/crossing pawns and central horses
def _pst(b: Board, i: int, piece: int) -> int:
    t = TYPE(piece)
    side = SIDE(piece)
    r, c = RC(i)
    if t == P:
        crossed = (r <= 4) if side == RED else (r >= 5)
        depth = (9 - r) if side == RED else r
        return 30 if crossed else 6 * depth
    if t == N or t == C:
        center = 4 - abs(c - 4)
        return 6 * center
    if t == K:
        return 0
    return 0


def move_score(b: Board, mv: tuple[int, int], side: int) -> int:
    f, t = mv
    piece = b.cells[f]
    score = VALUE.get(TYPE(b.cells[t]), 0) * 10 if b.cells[t] != EMPTY else 0
    score += _pst(b, t, piece) - _pst(b, f, piece)
    # prefer putting opponent under pressure: count their replies (tempo proxy)
    b.play(mv)
    reply = len(b.legal_moves())
    b.undo()
    score -= 2 * reply
    return score


class HeuristicPlayer:
    """1-ply greedy material/position teacher."""

    def __init__(self, temperature: float = 0.0, rng: random.Random | None = None):
        self.temperature = temperature
        self.rng = rng or random.Random(0)

    def move(self, board: Board) -> tuple[int, int]:
        moves = board.legal_moves()
        side = board.to_move
        scored = sorted(((move_score(board, m, side), m) for m in moves),
                        key=lambda x: -x[0])
        if self.temperature <= 0:
            top = scored[0][0]
            return self.rng.choice([m for s, m in scored if s >= top])
        mx = max(s for s, _ in scored)
        exp = [2.718281828 ** ((s - mx) / (400.0 * self.temperature)) for s, _ in scored]
        total = sum(exp)
        pick = self.rng.random() * total
        acc = 0.0
        for (s, m), w in zip(scored, exp):
            acc += w
            if acc >= pick:
                return m
        return scored[-1][1]


class RandomPlayer:
    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)

    def move(self, board: Board) -> tuple[int, int]:
        return self.rng.choice(board.legal_moves())
