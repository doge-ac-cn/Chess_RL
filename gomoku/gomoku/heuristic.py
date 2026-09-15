"""Rule-based heuristic player (pattern scoring). Serves as the BC teacher
and the main strength baseline for evaluation."""
from __future__ import annotations

import random

from .board import Board, DIRS, EMPTY, other

# Pattern scores for a 9-cell window string around a candidate cell.
# '1' = mover stone (candidate counted as placed), '0' = empty, '2' = opponent/wall.
# Order matters: first match wins (patterns checked strongest-first).
PATTERNS = [
    ("11111", 10000000),                                     # five
    ("011110", 300000),                                      # live four
    ("211110", 25000), ("011112", 25000),                    # rush four (blocked one end)
    ("11011", 25000), ("10111", 25000), ("11101", 25000),    # split four
    ("011100", 12000), ("001110", 12000),                    # live three
    ("010110", 9000), ("011010", 9000),                      # broken live three
    ("211100", 800), ("001112", 800), ("211010", 800), ("010112", 800),
    ("210110", 800), ("011012", 800),                        # sleep three
    ("10011", 800), ("11001", 800), ("10101", 800),
    ("001100", 300), ("010100", 300), ("001010", 300),       # live two
    ("211000", 60), ("000112", 60), ("210100", 60), ("001012", 60),
    ("210010", 60), ("010012", 60), ("10001", 60),           # sleep two
]


def _window(board: Board, r: int, c: int, dr: int, dc: int, mover: int) -> str:
    """9-char window centered at (r,c), with (r,c) counted as a mover stone."""
    n, cells = board.n, board.cells
    out = []
    for k in range(-4, 5):
        rr, cc = r + k * dr, c + k * dc
        if k == 0:
            out.append("1")
        elif 0 <= rr < n and 0 <= cc < n:
            v = cells[rr, cc]
            out.append("." if v == EMPTY else ("1" if v == mover else "2"))
        else:
            out.append("2")
    return "".join(out)


def cell_score(board: Board, r: int, c: int, mover: int) -> int:
    """Best pattern value per direction, summed across the 4 directions."""
    total = 0
    for dr, dc in DIRS:
        w = _window(board, r, c, dr, dc, mover)
        best = 0
        for pat, sc in PATTERNS:
            if pat in w:
                best = sc
                break
        total += best
    return total


def move_score(board: Board, r: int, c: int, player: int) -> int:
    """Attack value plus discounted defensive value."""
    atk = cell_score(board, r, c, player)
    dfn = cell_score(board, r, c, other(player))
    return atk + int(dfn * 0.8)


class HeuristicPlayer:
    """1-ply pattern agent. Decent intermediate-human strength baseline.

    temperature=0 -> greedy (deterministic apart from tie shuffling seeded
    externally); >0 -> sample among scored moves for BC data diversity.
    """

    def __init__(self, temperature: float = 0.0, rng: random.Random | None = None):
        self.temperature = temperature
        self.rng = rng or random.Random(0)

    def scored_moves(self, board: Board) -> list[tuple[int, tuple[int, int]]]:
        p = board.to_move
        return [(move_score(board, r, c, p), (r, c)) for r, c in board.candidates(radius=2)]

    def move(self, board: Board) -> tuple[int, int]:
        p = board.to_move
        wins = board.five_points(p)
        if wins:
            return wins[0]
        opp_wins = board.five_points(other(p))
        if opp_wins:
            return opp_wins[0]
        scored = self.scored_moves(board)
        if not scored:
            raise RuntimeError("no legal moves")
        scored.sort(key=lambda t: -t[0])
        if self.temperature <= 0.0:
            top = scored[0][0]
            best = [m for s, m in scored if s >= top]
            return self.rng.choice(best)
        exp = [(s / 8000.0) ** (1.0 / self.temperature) for s, _ in scored]
        total = sum(exp)
        pick = self.rng.random() * total
        acc = 0.0
        for (s, m), w in zip(scored, exp):
            acc += w
            if acc >= pick:
                return m
        return scored[-1][1]
