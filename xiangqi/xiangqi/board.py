"""Chinese Chess (Xiangqi) rules engine.

Board: 9 columns x 10 rows. Row 0 = black back rank, row 9 = red back rank.
Red moves first, red decreases row index. Cells: 0 empty, else piece code:
  red:   K1 A2 B3 (elephant) N4 R5 C6 P7
  black: 11 12 13 14 15 16 17  (type = code - 10)
Moves are (from_idx, to_idx) over 90 cells (idx = r * 9 + c).

Rules implemented: piece movement (incl. horse-leg / elephant-eye blocks,
cannon screen capture, pawn river crossing), palace confinement, flying
general (kings facing), check, checkmate, stalemate (no legal move = loss).
Not implemented (v1): perpetual check/chase judgement, 60-move rule.
"""
from __future__ import annotations

EMPTY = 0
K, A, B, N, R, C, P = 1, 2, 3, 4, 5, 6, 7           # red pieces
BK, BA, BB, BN, BR, BC, BP = 11, 12, 13, 14, 15, 16, 17  # black pieces
RED, BLACK = 1, 2

COLS, ROWS = 9, 10
IDX = lambda r, c: r * COLS + c
RC = lambda i: (i // COLS, i % COLS)

TYPE = lambda p: p if p < 10 else p - 10
SIDE = lambda p: RED if 0 < p < 10 else BLACK

PALACE = {RED: [(r, c) for r in (7, 8, 9) for c in (3, 4, 5)],
          BLACK: [(r, c) for r in (0, 1, 2) for c in (3, 4, 5)]}
PALACE_SET = {RED: set(PALACE[RED]), BLACK: set(PALACE[BLACK])}

START_ROWS = [
    "rnbakabnr",
    ".........",
    ".c.....c.",
    "p.p.p.p.p",
    ".........",
    ".........",
    "P.P.P.P.P",
    ".C.....C.",
    ".........",
    "RNBAKABNR",
]
CHAR = {"K": K, "A": A, "B": B, "N": N, "R": R, "C": C, "P": P,
        "k": BK, "a": BA, "b": BB, "n": BN, "r": BR, "c": BC, "p": BP}


class Board:
    __slots__ = ("cells", "to_move", "history", "pos_keys")

    def __init__(self, setup: list[str] | None = None):
        self.cells = [EMPTY] * (ROWS * COLS)
        if setup is None:
            setup = START_ROWS
        for r, row in enumerate(setup):
            for c, ch in enumerate(row):
                if ch in CHAR:
                    self.cells[IDX(r, c)] = CHAR[ch]
        self.to_move = RED
        self.history = []
        self.pos_keys = [self._pos_key()]

    def _pos_key(self) -> tuple:
        return (tuple(self.cells), self.to_move)

    def clone(self) -> "Board":
        b = Board.__new__(Board)
        b.cells = self.cells[:]
        b.to_move = self.to_move
        b.history = list(self.history)
        b.pos_keys = list(self.pos_keys)
        return b

    def play(self, mv: tuple[int, int]) -> None:
        f, t = mv
        self.history.append((f, t, self.cells[t]))
        self.cells[t] = self.cells[f]
        self.cells[f] = EMPTY
        self.to_move = BLACK if self.to_move == RED else RED
        self.pos_keys.append(self._pos_key())

    def undo(self) -> None:
        f, t, captured = self.history.pop()
        self.cells[f] = self.cells[t]
        self.cells[t] = captured
        self.to_move = BLACK if self.to_move == RED else RED
        self.pos_keys.pop()

    def is_repetition_draw(self, threshold: int = 3) -> bool:
        """Same position (pieces + side to move) occurred `threshold` times."""
        return self.pos_keys.count(self._pos_key()) >= threshold

    def king(self, side: int) -> int:
        target = K if side == RED else BK
        for i, p in enumerate(self.cells):
            if p == target:
                return i
        return -1

    def in_check(self, side: int) -> bool:
        """True if `side`'s king is attacked (incl. flying general)."""
        k = self.king(side)
        if k < 0:
            return True
        kr, kc = RC(k)
        # flying general: opposing king on same file with nothing between
        ek = self.king(other(side))
        if ek >= 0:
            er, ec = RC(ek)
            if ec == kc and all(self.cells[IDX(r, kc)] == EMPTY
                                for r in range(min(er, kr) + 1, max(er, kr))):
                return True
        # attacked by any enemy pseudo-move targeting k
        for i, p in enumerate(self.cells):
            if p != EMPTY and SIDE(p) != side:
                if _attacks(self, i, k, p):
                    return True
        return False

    def legal_moves(self, side: int | None = None) -> list[tuple[int, int]]:
        side = side if side is not None else self.to_move
        out = []
        for i, p in enumerate(self.cells):
            if p != EMPTY and SIDE(p) == side:
                for t in pseudo_moves(self, i, p):
                    self.play((i, t))
                    ok = not self.in_check(side)
                    self.undo()
                    if ok:
                        out.append((i, t))
        return out

    def status(self) -> int:
        """0 ongoing, RED/BLACK = winner, -1 draw (never in v1 rules)."""
        moves = self.legal_moves()
        if not moves:
            return other(self.to_move)   # checkmate or stalemate: mover loses
        if self.king(RED) < 0 or self.king(BLACK) < 0:
            return other(self.to_move)   # king captured (shouldn't happen)
        return 0


def other(side: int) -> int:
    return BLACK if side == RED else RED


def _attacks(b: "Board", src: int, dst: int, piece: int) -> bool:
    """Does the piece at src attack dst (ignoring king-safety)?"""
    sr, sc = RC(src)
    t = TYPE(piece)
    if t == K:
        return max(abs(RC(dst)[0] - sr), abs(RC(dst)[1] - sc)) == 1
    if t in (N, A, B, P):
        # horse/elephant/advisor/pawn attack exactly where they can move
        return dst in pseudo_moves(b, src, piece)
    # rook: slide; cannon: attack = screen capture
    return dst in _slide(b, src, piece, capture_only=True)


def _slide(b: "Board", src: int, piece: int, capture_only: bool) -> list[int]:
    """Rook-like sliding for R; C = slide for quiet, screen-capture for capture."""
    sr, sc = RC(src)
    t = TYPE(piece)
    out = []
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        rr, sc2 = sr + dr, sc + dc
        jumped = False
        while 0 <= rr < ROWS and 0 <= sc2 < COLS:
            i = IDX(rr, sc2)
            v = b.cells[i]
            if t == R:
                if v == EMPTY:
                    out.append(i)
                else:
                    if SIDE(v) != SIDE(piece):
                        out.append(i)
                    break
            else:  # cannon
                if not jumped:
                    if v == EMPTY:
                        if not capture_only:
                            out.append(i)
                    else:
                        jumped = True
                else:
                    if v != EMPTY:
                        if SIDE(v) != SIDE(piece):
                            out.append(i)
                        break
            rr += dr
            sc2 += dc
    return out


def pseudo_moves(b: "Board", src: int, piece: int) -> list[int]:
    sr, sc = RC(src)
    side = SIDE(piece)
    t = TYPE(piece)
    out: list[int] = []

    def add(r, c):
        if 0 <= r < ROWS and 0 <= c < COLS:
            i = IDX(r, c)
            v = b.cells[i]
            if v == EMPTY or SIDE(v) != side:
                out.append(i)

    if t == K:
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            r, c = sr + dr, sc + dc
            if (r, c) in PALACE_SET[side]:
                add(r, c)
    elif t == A:
        for dr, dc in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
            r, c = sr + dr, sc + dc
            if (r, c) in PALACE_SET[side]:
                add(r, c)
    elif t == B:
        for dr, dc in ((2, 2), (2, -2), (-2, 2), (-2, -2)):
            r, c = sr + dr, sc + dc
            if not (0 <= r < ROWS and 0 <= c < COLS):
                continue
            if side == RED and r < 5:
                continue
            if side == BLACK and r > 4:
                continue
            if b.cells[IDX(sr + dr // 2, sc + dc // 2)] != EMPTY:
                continue  # elephant eye blocked
            add(r, c)
    elif t == N:
        for dr, dc, lr, lc in ((2, 1, 1, 0), (2, -1, 1, 0), (-2, 1, -1, 0), (-2, -1, -1, 0),
                               (1, 2, 0, 1), (-1, 2, 0, 1), (1, -2, 0, -1), (-1, -2, 0, -1)):
            r, c = sr + dr, sc + dc
            if not (0 <= r < ROWS and 0 <= c < COLS):
                continue
            if b.cells[IDX(sr + lr, sc + lc)] != EMPTY:
                continue  # horse leg blocked
            add(r, c)
    elif t in (R, C):
        out.extend(_slide(b, src, piece, capture_only=False))
    elif t == P:
        fwd = -1 if side == RED else 1
        add(sr + fwd, sc)
        crossed = (sr <= 4) if side == RED else (sr >= 5)
        if crossed:
            add(sr, sc - 1)
            add(sr, sc + 1)
    return out
