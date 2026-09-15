"""Plain-assert test suite: run `python3 tests/test_gomoku.py`."""
import random
import sys
import time

sys.path.insert(0, ".")

import numpy as np

from gomoku.board import Board, BLACK, WHITE, DRAW, other, encode, apply_symmetry
from gomoku.heuristic import HeuristicPlayer
from gomoku.vcf import vcf_search


def place(b: Board, moves, color=None):
    for m in moves:
        b.play(*m)


def test_win_detection():
    for moves in (
        [(7, 3), (7, 4), (7, 5), (7, 6), (7, 7)],            # horizontal
        [(3, 7), (4, 7), (5, 7), (6, 7), (7, 7)],            # vertical
        [(3, 3), (4, 4), (5, 5), (6, 6), (7, 7)],            # diag down-right
        [(3, 11), (4, 10), (5, 9), (6, 8), (7, 7)],          # diag down-left
        [(7, 3), (7, 4), (7, 5), (7, 6), (7, 7), (7, 8)],    # overline
    ):
        b = Board(15)
        seq = []
        for i, m in enumerate(moves):
            seq.append(m)
            seq.append((0, i))                               # dummy white stones
        won = 0
        for mv in seq:
            b.play(*mv)
            st = b.status()
            if st:
                won = st
                break
        assert won == BLACK, f"black should win: {moves}, got {won}"
    # four in a row is NOT a win
    b = Board(15)
    place(b, [(7, 3), (0, 0), (7, 4), (0, 1), (7, 5), (0, 2), (7, 6), (0, 3)])
    assert b.status() == 0


def test_draw_and_turns():
    b = Board(3)  # 3x3 board can fill -> draw only if no line; 3x3 with 5 needed can't win
    for r in range(3):
        for c in range(3):
            b.play(r, c)
    assert b.status() == DRAW


def test_five_points_and_four_completions():
    b = Board(15)
    place(b, [(7, 5), (0, 0), (7, 6), (0, 1), (7, 7), (0, 2), (7, 8), (0, 3)])  # black four
    fp = set(b.five_points(BLACK))
    assert fp == {(7, 4), (7, 9)}, fp
    # four_completions only counts windows that CONTAIN the new stone;
    # placing (7,4) creates the four (7,3)..(7,7) with completion (7,3)
    comps = b.four_completions(7, 4, BLACK)
    assert comps == {(7, 3)}, comps
    # rush four: blocked one side -> single completion
    b2 = Board(15)
    place(b2, [(7, 5), (0, 0), (7, 6), (0, 1), (7, 7), (0, 2), (7, 8), (7, 9)])
    assert set(b2.five_points(BLACK)) == {(7, 4)}


def test_heuristic_finds_win_and_block():
    b = Board(15)
    # black rush four (7,5)..(7,8), right end blocked by white at (7,9)
    place(b, [(7, 5), (0, 0), (7, 6), (0, 1), (7, 7), (0, 2), (7, 8), (7, 9)])
    # white to move: single five point (7,4) must be blocked
    mv = HeuristicPlayer().move(b)
    assert mv == (7, 4), mv
    # black to move in the live-four version: take either five point
    b2 = Board(15)
    place(b2, [(7, 5), (0, 0), (7, 6), (0, 1), (7, 7), (0, 2), (7, 8), (0, 3)])
    mv = HeuristicPlayer().move(b2)
    assert mv in [(7, 4), (7, 9)], mv


def test_vcf_immediate():
    # live four on board -> VCF returns the winning five point in 1 move
    b = Board(15)
    place(b, [(7, 5), (0, 0), (7, 6), (0, 1), (7, 7), (0, 2), (7, 8), (0, 3)])
    line = vcf_search(b, BLACK, max_depth=8)
    assert line is not None and len(line) == 1 and line[0] in [(7, 4), (7, 9)], line


def test_vcf_multi_step():
    # White has a rush four on the main diagonal (2,2),(4,4),(5,5),(6,6) with a
    # gap at (3,3): its only five point is (3,3). Black must answer with a
    # BLOCKING four at (3,3) (row 3: (3,4),(3,5),(3,6), white (3,7)), survive
    # white's forced block at (3,2), then win with a straight four on row 9.
    b = Board(15)
    black = [(3, 4), (3, 5), (3, 6), (9, 9), (9, 10), (9, 11)]
    white = [(3, 7), (2, 2), (4, 4), (5, 5), (6, 6), (0, 0)]
    for bl, wh in zip(black, white):
        b.play(*bl)
        b.play(*wh)
    assert b.to_move == BLACK
    assert b.five_points(WHITE) == [(3, 3)]
    t0 = time.time()
    line = vcf_search(b, BLACK, max_depth=12, time_limit=5.0)
    assert line is not None, "VCF should prove this win"
    inter = [mv for pair in zip(black, white) for mv in pair]
    won = _verify_vcf_line(inter, line)
    assert won, f"VCF line did not win: {line}"
    print(f"    vcf line: {line}  ({time.time()-t0:.2f}s)")


def _verify_vcf_line(setup_moves, attacker_moves):
    """Game-theoretic check that a VCF line wins against ANY defense.

    After each attacker move: if attacker already has five -> win. If the
    attacker threatens five at >=2 points and the defender has no five point
    of their own -> win. Otherwise the defender's ONLY non-losing reply is the
    single threat point (any other reply must leave the attacker an immediate
    five); verify that for every alternative reply, then follow the block.
    """
    b = Board(15)
    for m in setup_moves:
        b.play(*m)
    attacker = b.to_move
    defender = other(attacker)
    for m in attacker_moves:
        b.play(*m)
        if b.status() == attacker:
            return True                      # actual five on the board
        threats = b.five_points(attacker)
        if not threats:
            return False                     # not a forcing move
        if b.five_points(defender):
            return False                     # defender answers with own five
        if len(threats) >= 2:
            return True                      # unstoppable straight/double four
        block = threats[0]
        for r, c in [(r, c) for r in range(b.n) for c in range(b.n) if b.cells[r, c] == 0]:
            if (r, c) == block:
                continue
            bb = b.clone()
            bb.play(r, c)
            if bb.status() == defender:
                return False
            if not bb.five_points(attacker):
                return False                 # threat vanished -> line unsound
        b.play(*block)
        if b.status() == defender:
            return False
    return False                             # line ran out without a win


def test_vcf_soundness_random():
    """On random positions VCF must never 'prove' a win that loses on replay."""
    rng = random.Random(7)
    checked = 0
    for _ in range(30):
        b = Board(15)
        moves = rng.sample([(r, c) for r in range(3, 12) for c in range(3, 12)], 12)
        for i, m in enumerate(moves):
            b.play(*m)
        if b.status():
            continue
        line = vcf_search(b, b.to_move, max_depth=8, time_limit=1.0)
        if line:
            checked += 1
            assert _verify_vcf_line(moves, line), f"unsound proof {line} from {moves}"
    print(f"    soundness checked on {checked} proven positions")


def test_encode_and_symmetry():
    b = Board(15)
    place(b, [(7, 7), (7, 8), (8, 8)])
    x = encode(b)
    assert x.shape == (4, 15, 15) and x.dtype == np.float32
    # after 3 plies white is to move: plane0=white stones, plane1=black stones
    assert x[0][7, 8] == 1.0 and x[1][7, 7] == 1.0 and x[1][8, 8] == 1.0
    assert x[2][8, 8] == 1.0 and x[3].max() == 0.0  # last move marked; black NOT to move
    pi = np.zeros((15, 15), np.float32)
    pi[3, 4] = 1.0
    x2, pi2 = apply_symmetry(x, pi, k=1, flip=False)
    assert x2.shape == x.shape and abs(pi2.sum() - 1.0) < 1e-6
    assert x2[0][6, 7] == 1.0  # rot90 ccw maps white stone (7,8) -> (n-1-8, 7)


def test_fork_guard_detection():
    """Regression: from this real game position, black's (1,6) allows white's
    (9,10) fork (two five points at once); NetPlayer._allows_fork must flag it."""
    from gomoku.players import NetPlayer
    from gomoku.model import GomokuNet
    from gomoku.board import WHITE
    moves = [(7, 7), (9, 5), (5, 9), (9, 7), (3, 7), (4, 8), (1, 5), (7, 8), (0, 3),
             (7, 10), (0, 1), (9, 11), (0, 0), (7, 6), (0, 2), (0, 4), (0, 5), (3, 4),
             (0, 6), (9, 9), (0, 7), (11, 3), (0, 8), (0, 9), (0, 10), (11, 12),
             (0, 11), (8, 10), (0, 12), (10, 13), (0, 13), (0, 14), (1, 0), (11, 13),
             (1, 1), (2, 14), (1, 2), (11, 10), (1, 3), (1, 4)]
    b = Board(15)
    for m in moves:
        b.play(*m)
    assert b.to_move == BLACK
    player = NetPlayer(GomokuNet(15))
    assert player._allows_fork(b, (1, 6), WHITE), "(1,6) allows the (9,10) fork"


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        print(f"[RUN ] {t.__name__}")
        t()
        print(f"[PASS] {t.__name__}")
    print(f"\nAll {len(tests)} tests passed.")


if __name__ == "__main__":
    main()
