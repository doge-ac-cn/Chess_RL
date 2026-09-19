"""Plain-assert tests for the xiangqi rules engine: `python3 tests/test_xiangqi.py`."""
import sys

sys.path.insert(0, ".")

from xiangqi.board import (Board, IDX, RED, BLACK, K, A, B, N, R, C, P, BK, BA,
                           BB, BN, BR, BC, BP, EMPTY)


def TYPE_OF(p):
    return p if p < 10 else p - 10


def test_initial_position():
    b = Board()
    assert b.to_move == RED
    assert b.cells[IDX(0, 4)] == BK and b.cells[IDX(9, 4)] == K
    mv = b.legal_moves()
    # well-known: 44 legal moves in the xiangqi opening
    assert len(mv) == 44, f"opening legal moves = {len(mv)}"


def test_horse_leg_block():
    b = Board()
    # opening horse at (9,1): (8,3) is blocked by the own elephant at (9,2);
    # only the two forward jumps are legal
    moves = [t for f, t in b.legal_moves() if f == IDX(9, 1)]
    assert IDX(7, 0) in moves and IDX(7, 2) in moves and IDX(8, 3) not in moves
    # block the leg (8,1): both forward jumps disappear
    b.cells[IDX(8, 1)] = BP
    moves = [t for f, t in b.legal_moves(RED) if f == IDX(9, 1)]
    assert IDX(7, 0) not in moves and IDX(7, 2) not in moves
    # remove the elephant: (8,3) becomes reachable (leg (9,2) now free)
    b2 = Board()
    b2.cells[IDX(9, 2)] = EMPTY
    moves = [t for f, t in b2.legal_moves() if f == IDX(9, 1)]
    assert IDX(8, 3) in moves


def test_elephant_eye_and_river():
    b = Board()
    moves = [t for f, t in b.legal_moves() if f == IDX(9, 2)]
    assert IDX(7, 4) in moves
    b.cells[IDX(8, 3)] = BP          # block the elephant eye
    moves = [t for f, t in b.legal_moves(RED) if f == IDX(9, 2)]
    assert IDX(7, 4) not in moves
    # red elephants can never cross the river
    for f, t in b.legal_moves(RED):
        if TYPE_OF(b.cells[f]) == B:
            assert (t // 9) >= 5
    # black elephants never cross
    for f, t in b.legal_moves(BLACK):
        if TYPE_OF(b.cells[f]) == BB:
            assert (t // 9) <= 4


def test_pawn_crossing_river():
    b = Board()
    # red pawn at (3,0) (already crossed): forward + sideways, never back
    b.cells[IDX(3, 0)] = P
    b.cells[IDX(9, 0)] = EMPTY
    moves = [t for f, t in b.legal_moves(RED) if f == IDX(3, 0)]
    assert IDX(2, 0) in moves and IDX(3, 1) in moves and IDX(4, 0) not in moves
    # uncrossed red pawn at (5,2): only forward
    b.cells[IDX(5, 2)] = P
    moves = [t for f, t in b.legal_moves(RED) if f == IDX(5, 2)]
    assert IDX(4, 2) in moves and IDX(5, 1) not in moves and IDX(5, 3) not in moves


def test_cannon_needs_screen_to_capture():
    b = Board(setup=[
        "..k......",     # black king (0,3) — off file 4 to avoid facing
        ".........",
        ".........",
        ".........",
        "...r.....",     # black rook at (4,3)
        ".........",
        ".........",
        "...C.....",     # red cannon (7,3)
        ".........",
        "....K....",     # red king (9,4)
    ])
    targets = [t for f, t in b.legal_moves(RED) if f == IDX(7, 3)]
    assert IDX(4, 3) not in targets          # no screen -> cannot capture
    assert IDX(6, 3) in targets              # quiet move up to the empty cell
    b.cells[IDX(6, 3)] = BP                  # insert a screen
    targets = [t for f, t in b.legal_moves(RED) if f == IDX(7, 3)]
    assert IDX(4, 3) in targets              # screen -> capture legal
    assert IDX(6, 3) not in targets          # quiet move onto own pawn gone


def test_flying_general():
    b = Board(setup=[
        "....k....",
        ".........",
        ".........",
        ".........",
        ".........",
        ".........",
        ".........",
        ".........",
        ".........",
        "....K....",
    ])
    assert b.in_check(RED) and b.in_check(BLACK)   # kings face on file 4
    for f, t in b.legal_moves():
        if f == IDX(9, 4):
            assert (t // 9) != 4                   # king must leave the file


def test_check_detection():
    b = Board(setup=[
        "...k.....",     # black king (0,3)
        ".........",
        ".........",
        ".........",
        "...R.....",     # red rook (4,3) checks along file 3
        ".........",
        ".........",
        ".........",
        ".........",
        "....K....",     # red king (9,4) — different file, no facing
    ])
    assert b.in_check(BLACK)
    assert not b.in_check(RED)


def test_checkmate():
    # back-rank rook mate: rook (0,0) checks king (0,3);
    # horse (2,3) covers (0,4); rook (1,0) covers (1,3) along row 1
    b = Board(setup=[
        "R..k.....",
        "R........",
        "...N.....",
        ".........",
        ".........",
        ".........",
        ".........",
        ".........",
        ".........",
        "....K....",
    ])
    b.to_move = BLACK
    assert b.in_check(BLACK)
    assert b.status() == RED, f"expected checkmate, status={b.status()}"


def test_stalemate_king_stuck():
    # black king (0,0): (0,1) covered by horse (1,3), (1,0) covered by horse (3,1),
    # (0,0) itself not attacked -> stalemate (no legal move) -> mover loses
    b = Board(setup=[
        "k........",
        "...N.....",
        ".........",
        ".N.......",
        ".........",
        ".........",
        ".........",
        ".........",
        ".........",
        "....K....",
    ])
    b.to_move = BLACK
    assert not b.in_check(BLACK)
    assert b.status() == RED, f"expected stalemate loss, status={b.status()}"


def test_undo_restores_position():
    import random
    rng = random.Random(1)
    b = Board()
    snapshot = list(b.cells)
    moves = b.legal_moves()
    for mv in rng.sample(moves, 5):
        b.play(mv)
        b.undo()
    assert b.cells == snapshot and b.to_move == RED and not b.history


def test_selfplay_smoke():
    """random vs random completes 300 plies without crashing."""
    import random
    rng = random.Random(0)
    b = Board()
    for ply in range(300):
        st = b.status()
        if st:
            break
        b.play(rng.choice(b.legal_moves()))
    assert isinstance(b.status(), int)


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        print(f"[RUN ] {t.__name__}")
        t()
        print(f"[PASS] {t.__name__}")
    print(f"\nAll {len(tests)} tests passed.")


if __name__ == "__main__":
    main()
