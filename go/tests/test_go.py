import sys
sys.path.insert(0, ".")
from go.board import Board, BLACK, WHITE, PASS


def test_capture_sequence():
    b = Board(5)
    # black surrounds white (2,2) on three sides, then takes the last liberty
    replies = [(4, 4), (4, 3), (4, 2), (4, 1)]
    for (r, c), reply in zip([(1, 2), (2, 1), (2, 3), (3, 2)], replies):
        b.play((r, c))
        b.play(reply)
    assert b.cells[2 * 5 + 2] == 0, "white stone should be captured"
    assert b.cells[3 * 5 + 2] == BLACK


def test_suicide_forbidden():
    b = Board(5)
    b.play((0, 1)); b.play((4, 4))
    b.play((1, 0)); b.play((4, 3))
    b.play((2, 2))                    # black's turn done -> white to move
    # white (0,0): surrounded by black (0,1),(1,0) and no capture -> suicide
    try:
        b.play((0, 0))
        assert False, "suicide allowed!"
    except ValueError:
        pass


def test_ko():
    b = Board(5)
    setup = [
        ".BW..",
        "B.B..",
        ".BW..",
        ".....",
        ".....",
    ]
    for r, row in enumerate(setup):
        for c, ch in enumerate(row):
            if ch == "B":
                b.play((r, c)); 
            elif ch == "W":
                b.play((r, c))
    # messy to seed via alternation; build ko directly by hand:
    b = Board(5)
    b.cells[0*5+1] = BLACK; b.cells[0*5+3] = BLACK   # B at (0,1),(0,3)
    b.cells[1*5+2] = BLACK                            # B at (1,2)
    b.cells[1*5+1] = WHITE; b.cells[1*5+3] = WHITE    # W at (1,1),(1,3)
    b.cells[0*5+2] = WHITE                            # W at (0,2) — one liberty (0,0)? no:
    # W (0,2) neighbors: (0,1)B (0,3)B (1,2)B -> already captured? handle below.
    # fix: make (0,0) empty but not a liberty: use (0,0) WHITE
    b = Board(5)
    b.cells[0*5+0] = WHITE
    b.cells[0*5+1] = BLACK; b.cells[1*5+0] = BLACK; b.cells[0*5+2] = WHITE
    # W(0,0): neighbors (0,1)B (1,0)B -> dead on placement; skip. ko classic:
    b = Board(5)
    #  . B W .
    #  B W . W      black plays (1,2) capturing white (1,1)? craft:
    #  . B W .
    b.cells[0*5+1] = BLACK; b.cells[0*5+2] = WHITE
    b.cells[1*5+0] = BLACK; b.cells[1*5+1] = WHITE
    b.cells[2*5+1] = BLACK; b.cells[2*5+2] = WHITE
    b.cells[1*5+3] = WHITE
    b.to_move = BLACK
    b.play((1, 2))   # black takes white (1,1): its last liberty
    assert b.cells[1*5+1] == 0
    # immediate recapture is ko-forbidden: white (1,1) would take black (1,2)
    try:
        b.play((1, 1))
        assert False, "ko recapture allowed!"
    except ValueError:
        pass
    # after a move elsewhere, recapture is fine again
    b.play((4, 4))
    b.play((4, 0))
    b.play((1, 1))
    assert b.cells[1*5+2] == 0  # black stone captured back


def test_two_passes_end_and_scoring():
    # black wall col 0 + col 2 encloses column 1; white has nothing
    b = Board(5)
    for r in range(5):
        b.play((r, 0)); b.play((r, 4))
    for r in range(5):
        b.play((r, 2)); b.play(PASS)
    b.play(PASS); b.play(PASS)
    assert b.is_over()
    black, white = b.score()
    # black: 10 stones + col1 (5) + col3 (5, borders white+black? no: col3 borders
    # col2 black and col4 white -> neutral) -> 10 + 5 = 15; white: 5 stones + 7.5
    assert black == 15.0 and white == 12.5, f"got {black}, {white}"


def test_random_game_smoke():
    import random
    rng = random.Random(0)
    b = Board(9)
    for ply in range(120):
        if b.is_over():
            break
        moves = b.legal_moves()
        if rng.random() < 0.05 or not moves:
            b.play(PASS)
        else:
            b.play(rng.choice(moves))
    print("9x9 random game plies:", ply + 1, "over:", b.is_over())
    assert isinstance(b.score(), tuple)


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        print(f"[RUN ] {t.__name__}")
        t()
        print(f"[PASS] {t.__name__}")
    print(f"\nAll {len(tests)} tests passed.")


if __name__ == "__main__":
    main()
