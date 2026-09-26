"""Self-play data generation for xiangqi (pure RL; net + MCTS).

Each position sample: (planes, legal, (moves, probs), z) where (moves, probs)
is the MCTS visit distribution ALIGNED with `legal`, and z the game result
from the mover's perspective. Shuffle draws (3-fold repetition without any
capture/mate progress) are scored -0.5 for BOTH sides to push the net away
from repetitive play.
"""
from __future__ import annotations

import numpy as np

from .board import Board
from .encode import encode
from .mcts import XiangqiMCTS


def selfplay_game(mcts: XiangqiMCTS, sims: int = 32, temp_moves: int = 24,
                  temperature: float = 1.0, rng=None, max_plies: int = 300):
    import random
    rng = rng or np.random.default_rng(0)
    board = Board()
    data = []
    plies = 0
    ended_by = "cap"
    while True:
        st = board.status()
        if st:
            ended_by = "mate"
            winner = st
            break
        if board.is_repetition_draw(3):
            ended_by = "repetition"
            winner = 0
            break
        if plies >= max_plies:
            ended_by = "cap"
            winner = 0
            break
        x = encode(board)
        legal = board.legal_moves()
        ply = len(board.history)
        t = temperature if ply < temp_moves else 0.0
        move, pi, _q = mcts.run(board, sims, root_noise=True, temperature=t,
                                rng=rng)
        data.append((x, legal, pi, board.to_move))
        board.play(move)
        plies += 1

    out = []
    for x, legal, pi, mover in data:
        if winner == 0:
            z = -0.5 if ended_by == "repetition" else 0.0
        else:
            z = 1.0 if mover == winner else -1.0
        moves = list(pi.keys())
        pr = np.array([pi[m] for m in moves], dtype=np.float32)
        out.append((x, legal, (moves, pr), z))
    return out, plies, winner, ended_by
