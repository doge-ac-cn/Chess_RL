"""Self-play data generation for xiangqi (pure RL; net + MCTS, single process).

Each position sample: (planes, legal, pi, z) where pi is the MCTS visit
distribution ALIGNED with `legal`, and z the game result from the mover's
perspective. Kept single-process for the smoke stage; scale-up later.
"""
from __future__ import annotations

import numpy as np

from .board import Board
from .encode import encode
from .mcts import XiangqiMCTS


def selfplay_game(mcts: XiangqiMCTS, sims: int = 32, temp_moves: int = 12,
                  temperature: float = 1.0, rng=None, max_plies: int = 300):
    import random
    rng = rng or np.random.default_rng(0)
    pyrng = random.Random(int(rng.integers(1 << 30)))
    board = Board()
    data = []
    plies = 0
    while (not board.status() and not board.is_repetition_draw(3)
           and plies < max_plies):
        x = encode(board)
        legal = board.legal_moves()
        ply = len(board.history)
        t = temperature if ply < temp_moves else 0.0
        move, pi, _q = mcts.run(board, sims, root_noise=True, temperature=t,
                                rng=rng)
        data.append((x, legal, pi, board.to_move))
        board.play(move)
        plies += 1
    winner = board.status()
    if winner == -1:
        winner = 0
    out = []
    for x, legal, pi, mover in data:
        z = 0.0 if winner == 0 else (1.0 if mover == winner else -1.0)
        moves = list(pi.keys())
        pr = np.array([pi[m] for m in moves], dtype=np.float32)
        out.append((x, legal, (moves, pr), z))
    return out, plies, winner
