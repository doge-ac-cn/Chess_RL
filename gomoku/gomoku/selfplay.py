"""Self-play game generation with multiprocessing workers."""
from __future__ import annotations

import io
import os

import numpy as np
import torch

from .board import Board, DRAW, BLACK, other
from .mcts import MCTSEngine
from .model import GomokuNet

_WORKER: dict = {}


def _init_worker(n: int, device: str, sims: int, ch: int = 48, blocks: int = 4):
    torch.set_num_threads(1)
    _WORKER["n"] = n
    _WORKER["device"] = device
    _WORKER["sims"] = sims
    seed = (os.getpid() * 1_000_003) % (2**31)
    _WORKER["rng"] = np.random.default_rng(seed)
    net = GomokuNet(n=n, ch=ch, blocks=blocks)
    net.to(device).eval()
    _WORKER["net"] = net
    _WORKER["engine"] = MCTSEngine(net, device=device)


def _load_weights(weights: bytes) -> None:
    buf = io.BytesIO(weights)
    state = torch.load(buf, map_location=_WORKER["device"], weights_only=True)
    _WORKER["net"].load_state_dict(state)


def selfplay_game(weights: bytes, cfg: dict, seed: int | None = None):
    """Play one full self-play game with the given weights.

    Returns a list of (planes int8 (4,n,n), pi float32 (n*n), z float) where
    z is the game outcome from the perspective of the player to move.
    """
    rng = _WORKER["rng"] if seed is None else np.random.default_rng(seed)
    _load_weights(weights)
    engine = _WORKER["engine"]
    sims = cfg["sims"]
    temp_moves = cfg.get("temp_moves", 8)
    temperature = cfg.get("temperature", 1.0)

    board = Board(_WORKER["n"])
    data = []
    while True:
        status = board.status()
        if status == DRAW:
            outcome = {BLACK: 0.0, 2: 0.0}
            break
        if status:
            outcome = {status: 1.0, other(status): -1.0}
            break
        ply = len(board.history)
        # forced-move shortcut (logically safe, speeds up self-play a lot)
        my5 = board.five_points(board.to_move)
        if my5:
            move, pi = my5[0], None
        else:
            opp5 = board.five_points(other(board.to_move))
            if opp5:
                move, pi = opp5[0], None
            else:
                t = temperature if ply < temp_moves else 0.0
                move, pi, _q = engine.run(board, sims, root_noise=True,
                                          temperature=t, rng=rng)
        if pi is None:  # shortcut move: one-hot policy target
            pi = np.zeros(board.n * board.n, dtype=np.float32)
            pi[move[0] * board.n + move[1]] = 1.0
        planes = board.cells
        me = board.to_move
        x = np.zeros((4, board.n, board.n), dtype=np.int8)
        x[0] = planes == me
        x[1] = planes == other(me)
        if board.last:
            x[2][board.last] = 1
        x[3] = 1 if me == BLACK else 0
        data.append((x, pi, me))
        board.play(*move)

    return [(x, pi, float(outcome[mover])) for x, pi, mover in data]


def parallel_selfplay(weights: bytes, cfg: dict, n_games: int,
                      n_workers: int, n: int, device: str = "cpu", sims: int = 64):
    """Generate n_games self-play games across a process pool."""
    import multiprocessing as mp

    tasks = [(weights, cfg, i) for i in range(n_games)]
    ch, blocks = cfg.get("ch", 48), cfg.get("blocks", 4)
    if n_workers <= 1:
        _init_worker(n, device, sims, ch, blocks)
        return [selfplay_game(*t) for t in tasks]
    ctx = mp.get_context("fork")
    with ctx.Pool(n_workers, initializer=_init_worker,
                  initargs=(n, device, sims, ch, blocks)) as pool:
        return pool.map(_game_task, tasks)


def _game_task(t):
    return selfplay_game(*t)
