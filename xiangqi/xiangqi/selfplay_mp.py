"""Multiprocess self-play for xiangqi (pool pattern from gomoku/selfplay.py).

Workers load the net checkpoint from disk per pool creation, so the training
loop simply saves rl_net.pt and recreates the pool each iteration.
"""
from __future__ import annotations

import io
import multiprocessing as mp

import numpy as np
import torch

from .board import Board
from .encode import encode
from .mcts import XiangqiMCTS
from .net import XiangqiNet
from .selfplay import selfplay_game

_W: dict = {}


def _init_worker(net_path: str, device: str, sims: int):
    torch.set_num_threads(1)
    net = XiangqiNet().to(device).eval()
    state = torch.load(net_path, map_location="cpu", weights_only=True)
    net.load_state_dict(state["model"] if "model" in state else state)
    _W["net"] = net
    _W["device"] = device
    _W["sims"] = sims


def _game_task(seed: int):
    import random
    rng = np.random.default_rng(seed)
    mcts = XiangqiMCTS(_W["net"], device=_W["device"])
    out, plies, winner, ended_by = selfplay_game(mcts, sims=_W["sims"], rng=rng)
    return out, plies, winner, ended_by


def parallel_selfplay(net_path: str, n_games: int, n_workers: int,
                      device: str = "cpu", sims: int = 24, seed0: int = 0):
    """Returns list of selfplay_game outputs (one per game)."""
    if n_workers <= 1:
        _init_worker(net_path, device, sims)
        return [_game_task(seed0 + i) for i in range(n_games)]
    ctx = mp.get_context("spawn")  # safe when the parent has initialized CUDA
    with ctx.Pool(n_workers, initializer=_init_worker,
                  initargs=(net_path, device, sims)) as pool:
        return pool.map(_game_task, range(seed0, seed0 + n_games))
