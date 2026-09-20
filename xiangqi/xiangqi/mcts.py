"""MCTS for xiangqi: PUCT over legal moves, factorized policy priors.

Port of gomoku/mcts.py adapted to the xiangqi Board (legal_moves as candidate
source; no radius pruning — xiangqi has ~40-60 legal moves per position).
Priors combine the from/to factorized heads: P(f,t) = softmax(logits_f[f] +
logits_t[t]) over the legal move list.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from .board import Board, other
from .encode import encode


class XiangqiMCTS:
    def __init__(self, net, device: str = "cpu", c_puct: float = 1.6):
        self.net = net
        self.device = device
        self.c_puct = c_puct
        self._rng = np.random.default_rng(0)

    @torch.no_grad()
    def _evaluate(self, board: Board) -> tuple[list[tuple[tuple[int, int], float]], float]:
        x = torch.from_numpy(encode(board)).unsqueeze(0).to(self.device)
        flo, tlo, v = self.net(x)
        flo = flo.flatten().cpu().numpy()
        tlo = tlo.flatten().cpu().numpy()
        legal = board.legal_moves()
        scores = np.array([flo[f] + tlo[t] for f, t in legal], dtype=np.float64)
        scores -= scores.max()
        pr = np.exp(scores)
        pr /= pr.sum()
        return list(zip(legal, pr.tolist())), float(v)

    def run(self, board: Board, sims: int, root_noise: bool = False,
            alpha: float = 1.0, eps: float = 0.25, temperature: float = 0.0,
            rng=None):
        rng = rng or self._rng
        root = _Node(1.0)
        prior, v0 = self._evaluate(board)
        root.children = {m: _Node(pr) for m, pr in prior}
        if root_noise and len(root.children) > 1:
            noise = rng.dirichlet([alpha] * len(root.children))
            for (m, ch), nz in zip(root.children.items(), noise):
                ch.prior = eps * ch.prior + (1 - eps) * float(nz)

        for _ in range(sims):
            scratch = board.clone()
            path = []
            node = root
            # selection
            while node.children:
                total = sum(ch.N for ch in node.children.values())
                sqrt_total = math.sqrt(total) or 1.0
                best_m = best_ch = None
                best_v = -1e18
                for m, ch in node.children.items():
                    q = ch.W / ch.N if ch.N else 0.0
                    u = self.c_puct * ch.prior * sqrt_total / (1 + ch.N)
                    if q + u > best_v:
                        best_v, best_m, best_ch = q + u, m, ch
                scratch.play(best_m)
                path.append(best_ch)
                node = best_ch
            # terminal / expansion
            st = scratch.status()
            if st:                       # side to move has no legal move -> loses
                value = -1.0
            else:
                pr, value = self._evaluate(scratch)
                node.children = {m: _Node(p) for m, p in pr}
            # backup
            for ch in reversed(path):
                value = -value
                ch.W += value
                ch.N += 1

        moves = list(root.children.keys())
        visits = np.array([root.children[m].N for m in moves], dtype=np.float64)
        pi = dict(zip(moves, (visits / visits.sum()).tolist())) if visits.sum() else \
            dict(zip(moves, [1.0 / len(moves)] * len(moves)))
        if temperature <= 0:
            move = moves[int(np.argmax(visits))]
        else:
            w = visits ** (1.0 / temperature)
            move = moves[int(rng.choice(len(moves), p=(w / w.sum())))]
        ch = root.children[move]
        return move, pi, (ch.W / ch.N if ch.N else 0.0)


class _Node:
    __slots__ = ("prior", "N", "W", "children")

    def __init__(self, prior: float):
        self.prior = prior
        self.N = 0
        self.W = 0.0
        self.children: dict | None = None
