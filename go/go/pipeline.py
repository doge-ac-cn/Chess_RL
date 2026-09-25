"""Go P1 pipeline: encode + policy-value net + MCTS + self-play + training.

Single-file implementation for the 9x9 first stage (see go/DESIGN.md).
  python3 -m go.pipeline --iters 2 --games 2 --sims 16
"""
from __future__ import annotations

import argparse
import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from go.board import Board, BLACK, WHITE, PASS, KOMI

N_PLANES = 4
N_MOVES = 81 + 1  # 81 points + pass


def encode(b: Board) -> np.ndarray:
    me, opp = b.to_move, (3 - b.to_move)
    x = np.zeros((N_PLANES, b.n, b.n), dtype=np.float32)
    for i, v in enumerate(b.cells):
        if v == me:
            x[0].flat[i] = 1
        elif v == opp:
            x[1].flat[i] = 1
    if b.ko_point is not None:
        x[2].flat[b.ko_point] = 1
    x[3] = 1.0 if me == BLACK else 0.0
    return x


def decode_move(idx: int, n: int):
    return PASS if idx == n * n else (idx // n, idx % n)


def encode_move(mv, n: int) -> int:
    return n * n if mv is PASS else mv[0] * n + mv[1]


class GoNet(nn.Module):
    def __init__(self, n: int = 9, ch: int = 64, blocks: int = 4):
        super().__init__()
        self.n = n
        self.inc = nn.Conv2d(N_PLANES, ch, 3, padding=1, bias=False)
        self.inb = nn.BatchNorm2d(ch)
        self.trunk = nn.Sequential(*[ResBlock(ch) for _ in range(blocks)])
        self.pol = nn.Conv2d(ch, 2, 1)
        self.pb = nn.BatchNorm2d(2)
        self.pol_head = nn.Linear(2 * n * n, N_MOVES)
        self.vc = nn.Conv2d(ch, 8, 1)
        self.vb = nn.BatchNorm2d(8)
        self.vf = nn.Linear(8 * n * n, 64)
        self.vout = nn.Linear(64, 1)

    def forward(self, x):
        h = F.relu(self.inb(self.inc(x)))
        h = self.trunk(h)
        p = F.relu(self.pb(self.pol(h))).flatten(1)
        logits = self.pol_head(p)
        v = F.relu(self.vb(self.vc(h))).flatten(1)
        v = torch.tanh(self.vout(F.relu(self.vf(v))))
        return logits, v.squeeze(1)

    def load_bc(self, path):
        ck = torch.load(path, map_location="cpu", weights_only=True)
        self.load_state_dict(ck["model"] if "model" in ck else ck)


class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.b1 = nn.BatchNorm2d(ch)
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.b2 = nn.BatchNorm2d(ch)

    def forward(self, x):
        y = F.relu(self.b1(self.c1(x)))
        y = self.b2(self.c2(y))
        return F.relu(x + y)


def count_params(net):
    return sum(p.numel() for p in net.parameters())


# ---------------- MCTS ----------------

class _Node:
    __slots__ = ("prior", "N", "W", "children")

    def __init__(self, prior):
        self.prior = prior
        self.N = 0
        self.W = 0.0
        self.children = None  # dict idx -> _Node


class GoMCTS:
    def __init__(self, net, device="cpu", c_puct=1.6):
        self.net = net
        self.device = device
        self.c_puct = c_puct

    @torch.no_grad()
    def _evaluate(self, board: Board):
        x = torch.from_numpy(encode(board)).unsqueeze(0).to(self.device)
        logits, v = self.net(x)
        logits = logits.flatten().cpu().numpy()
        n = board.n
        legal = board.legal_moves() + [PASS]
        idxs = [encode_move(m, n) for m in legal]
        sc = logits[idxs]
        sc -= sc.max()
        pr = np.exp(sc)
        pr /= pr.sum()
        return list(zip(legal, pr.tolist())), float(v)

    def run(self, board: Board, sims: int, root_noise=False, alpha=1.0, eps=0.25,
            temperature=0.0, rng=None):
        rng = rng or np.random.default_rng(0)
        root = _Node(1.0)
        prior, _ = self._evaluate(board)
        root.children = {encode_move(m, board.n): _Node(p) for m, p in prior}
        if root_noise and len(root.children) > 1:
            noise = rng.dirichlet([alpha] * len(root.children))
            for (m, ch), nz in zip(root.children.items(), noise):
                ch.prior = eps * ch.prior + (1 - eps) * float(nz)

        for _ in range(sims):
            scratch = board.clone()
            path = []
            node = root
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
                scratch.play(decode_move(best_m, board.n))
                path.append(best_ch)
                node = best_ch
            # terminal / expansion
            if scratch.is_over():
                black, white = scratch.score()
                value = 1.0 if black > white else (-1.0 if white > black else 0.0)
                if scratch.to_move != BLACK:
                    value = -value        # value is from side-to-move perspective
            else:
                pr, value = self._evaluate(scratch)
                node.children = {encode_move(m, board.n): _Node(p) for m, p in pr}
            for ch in reversed(path):
                value = -value
                ch.W += value
                ch.N += 1

        encoded = list(root.children.keys())
        visits = np.array([root.children[m].N for m in encoded], dtype=np.float64)
        pi = {decode_move(m, board.n): (v / visits.sum())
              for m, v in zip(encoded, visits / visits.sum())}
        if temperature <= 0:
            move = decode_move(encoded[int(np.argmax(visits))], board.n)
        else:
            w = visits ** (1.0 / temperature)
            move = decode_move(encoded[int(rng.choice(len(encoded), p=(w / w.sum())))], board.n)
        return move, pi

