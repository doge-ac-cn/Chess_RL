"""MCTS (PUCT) guided by the policy-value network, AlphaZero-style.

Prior probabilities live on the edges (child.prior = P(parent -> child)).
"""
from __future__ import annotations

import math

import numpy as np
import torch

from .board import Board, DRAW, encode


class _Node:
    __slots__ = ("prior", "N", "W", "children")

    def __init__(self, prior: float = 0.0):
        self.prior = prior          # P of the edge from the parent to this node
        self.N = 0
        self.W = 0.0
        self.children: dict[tuple[int, int], "_Node"] = {}


class MCTSEngine:
    def __init__(self, net: torch.nn.Module, device: str = "cpu",
                 c_puct: float = 1.8, dirichlet_alpha: float = 0.3,
                 dirichlet_eps: float = 0.25, candidate_radius: int = 2):
        self.net = net
        self.device = device
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_eps = dirichlet_eps
        self.candidate_radius = candidate_radius
        self._noise_rng = np.random.default_rng(0)

    @torch.no_grad()
    def _priors(self, board: Board) -> dict[tuple[int, int], float]:
        x = torch.from_numpy(encode(board)).unsqueeze(0).to(self.device)
        logits = self.net(x)[0].flatten()
        cand = board.candidates(radius=self.candidate_radius)
        n = board.n
        mask = torch.full((n * n,), -1e9, device=logits.device)
        idx = torch.tensor([r * n + c for r, c in cand], dtype=torch.long, device=logits.device)
        p = torch.softmax((logits + mask), dim=0)
        p = p[idx].cpu().numpy()
        return {m: float(v) for m, v in zip(cand, p)}

    def run(self, board: Board, sims: int, root_noise: bool = True,
            temperature: float = 0.0, rng: np.random.Generator | None = None):
        """Returns (move, visit_pi (n*n float32), root Q of chosen move).

        temperature=0 -> greedy on visit counts; >0 -> sample visits^(1/T).
        """
        rng = rng or self._noise_rng
        root = _Node()
        root.children = {m: _Node(pr) for m, pr in self._priors(board).items()}
        if root_noise and len(root.children) > 1:
            noise = rng.dirichlet([self.dirichlet_alpha] * len(root.children))
            eps = self.dirichlet_eps
            for (m, ch), nz in zip(root.children.items(), noise):
                ch.prior = eps * ch.prior + (1.0 - eps) * float(nz)

        for _ in range(sims):
            node = root
            path: list[_Node] = []
            scratch = board.clone()

            # selection: descend while expanded (nodes with children)
            while node.children:
                sqrt_total = math.sqrt(sum(ch.N for ch in node.children.values())) or 1.0
                best_m, best_ch, best_v = None, None, -1e18
                for m, ch in node.children.items():
                    q = ch.W / ch.N if ch.N else 0.0
                    u = self.c_puct * ch.prior * sqrt_total / (1 + ch.N)
                    if q + u > best_v:
                        best_v, best_m, best_ch = q + u, m, ch
                scratch.play(*best_m)
                path.append(best_ch)
                node = best_ch

            # expansion / evaluation
            winner = scratch.status()
            if winner == DRAW:
                value = 0.0
            elif winner:
                value = -1.0        # side to move at leaf just lost
            else:
                node.children = {m: _Node(pr) for m, pr in self._priors(scratch).items()}
                value = float(self.net(
                    torch.from_numpy(encode(scratch)).unsqueeze(0).to(self.device))[1].item())

            # backup: flip value perspective each ply on the way up
            for ch in reversed(path):
                value = -value
                ch.W += value
                ch.N += 1

        moves = list(root.children.keys())
        visits = np.array([root.children[m].N for m in moves], dtype=np.float64)
        pi = np.zeros(board.n * board.n, dtype=np.float32)
        if visits.sum() > 0:
            for m, nv in zip(moves, visits):
                pi[m[0] * board.n + m[1]] = nv / visits.sum()
        if temperature <= 0.0:
            move = moves[int(np.argmax(visits))]
        else:
            w = visits ** (1.0 / temperature)
            move = moves[int(rng.choice(len(moves), p=w / w.sum()))]
        ch = root.children[move]
        q = ch.W / ch.N if ch.N else 0.0
        return move, pi, q
