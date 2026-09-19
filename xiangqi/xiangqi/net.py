"""Policy-value net for xiangqi with factorized from/to policy heads.

Policy P(move) is the product of a from-square distribution and a to-square
distribution, legality-masked at inference. This keeps the head tiny
(2 x Linear(ch*90 -> 90)) compared to a raw 8100-way head.
Value head outputs tanh scalar from the side-to-move perspective.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encode import N_PLANES, ROWS, COLS


class ResBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.b1 = nn.BatchNorm2d(ch)
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.b2 = nn.BatchNorm2d(ch)

    def forward(self, x):
        y = F.relu(self.b1(self.c1(x)))
        y = self.b2(self.c2(y))
        return F.relu(x + y)


class XiangqiNet(nn.Module):
    def __init__(self, ch: int = 64, blocks: int = 4, n_planes: int = N_PLANES):
        super().__init__()
        self.n = COLS * ROWS  # 90
        self.inc = nn.Conv2d(n_planes, ch, 3, padding=1, bias=False)
        self.inb = nn.BatchNorm2d(ch)
        self.trunk = nn.Sequential(*[ResBlock(ch) for _ in range(blocks)])
        self.pol = nn.Linear(ch * ROWS * COLS, self.n)   # from logits
        self.pot = nn.Linear(ch * ROWS * COLS, self.n)   # to logits
        self.vc = nn.Conv2d(ch, 8, 1)
        self.vb = nn.BatchNorm2d(8)
        self.vf = nn.Linear(8 * ROWS * COLS, 64)
        self.vout = nn.Linear(64, 1)

    def forward(self, x):
        h = F.relu(self.inb(self.inc(x)))
        h = self.trunk(h)
        flat = h.flatten(1)
        from_logits = self.pol(flat)                       # (B, 90)
        to_logits = self.pot(flat)                         # (B, 90)
        v = F.relu(self.vb(self.vc(h))).flatten(1)
        v = torch.tanh(self.vout(F.relu(self.vf(v))))
        return from_logits, to_logits, v.squeeze(1)

    def move_logits(self, from_logits: torch.Tensor, to_logits: torch.Tensor,
                   legal: list[list[int]]) -> tuple[torch.Tensor, list[list[tuple[int, int]]]]:
        """Combine factorized heads into per-move scores with legality mask.

        Returns (score_tensor (B, max_moves), moves_per_sample); illegal
        positions get -1e9 so argmax/softmax is legal-safe.
        """
        Bsz = from_logits.shape[0]
        scores = torch.full((Bsz, 120), -1e9, device=from_logits.device)
        all_moves = []
        for b in range(Bsz):
            lm = legal[b]
            mv_list = []
            for k, m in enumerate(lm):
                f, t = m
                scores[b, k] = from_logits[b, f] + to_logits[b, t]
                mv_list.append(m)
            all_moves.append(mv_list)
        return scores, all_moves


def count_params(net: nn.Module) -> int:
    return sum(p.numel() for p in net.parameters())
