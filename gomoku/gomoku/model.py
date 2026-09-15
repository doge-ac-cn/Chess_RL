"""Small policy-value ResNet, sized for mobile deployment (~230k params)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


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


class GomokuNet(nn.Module):
    """Input (B, 4, n, n) -> policy logits (B, n*n), value in [-1, 1] (B,)."""

    def __init__(self, n: int = 15, ch: int = 48, blocks: int = 4, in_planes: int = 4):
        super().__init__()
        self.n = n
        self.inc = nn.Conv2d(in_planes, ch, 3, padding=1, bias=False)
        self.inb = nn.BatchNorm2d(ch)
        self.trunk = nn.Sequential(*[ResBlock(ch) for _ in range(blocks)])
        self.p1 = nn.Conv2d(ch, 2, 1)
        self.pb1 = nn.BatchNorm2d(2)
        self.p2 = nn.Conv2d(2, 1, 1)
        self.v1 = nn.Conv2d(ch, 8, 1)
        self.vb1 = nn.BatchNorm2d(8)
        self.vfc = nn.Linear(8 * n * n, 64)
        self.vout = nn.Linear(64, 1)

    def forward(self, x):
        h = F.relu(self.inb(self.inc(x)))
        h = self.trunk(h)
        p = F.relu(self.pb1(self.p1(h)))
        logits = self.p2(p).flatten(1)                       # (B, n*n)
        v = F.relu(self.vb1(self.v1(h))).flatten(1)          # (B, 8*n*n)
        v = torch.tanh(self.vout(F.relu(self.vfc(v))))       # (B, 1)
        return logits, v.squeeze(1)


def count_params(net: nn.Module) -> int:
    return sum(p.numel() for p in net.parameters())
