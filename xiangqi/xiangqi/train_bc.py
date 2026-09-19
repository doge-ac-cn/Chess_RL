"""Stage-1 xiangqi training: behavior cloning of the heuristic teacher.

Generates teacher self-play games (asymmetric temperatures for decisive
outcomes), encodes positions, and trains the factorized policy-value net with
legality-masked cross entropy on the (from, to) pair + value MSE.

  python3 -m xiangqi.train_bc --games 200 --epochs 12
"""
from __future__ import annotations

import argparse
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from .board import Board, RED
from .encode import encode
from .heuristic import HeuristicPlayer, RandomPlayer
from .net import XiangqiNet, count_params


def generate_bc_data(n_games: int, seed: int = 0):
    """[(planes, legal, (f,t), z)] from teacher-vs-random decisive games."""
    data = []
    for g in range(n_games):
        if g and g % 20 == 0:
            print(f"[bc] games {g}/{n_games}, positions {len(data)}", flush=True)
        b = Board()
        teacher_is_red = (g % 2 == 0)
        teacher = HeuristicPlayer(0.2, random.Random(seed + 1000 + g))
        rnd = RandomPlayer(seed + 50000 + g)
        steps = []
        while not b.status() and len(b.history) < 200:
            legal = b.legal_moves()
            agent = teacher if (b.to_move == RED) == teacher_is_red else rnd
            mv = agent.move(b)
            steps.append((encode(b), legal, mv, b.to_move))
            b.play(mv)
        winner = b.status()
        for x, legal, mv, mover in steps:
            z = 0.0 if winner == 0 else (1.0 if mover == winner else -1.0)
            data.append((x, legal, mv, z))
    return data


def train(data, epochs: int = 12, batch: int = 128, lr: float = 1e-3,
          device: str = "cpu"):
    net = XiangqiNet().to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    rng = np.random.default_rng(0)
    idx = np.arange(len(data))
    net.train()
    for ep in range(epochs):
        rng.shuffle(idx)
        tot, n, exact, cnt = 0.0, 0, 0, 0
        for i in range(0, len(idx) - batch + 1, batch):
            bidx = idx[i:i + batch]
            xs = torch.from_numpy(np.stack([data[j][0] for j in bidx])).to(device)
            legal = [data[j][1] for j in bidx]
            mv = [data[j][2] for j in bidx]
            gt = torch.tensor([data[j][3] for j in bidx], device=device)
            flo, tlo, v = net(xs)
            sc, _ = net.move_logits(flo, tlo, legal)
            losses = []
            for k in range(len(bidx)):
                lml = sc[k, :len(legal[k])].log_softmax(0)
                try:
                    pos = legal[k].index(mv[k])
                except ValueError:
                    continue
                losses.append(-lml[pos])
            loss = torch.stack(losses).mean() + F.mse_loss(v, gt)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item(); n += 1
            with torch.no_grad():
                for k in range(len(bidx)):
                    pred = legal[k][int(sc[k, :len(legal[k])].argmax())]
                    exact += int(pred == mv[k]); cnt += 1
        print(f"ep{ep}: loss={tot / max(1, n):.3f} legal-top1={exact / max(1, cnt):.3f}",
              flush=True)
    net.eval()
    return net


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="bc_net.pt")
    args = ap.parse_args()

    t0 = time.time()
    data = generate_bc_data(args.games)
    zs = [d[3] for d in data]
    print(f"[bc] {len(data)} positions ({sum(1 for z in zs if z > 0)}W/"
          f"{sum(1 for z in zs if z < 0)}L) in {time.time()-t0:.0f}s", flush=True)
    net = train(data, epochs=args.epochs, device=args.device)
    torch.save({"model": net.state_dict(),
                "arch": {"ch": 64, "blocks": 4}}, args.out)
    print(f"[bc] saved -> {args.out} ({count_params(net)/1e6:.2f}M params)", flush=True)


if __name__ == "__main__":
    main()
