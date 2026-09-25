"""Go P1 training: self-play -> replay buffer -> policy CE + value MSE.

  python3 -m go.train_go --iters 2 --games 2 --sims 16 --out go_net.pt
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch
import torch.nn.functional as F

from .board import Board
from .pipeline import GoMCTS, GoNet, count_params, encode, encode_move, N_MOVES


def selfplay_game(mcts: GoMCTS, sims: int, rng, temp_moves: int = 14,
                  temperature: float = 0.5, max_plies: int = 160):
    board = Board(mcts.net.n if hasattr(mcts.net, "n") else 9)
    n = board.n
    mcts.net.n = n
    data = []
    plies = 0
    while not board.is_over() and plies < max_plies:
        x = encode(board)
        t = temperature if plies < temp_moves else 0.0
        mv, pi = mcts.run(board, sims, root_noise=True, temperature=t, rng=rng)[:2]
        target = np.zeros(N_MOVES, dtype=np.float32)
        for m, pr in pi.items():
            target[encode_move(m, n)] = pr
        target /= target.sum()
        mover = board.to_move
        data.append((x, target, mover))
        board.play(mv)
        plies += 1
    black, white = board.score()
    winner = 1 if black > white else (2 if white > black else 0)
    out = []
    for x, target, mover in data:
        z = 0.0 if winner == 0 else (1.0 if mover == winner else -1.0)
        out.append((x, target, z))
    return out, plies, winner


def run(buffer, net, opt, device, epochs=1, batch=64, rng=None):
    rng = rng or np.random.default_rng(0)
    idx = np.arange(len(buffer))
    net.train()
    tot, nb = 0.0, 0
    for _ in range(epochs):
        rng.shuffle(idx)
        for i in range(0, len(idx) - batch + 1, batch):
            bidx = idx[i:i + batch]
            xs = torch.from_numpy(np.stack([buffer[j][0] for j in bidx])).to(device)
            targets = torch.from_numpy(np.stack([buffer[j][1] for j in bidx])).to(device)
            gt = torch.tensor([buffer[j][2] for j in bidx], device=device)
            logits, v = net(xs)
            logp = F.log_softmax(logits, dim=1)
            loss = -(targets * logp).sum(1).mean() + F.mse_loss(v, gt)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item()
            nb += 1
    return tot / max(1, nb)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=9)
    ap.add_argument("--iters", type=int, default=2)
    ap.add_argument("--games", type=int, default=2)
    ap.add_argument("--sims", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="go_net.pt")
    args = ap.parse_args()

    net = GoNet(n=args.n).to(args.device)
    print(f"[init] params {count_params(net) / 1e6:.2f}M device={args.device}", flush=True)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    buffer: list = []
    rng = np.random.default_rng(0)
    mcts = GoMCTS(net, device=args.device)

    for it in range(1, args.iters + 1):
        t0 = time.time()
        net.eval()
        for g in range(args.games):
            gdata, plies, winner = selfplay_game(mcts, args.sims,
                                                 rng=np.random.default_rng(it * 1000 + g))
            buffer.extend(gdata)
            print(f"[sp] iter{it} game{g}: plies={plies} winner={winner}", flush=True)
        del buffer[: max(0, len(buffer) - 40000)]
        loss = run(buffer, net, opt, args.device, epochs=args.epochs, batch=args.batch,
                   rng=rng)
        net.eval()
        torch.save({"model": net.state_dict(), "n": args.n}, args.out)
        print(f"[rl] iter{it} buffer={len(buffer)} loss={loss:.3f} "
              f"t={time.time() - t0:.0f}s saved->{args.out}", flush=True)


if __name__ == "__main__":
    main()
