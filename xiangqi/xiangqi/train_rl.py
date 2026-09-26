"""Stage-2 xiangqi training: BC warm start -> self-play RL (AlphaZero-lite).

  python3 -m xiangqi.train_rl --iters 3 --games 4 --sims 24 --bc-init bc_net.pt

Pipeline per iteration: self-play games (net + MCTS + root noise) -> replay
buffer -> masked policy CE (visit distributions) + value MSE -> checkpoint.
BC init uses the teacher-cloned bc_net.pt when available (falls back to a
fresh net = the pure-RL route).
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch
import torch.nn.functional as F

from .encode import encode
from .mcts import XiangqiMCTS
from .net import XiangqiNet, count_params
from .selfplay import selfplay_game
from .selfplay_mp import parallel_selfplay


def policy_loss(pi_targets, flo, tlo):
    """pi_targets: list of (moves [(f,t)...], probs np (M,)) per sample.

    Move score = logits_f[f] + logits_t[t]; masked CE against the visit
    distribution restricted to the same legal list.
    """
    losses = []
    for k, (moves, probs) in enumerate(pi_targets):
        fidx = torch.tensor([m[0] for m in moves], device=flo.device)
        tidx = torch.tensor([m[1] for m in moves], device=flo.device)
        sc = flo[k, fidx] + tlo[k, tidx]
        logp = sc.log_softmax(0)
        pr = torch.from_numpy(probs).to(flo.device)
        losses.append(-(pr * logp).sum())
    return torch.stack(losses).mean()


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
            pi_targets = [buffer[j][2] for j in bidx]
            gt = torch.tensor([buffer[j][3] for j in bidx], device=device)
            flo, tlo, v = net(xs)
            loss = policy_loss(pi_targets, flo, tlo) + F.mse_loss(v, gt)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item()
            nb += 1
    return tot / max(1, nb)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=2)
    ap.add_argument("--games", type=int, default=4)
    ap.add_argument("--sims", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--bc-init", default="bc_net.pt")
    ap.add_argument("--workers", type=int, default=1,
                    help=">1 = multiprocess self-play (net loaded from --bc-init path)")
    ap.add_argument("--out", default="rl_net.pt")
    args = ap.parse_args()

    net = XiangqiNet().to(args.device)
    try:
        ck = torch.load(args.bc_init, map_location="cpu", weights_only=True)
        net.load_state_dict(ck["model"] if "model" in ck else ck)
        print(f"[init] BC warm start loaded from {args.bc_init}", flush=True)
    except FileNotFoundError:
        print("[init] pure RL (no BC warm start found)", flush=True)
    print(f"[init] params {count_params(net) / 1e6:.2f}M device={args.device}", flush=True)

    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    buffer: list = []
    rng = np.random.default_rng(0)

    net_path_for_mp = args.out  # workers load the latest saved net
    for it in range(1, args.iters + 1):
        t0 = time.time()
        net.eval()
        if args.workers > 1:
            # multiprocess self-play; workers load the net saved after the
            # previous iteration (first iteration: the BC/RL init checkpoint)
            init_path = args.bc_init if it == 1 else net_path_for_mp
            games = parallel_selfplay(init_path, args.games, args.workers,
                                      "cpu", args.sims, seed0=it * 1000)
            for gdata, plies, winner, ended_by in games:
                print(f"[sp] iter{it}: plies={plies} winner={winner} by={ended_by}",
                      flush=True)
                buffer.extend(gdata)
        else:
            mcts = XiangqiMCTS(net, device=args.device)
            for g in range(args.games):
                gdata, plies, winner, ended_by = selfplay_game(
                    mcts, sims=args.sims,
                    rng=np.random.default_rng(hash((it, g)) % (2**31)))
                buffer.extend(gdata)
                print(f"[sp] iter{it} game{g}: plies={plies} winner={winner} "
                      f"by={ended_by}", flush=True)
        del buffer[: max(0, len(buffer) - 20000)]

        stats = run(buffer, net, opt, args.device, epochs=args.epochs,
                    batch=args.batch, rng=rng)
        net.eval()
        torch.save({"model": net.state_dict(),
                    "arch": {"ch": 64, "blocks": 4}}, args.out)
        print(f"[rl] iter{it} buffer={len(buffer)} loss={stats:.3f} "
              f"t={time.time() - t0:.0f}s saved->{args.out}", flush=True)


if __name__ == "__main__":
    main()
