"""Training pipeline.

Stage 1 (BC):   behavior-clone the rule-based heuristic teacher (its games vs
                itself) to warm-start the policy/value net.
Stage 2 (RL):   AlphaZero-style self-play: MCTS(64) + Dirichlet noise -> replay
                buffer -> train -> gate against the previous best -> promote.

Use --skip-bc for the pure self-play RL route.

Run:  python3 -m gomoku.train --out checkpoints --hours 1.5
"""
from __future__ import annotations

import argparse
import io
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .board import Board, BLACK
from .eval import match_parallel, score
from .heuristic import HeuristicPlayer
from .model import GomokuNet, count_params
from .selfplay import parallel_selfplay

AUG = 8  # dihedral symmetries


def weights_bytes(net: torch.nn.Module) -> bytes:
    buf = io.BytesIO()
    torch.save(net.state_dict(), buf)
    return buf.getvalue()


def load_net(path: str, n: int, device: str) -> GomokuNet:
    ck = torch.load(path, map_location="cpu", weights_only=True)
    arch = ck.get("arch", {}) if isinstance(ck, dict) else {}
    net = GomokuNet(n=n, ch=arch.get("ch", 48), blocks=arch.get("blocks", 4))
    net.load_state_dict(ck["model"] if "model" in ck else ck)
    return net.to(device)


# ---------------- teacher data (BC) ----------------

def bc_dataset(n_games: int, n: int, seed: int):
    """(planes int8, policy one-hot float32, z float) triples from teacher games.

    Teacher-vs-teacher is inherently drawish (1-ply defense holds forever), so
    games are played teacher vs a random opponent: decisive outcomes, fast
    games, and value labels that balance naturally (+1 on teacher positions,
    -1 on random-side positions). Every position is relabeled with the
    deterministic teacher's move, so policy targets are always best-play.
    """
    import random
    from .players import RandomPlayer
    from .board import BLACK as _B
    data = []
    for g in range(n_games):
        b = Board(n)
        teacher_is_black = (g % 2 == 0)
        teacher = HeuristicPlayer(temperature=0.2, rng=random.Random(seed + 1000 * g))
        rnd = RandomPlayer(seed=seed + 1000 * g + 31337)
        labeler = HeuristicPlayer(temperature=0.0, rng=random.Random(seed + 77))
        steps = []
        while True:
            st = b.status()
            if st == -1:
                outcome = {1: 0.0, 2: 0.0}
                break
            if st:
                outcome = {st: 1.0, other_(st): -1.0}
                break
            is_teacher_turn = (b.to_move == _B) == teacher_is_black
            agent = teacher if is_teacher_turn else rnd
            steps.append((b.cells.copy(), b.to_move))
            b.play(*agent.move(b))
        label_board = Board(n)
        for cells, mover in steps:
            label_board.cells = cells
            label_board.to_move = mover
            label_board.last = None
            # candidates() treats an empty history as an empty board; rebuild
            # it from the stones so the labeler sees the real position
            label_board.history = [
                (int(r), int(c)) for r, c in np.argwhere(cells != 0)
            ]
            try:
                mv = labeler.move(label_board)
            except RuntimeError:
                continue
            x = np.zeros((4, n, n), dtype=np.int8)
            x[0] = cells == mover
            x[1] = cells == (3 - mover)
            x[3] = 1 if mover == 1 else 0
            pi = np.zeros(n * n, dtype=np.float32)
            pi[mv[0] * n + mv[1]] = 1.0
            data.append((x, pi, float(outcome[mover])))
    return data


def other_(p):
    return 3 - p


# ---------------- training steps ----------------

def augment_batch(xs, pis, rng):
    """Apply one random symmetry to the whole batch (cheap and effective).
    `pis` is (B, n*n) flattened policy targets."""
    k = int(rng.integers(4))
    flip = bool(rng.integers(2))
    n = xs.shape[-1]
    xs = torch.rot90(xs, k, dims=(2, 3))
    pis = torch.rot90(pis.view(-1, n, n), k, dims=(1, 2))
    if flip:
        xs = torch.flip(xs, dims=(3,))
        pis = torch.flip(pis, dims=(2,))
    return xs.contiguous(), pis.reshape(-1, n * n).contiguous()


def train_on_buffer(net, opt, buffer, device, epochs, batch, rng, value_w=1.0):
    net.train()
    idx_all = np.arange(len(buffer))
    stats = {"loss": 0.0, "p_loss": 0.0, "v_loss": 0.0, "n": 0}
    for _ in range(epochs):
        rng.shuffle(idx_all)
        for i in range(0, len(idx_all) - batch + 1, batch):
            idx = idx_all[i:i + batch]
            xs = torch.from_numpy(np.stack([buffer[j][0] for j in idx])).float().to(device)
            pis = torch.from_numpy(np.stack([buffer[j][1] for j in idx])).to(device)
            zs = torch.tensor([buffer[j][2] for j in idx], dtype=torch.float32,
                              device=device)
            xs, pis = augment_batch(xs, pis, rng)
            logits, v = net(xs)
            logp = F.log_softmax(logits, dim=1)
            p_loss = -(pis.flatten(1) * logp).sum(1).mean()
            v_loss = F.mse_loss(v, zs)
            loss = p_loss + value_w * v_loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            bs = loss.item()
            stats["loss"] += bs
            stats["p_loss"] += p_loss.item()
            stats["v_loss"] += v_loss.item()
            stats["n"] += 1
    net.eval()
    n = max(1, stats["n"])
    return {k: stats[k] / n for k in ("loss", "p_loss", "v_loss")}


# ---------------- main ----------------

@torch.no_grad()
def val_top1(net, data, val_frac: float = 0.1, device: str = "cpu") -> float:
    """Top-1 policy accuracy on a held-out slice of the BC dataset."""
    net.eval()
    k = max(1, int(len(data) * val_frac))
    val = data[-k:]
    hit = tot = 0
    for i in range(0, len(val), 256):
        chunk = val[i:i + 256]
        xs = torch.from_numpy(np.stack([c[0] for c in chunk])).float().to(device)
        logits = net(xs)[0]
        empty = ((xs[:, 0] == 0) & (xs[:, 1] == 0)).flatten(1)
        pred = torch.argmax(logits + torch.where(empty, 0.0, -1e9), dim=1)
        target = torch.tensor([int(c[1].argmax()) for c in chunk], device=device)
        hit += int((pred == target).sum())
        tot += len(chunk)
    return hit / max(1, tot)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--board", type=int, default=15)
    ap.add_argument("--out", default="checkpoints")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 8) - 4))
    ap.add_argument("--hours", type=float, default=1.5, help="total time budget")
    ap.add_argument("--bc-games", type=int, default=60)
    ap.add_argument("--skip-bc", action="store_true", help="pure self-play RL route")
    ap.add_argument("--games-per-iter", type=int, default=24)
    ap.add_argument("--sims", type=int, default=64)
    ap.add_argument("--buffer-cap", type=int, default=60_000)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4,
                    help="RL-stage LR (BC uses 2e-3); keep low to avoid drift")
    ap.add_argument("--gate-games", type=int, default=12, help="new vs best (both colors)")
    ap.add_argument("--teacher-eval-games", type=int, default=30)
    ap.add_argument("--stop-winrate", type=float, default=1.01,
                    help="stop when black-vs-teacher winrate exceeds this")
    ap.add_argument("--ch", type=int, default=48)
    ap.add_argument("--blocks", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", default="")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    net = GomokuNet(n=args.board, ch=args.ch, blocks=args.blocks).to(args.device)
    print(f"[init] params={count_params(net)/1e3:.0f}k device={args.device} "
          f"workers={args.workers}", flush=True)

    best = net
    start_iter = 1
    if args.resume:
        ck = torch.load(args.resume, map_location="cpu", weights_only=True)
        net.load_state_dict(ck["model"])
        best = load_net(args.resume, args.board, args.device)
        start_iter = ck.get("iter", 0) + 1
        print(f"[resume] iter {start_iter}", flush=True)

    buffer: list = []
    if args.resume and start_iter > 1:
        # anchor the replay buffer with teacher data so the policy keeps a
        # stable imitation signal while self-play data accumulates
        buffer.extend(bc_dataset(args.bc_games, args.board, args.seed))
        print(f"[resume] seeded buffer with {len(buffer)} teacher positions", flush=True)

    # ---------- stage 1: behavior cloning ----------
    if not args.skip_bc and start_iter == 1:
        t = time.time()
        data = bc_dataset(args.bc_games, args.board, args.seed)
        opt = torch.optim.Adam(net.parameters(), lr=2e-3, weight_decay=1e-4)
        idx = np.arange(len(data))
        net.train()
        for ep in range(24):
            rng.shuffle(idx)
            for i in range(0, len(idx) - args.batch + 1, args.batch):
                bidx = idx[i:i + args.batch]
                xs = torch.from_numpy(np.stack([data[j][0] for j in bidx])).float().to(args.device)
                pis = torch.from_numpy(np.stack([data[j][1] for j in bidx])).to(args.device)
                zs = torch.tensor([data[j][2] for j in bidx], dtype=torch.float32,
                                  device=args.device)
                xs, pis = augment_batch(xs, pis, rng)
                logits, v = net(xs)
                loss = -(pis.flatten(1) * F.log_softmax(logits, 1)).sum(1).mean() \
                    + F.mse_loss(v, zs)
                opt.zero_grad()
                loss.backward()
                opt.step()
        net.eval()
        vacc = val_top1(net, data, device=args.device)
        print(f"[bc] {len(data)} positions in {time.time()-t:.0f}s, "
              f"held-out teacher-move top-1 acc={vacc:.2f}", flush=True)
        buffer.extend(data[: args.buffer_cap])
        best = GomokuNet(n=args.board, ch=args.ch, blocks=args.blocks).to(args.device)
        best.load_state_dict(net.state_dict())
        best.eval()
        save_ckpt(best, args.out, "best", 0, {"stage": "bc"}, args.ch, args.blocks)

    net_spec = lambda: ("net", weights_bytes(best), args.sims, args.ch, args.blocks)
    heuristic_spec = ("heuristic",)

    # ---------- stage 2: AlphaZero self-play ----------
    opt = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-4)
    hp = os.path.join(args.out, "history.json")
    history = []
    if os.path.exists(hp):
        try:
            with open(hp) as f:
                history = json.load(f)
        except (json.JSONDecodeError, OSError):
            history = []
    it = start_iter
    while (time.time() - t0) / 3600 < args.hours:
        t_iter = time.time()

        # 1) self-play with the CURRENT net (explore) across workers
        w = weights_bytes(net)
        games = parallel_selfplay(w, {"sims": args.sims, "temp_moves": 8,
                                    "ch": args.ch, "blocks": args.blocks,
                                      "temperature": 1.0},
                                  args.games_per_iter, args.workers,
                                  args.board, "cpu", args.sims)
        new_pos = sum(len(g) for g in games)
        for g in games:
            buffer.extend(g)
        del buffer[: max(0, len(buffer) - args.buffer_cap)]

        # 2) train
        losses = train_on_buffer(net, opt, buffer, args.device,
                                 args.epochs, args.batch, rng)

        # 3) gate: candidate vs best (both colors) + black vs teacher
        cand = ("net", weights_bytes(net), args.sims, args.ch, args.blocks)
        gate = match_parallel(cand, net_spec(), args.gate_games, args.workers,
                              args.board, "cpu", seed0=1000 * it)
        gate_s = score(gate)
        te = match_parallel(cand, heuristic_spec, args.teacher_eval_games,
                            args.workers, args.board, "cpu", seed0=7000 * it)
        te_black = te["a_black"]
        te_wr = (te_black[0] + 0.5 * te_black[2]) / max(1, sum(te_black))

        promoted = gate_s >= 0.55 or it == start_iter
        if promoted:
            best = GomokuNet(n=args.board, ch=args.ch, blocks=args.blocks).to(args.device)
            best.load_state_dict(net.state_dict())
            best.eval()
        row = {"iter": it, "buffer": len(buffer), "new_pos": new_pos,
               "loss": round(losses["loss"], 4), "p": round(losses["p_loss"], 4),
               "v": round(losses["v_loss"], 4), "gate": round(gate_s, 3),
               "vs_teacher_black": round(te_wr, 3),
               "promoted": promoted,
               "t": round(time.time() - t_iter, 1)}
        history.append(row)
        print("[rl] " + json.dumps(row), flush=True)
        with open(os.path.join(args.out, "history.json"), "w") as f:
            json.dump(history, f, indent=1)
        save_ckpt(best, args.out, "best", it, row, args.ch, args.blocks)
        save_ckpt(net, args.out, "latest", it, row, args.ch, args.blocks)

        if te_wr >= args.stop_winrate:
            print(f"[rl] reached stop-winrate {args.stop_winrate} — done", flush=True)
            break
        it += 1

    print(f"[done] total {(time.time()-t0)/60:.1f} min, iterations {it}", flush=True)


def save_ckpt(net, out_dir: str, name: str, it: int, row: dict,
              ch: int = 48, blocks: int = 4) -> str:
    path = os.path.join(out_dir, f"{name}.pt")
    torch.save({"model": net.state_dict(), "iter": it, "stats": row,
                "arch": {"ch": ch, "blocks": blocks}}, path)
    return path


@torch.no_grad()
def quick_teacher_accuracy(net, n: int, device: str, games: int = 6) -> float:
    """Held-out teacher-move top-1 accuracy on fresh teacher games."""
    import random
    net.eval()
    hit = tot = 0
    for g in range(games):
        b = Board(n)
        black = HeuristicPlayer(temperature=0.0, rng=random.Random(99 + g))
        white = HeuristicPlayer(temperature=0.0, rng=random.Random(199 + g))
        while len(b.history) < 40 and not b.status():
            agent = black if b.to_move == 1 else white
            mv = agent.move(b)
            from .board import encode
            x = torch.from_numpy(encode(b)).unsqueeze(0).to(device)
            logits = net(x)[0].flatten()
            cand = b.candidates(radius=2)
            mask = torch.full((n * n,), -1e9, device=logits.device)
            for r, c in cand:
                mask[r * n + c] = 0.0
            pred = int(torch.argmax(logits + mask))
            hit += int(pred == mv[0] * n + mv[1])
            tot += 1
            b.play(*mv)
    return hit / max(1, tot)


if __name__ == "__main__":
    main()
