"""Final-report stats: play our engine as BLACK vs an opponent and tally how
games were won — in particular how many wins carry a VCF forced-win PROOF.

  python3 -m gomoku.final_stats --ckpt checkpoints/best.pt --opp heuristic --games 20
"""
from __future__ import annotations

import argparse
import collections

from .board import Board, BLACK, other
from .heuristic import HeuristicPlayer
from .players import NetPlayer
from .players import RandomPlayer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/best.pt")
    ap.add_argument("--opp", default="heuristic", choices=["heuristic", "random"])
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--sims", type=int, default=128)
    ap.add_argument("--board", type=int, default=15)
    ap.add_argument("--opp-seed0", type=int, default=0)
    args = ap.parse_args()

    net = NetPlayer.from_ckpt(args.ckpt, n=args.board, device="cpu", sims=args.sims)
    tally = collections.Counter()
    lengths = []

    for g in range(args.games):
        b = Board(args.board)
        import random
        opp = (HeuristicPlayer(temperature=0.0, rng=random.Random(args.opp_seed0 + g))
               if args.opp == "heuristic" else RandomPlayer(seed=args.opp_seed0 + g))
        our_info = None
        while not b.status():
            if b.to_move == BLACK:
                mv = net.move(b)
                our_info = dict(net.last_info)
                b.play(*mv)
            else:
                b.play(*opp.move(b))
        st = b.status()
        lengths.append(len(b.history))
        if st == BLACK:
            tally["win"] += 1
            kind = our_info["kind"] if our_info else "?"
            tally[f"last_move:{kind}"] += 1
            if our_info and our_info.get("kind") == "vcf-proof":
                tally["win_vcf_proven"] += 1
        elif st == other(BLACK):
            tally["loss"] += 1
        else:
            tally["draw"] += 1
        print(f"game {g+1:3d}: {'WIN ' if st==BLACK else ('LOSS' if st==other(BLACK) else 'DRAW')}"
              f"  plies={len(b.history):3d}  our last move: {our_info['kind'] if our_info else '-'}")

    print("\n==== summary (our engine as BLACK) ====")
    print(f"opponent: {args.opp}, games: {args.games}, sims: {args.sims}")
    for k, v in sorted(tally.items()):
        print(f"  {k:<22} {v}")
    print(f"  avg game length       {sum(lengths)/max(1,len(lengths)):.1f} plies")


if __name__ == "__main__":
    main()
