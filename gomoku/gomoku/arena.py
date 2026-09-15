"""Arena: run evaluation matches and print a strength report.

Examples:
  python3 -m gomoku.arena --ckpt checkpoints/best.pt --opponents random heuristic net:checkpoints/iter_5.pt --games 100
  python3 -m gomoku.arena --ckpt checkpoints/best.pt --opponents heuristic --games 100 --sims 192
"""
from __future__ import annotations

import argparse
import json
import os
import time

from .eval import match_parallel, score

DRAW = -1


def spec_from_str(s: str, default_sims: int):
    """'random' | 'heuristic' | 'net:path[:sims]' -> player spec tuple."""
    if s == "random":
        return ("random", 0)
    if s == "heuristic":
        return ("heuristic",)
    if s.startswith("net:"):
        parts = s.split(":")
        path = parts[1]
        sims = int(parts[2]) if len(parts) > 2 else default_sims
        import torch as _t
        meta = _t.load(path, map_location="cpu", weights_only=True)
        arch = meta.get("arch", {}) if isinstance(meta, dict) else {}
        with open(path, "rb") as f:
            return ("net", f.read(), sims, arch.get("ch", 48), arch.get("blocks", 4))
    raise ValueError(f"unknown opponent '{s}'")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="our network checkpoint")
    ap.add_argument("--opponents", nargs="+", required=True,
                    help="random | heuristic | net:path[:sims]")
    ap.add_argument("--games", type=int, default=100, help="per opponent (half black/half white)")
    ap.add_argument("--sims", type=int, default=128, help="our MCTS simulations")
    ap.add_argument("--board", type=int, default=15)
    ap.add_argument("--workers", type=int, default=max(1, (__import__("os").cpu_count() or 8) - 4))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--vcf", action="store_true", default=True)
    ap.add_argument("--out", default="", help="optional json report path")
    args = ap.parse_args()

    import torch as _t
    meta = _t.load(args.ckpt, map_location="cpu", weights_only=True)
    arch = meta.get("arch", {}) if isinstance(meta, dict) else {}
    with open(args.ckpt, "rb") as f:
        ours = ("net", f.read(), args.sims, arch.get("ch", 48), arch.get("blocks", 4))

    report = {"ckpt": args.ckpt, "sims": args.sims, "board": args.board,
              "games_each": args.games, "rows": []}
    print(f"Arena: {args.ckpt} (sims={args.sims}) on {args.board}x{args.board}, "
          f"{args.games} games per opponent, workers={args.workers}\n")
    header = f"{'opponent':<34}{'W':>5}{'L':>5}{'D':>4}{'score':>8}{'as black':>10}{'avg plies':>10}"
    print(header)
    print("-" * len(header))
    for opp in args.opponents:
        spec = spec_from_str(opp, args.sims)
        t0 = time.time()
        res = match_parallel(ours, spec, args.games, args.workers,
                             args.board, args.device)
        w = res["a_black"][0] + res["a_white"][0]
        l = res["a_black"][1] + res["a_white"][1]
        d = res["a_black"][2] + res["a_white"][2]
        s = score(res)
        ab = res["a_black"]
        ab_wr = (ab[0] + 0.5 * ab[2]) / max(1, sum(ab))
        name = os.path.basename(opp.replace("net:", "net/").replace(":", "/"))
        print(f"{name:<34}{w:>5}{l:>5}{d:>4}{s:>8.3f}{ab_wr:>10.3f}"
              f"{res['avg_plies']:>10.1f}  ({time.time()-t0:.0f}s)")
        report["rows"].append({"opponent": opp, "w": w, "l": l, "d": d,
                               "score": round(s, 3), "black_wr": round(ab_wr, 3),
                               "avg_plies": round(res["avg_plies"], 1)})
    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=1)
        print(f"\nreport -> {args.out}")


if __name__ == "__main__":
    main()
