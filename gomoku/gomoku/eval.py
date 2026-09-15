"""Match/evaluation helpers: parallel game runner between two player specs."""
from __future__ import annotations

import hashlib
import io

import torch

from .board import Board, BLACK, WHITE, DRAW
from .heuristic import HeuristicPlayer
from .players import RandomPlayer

_W: dict = {}


def _spec_key(spec) -> str:
    kind = spec[0]
    if kind == "net":
        return "net-" + hashlib.md5(spec[1]).hexdigest()[:10]
    return f"{kind}-{spec[1] if len(spec) > 1 else 0}"


def _init_eval(n: int, device: str):
    torch.set_num_threads(1)
    _W.update(n=n, device=device, cache={})


def _build(spec):
    """spec: ('net', weights_bytes, sims) | ('heuristic',) | ('random', seed)."""
    kind = spec[0]
    key = _spec_key(spec)
    if key in _W["cache"]:
        return _W["cache"][key]
    if kind == "net":
        from .model import GomokuNet
        from .players import NetPlayer
        ch = spec[3] if len(spec) > 3 else 48
        blocks = spec[4] if len(spec) > 4 else 4
        net = GomokuNet(n=_W["n"], ch=ch, blocks=blocks)
        state = torch.load(io.BytesIO(spec[1]), map_location="cpu", weights_only=True)
        net.load_state_dict(state["model"] if isinstance(state, dict) and "model" in state else state)
        player = NetPlayer(net, device=_W["device"], sims=spec[2], seed=0,
                           sample_ply=2, temperature=0.3)
    elif kind == "heuristic":
        import random
        player = HeuristicPlayer(temperature=0.0, rng=random.Random(1))
    elif kind == "random":
        player = RandomPlayer(seed=spec[1])
    else:
        raise ValueError(kind)
    _W["cache"][key] = player
    return player


def _play_one(args):
    black_spec, white_spec, seed = args
    black = _build(black_spec)
    white = _build(white_spec)
    board = Board(_W["n"])
    while True:
        st = board.status()
        if st == DRAW:
            return DRAW, len(board.history)
        if st:
            return st, len(board.history)
        agent = black if board.to_move == BLACK else white
        board.play(*agent.move(board))


def match_parallel(black_spec, white_spec, games: int, n_workers: int,
                   n: int, device: str = "cpu", seed0: int = 0) -> dict:
    """Play `games` (even) — spec_a as black in half, as white in the other half."""
    import multiprocessing as mp

    half = games // 2
    tasks = [((black_spec, white_spec, seed0 + i) if i % 2 == 0
              else (white_spec, black_spec, seed0 + i))
             for i in range(games)]
    # note: for odd i the tuple order means white_spec plays black; we track below
    results = {"a_black": [0, 0, 0], "a_white": [0, 0, 0]}  # [win, loss, draw] for spec_a
    if n_workers <= 1:
        _init_eval(n, device)
        outs = [_play_one(t) for t in tasks]
    else:
        ctx = mp.get_context("fork")
        with ctx.Pool(n_workers, initializer=_init_eval,
                      initargs=(n, device)) as pool:
            outs = pool.map(_play_one, tasks)

    for i, (winner, plies) in enumerate(outs):
        a_is_black = (i % 2 == 0)
        bucket = results["a_black" if a_is_black else "a_white"]
        results.setdefault("plies", []).append(plies)
        if winner == DRAW:
            bucket[2] += 1
        elif (winner == BLACK) == a_is_black:
            bucket[0] += 1
        else:
            bucket[1] += 1
    results["avg_plies"] = sum(results["plies"]) / max(1, len(results["plies"]))
    return results


def score(res: dict) -> float:
    """Score of spec_a in [0,1] across both colors."""
    w = res["a_black"][0] + res["a_white"][0]
    l = res["a_black"][1] + res["a_white"][1]
    d = res["a_black"][2] + res["a_white"][2]
    total = max(1, w + l + d)
    return (w + 0.5 * d) / total
