"""Player wrappers: random, heuristic teacher, and the final NetPlayer
(policy-value net + MCTS + forced-move shortcuts + VCF proof search)."""
from __future__ import annotations

import numpy as np

from .board import Board, BLACK, other
from .heuristic import HeuristicPlayer
from .mcts import MCTSEngine
from .vcf import vcf_search


class RandomPlayer:
    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def move(self, board: Board) -> tuple[int, int]:
        cand = board.candidates(radius=2)
        return tuple(cand[int(self.rng.integers(len(cand)))])


class NetPlayer:
    """The full engine: shortcut forced moves, VCF proof, else MCTS.

    shortcut_forced=True is logically safe (immediate win / only-block).
    vcf_color: which color may claim proven VCF wins (default BLACK).
    sample_ply: for the first `sample_ply` plies of a game, sample among
    MCTS visits with `temperature` (used to vary openings in evaluation).
    """

    def __init__(self, net, device: str = "cpu", sims: int = 128,
                 use_vcf: bool = True, vcf_depth: int = 14,
                 vcf_time: float = 2.0, vcf_color: int | None = 1,
                 c_puct: float = 1.8, sample_ply: int = 0, temperature: float = 0.5,
                 fork_guard: bool = True, seed: int = 0):
        import torch
        self.net = net.to(device).eval()
        self.device = device
        self.mcts = MCTSEngine(self.net, device=device, c_puct=c_puct)
        self.sims = sims
        self.use_vcf = use_vcf
        self.vcf_depth = vcf_depth
        self.vcf_time = vcf_time
        self.vcf_color = vcf_color
        self.sample_ply = sample_ply
        self.temperature = temperature
        self.fork_guard = fork_guard
        self.rng = np.random.default_rng(seed)
        self.last_info: dict = {}

    def move(self, board: Board) -> tuple[int, int]:
        p = board.to_move
        info: dict = {}
        move = self._shortcut_or_search(board, p, info, ply=len(board.history))
        self.last_info = info
        return move

    def _shortcut_or_search(self, board: Board, p: int, info: dict, ply: int):
        my5 = board.five_points(p)
        if my5:
            info["kind"] = "win-now"
            return my5[0]
        opp5 = board.five_points(other(p))
        if opp5:
            info["kind"] = "block-five"
            return opp5[0]
        if self.use_vcf and (self.vcf_color is None or p == self.vcf_color):
            line = vcf_search(board, p, max_depth=self.vcf_depth, time_limit=self.vcf_time)
            if line:
                info["kind"] = "vcf-proof"
                info["vcf_len"] = len(line)
                return line[0]
        temp = self.temperature if ply < self.sample_ply else 0.0
        move, pi, q = self.mcts.run(board, self.sims, root_noise=False,
                                    temperature=temp, rng=self.rng)
        # fork guard: never allow the opponent an unstoppable four (a move that
        # creates >=2 five points), unless our own move is itself forcing
        if self.fork_guard and not board.four_completions(move[0], move[1], p):
            n = board.n
            opp = other(p)
            ranked = sorted(range(n * n), key=lambda i: -pi[i])[:6]
            ranked = [divmod(i, n) for i in ranked if pi[i] > 0]
            chosen = None
            for m in ranked:
                if not self._allows_fork(board, m, opp):
                    chosen = m
                    break
            if chosen is None:
                # every MCTS candidate loses by force: fall back to the cells
                # that defuse the opponent's fork moves (the fork point itself
                # or one of its four-completion points), ranked by MCTS prior
                defenses: set[tuple[int, int]] = set()
                for w in board.candidates(radius=1):
                    comps = board.four_completions(w[0], w[1], opp)
                    if len(comps) >= 2:
                        defenses.add(w)
                        defenses.update(comps)
                for m in sorted(defenses, key=lambda m: -pi[m[0] * n + m[1]]):
                    if not self._allows_fork(board, m, opp):
                        chosen = m
                        break
            if chosen is not None and chosen != move:
                info["fork_guard"] = f"{move}->{chosen}"
                move = chosen
        info["kind"] = "mcts"
        info["q"] = round(q, 4)
        return move

    def _allows_fork(self, board: Board, m: tuple[int, int], opp: int) -> bool:
        """True if playing m lets the opponent create >=2 five points at once
        (an unstoppable four) or leaves any immediate five point open."""
        board.play(*m)
        bad = bool(board.five_points(opp))
        if not bad:
            for w in board.candidates(radius=1):
                comps = board.four_completions(w[0], w[1], opp)
                if len(comps) >= 2:
                    bad = True
                    break
        board.undo()
        return bad


    @classmethod
    def from_ckpt(cls, ckpt: str, n: int = 15, device: str = "cpu",
                  sims: int = 200, vcf_depth: int = 14, vcf_color: int | None = BLACK):
        import torch
        from .model import GomokuNet
        net = GomokuNet(n=n)
        state = torch.load(ckpt, map_location="cpu", weights_only=True)
        net.load_state_dict(state["model"] if isinstance(state, dict) and "model" in state else state)
        return cls(net, device=device, sims=sims, vcf_depth=vcf_depth,
                   vcf_color=vcf_color)


def make_player(kind: str, n: int = 15, ckpt: str | None = None, device: str = "cpu",
                sims: int = 128, seed: int = 0, sample_ply: int = 0):
    """Factory used by arena/play: kind in {random, heuristic, net}."""
    if kind == "random":
        return RandomPlayer(seed)
    if kind == "heuristic":
        return HeuristicPlayer(temperature=0.0, rng=__import__("random").Random(seed))
    if kind == "net":
        import torch
        from .model import GomokuNet
        net = GomokuNet(n=n)
        state = torch.load(ckpt, map_location="cpu", weights_only=True)
        net.load_state_dict(state["model"] if "model" in state else state)
        return NetPlayer(net, device=device, sims=sims, seed=seed, sample_ply=sample_ply)
    raise ValueError(kind)
