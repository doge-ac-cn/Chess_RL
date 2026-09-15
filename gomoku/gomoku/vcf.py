"""VCF (Victory by Continuous Fours) exact prover.

Searches ONLY forcing moves for the attacker: every attacker move creates a
four (a direct five threat), so the defender's reply is forced. If the search
returns a line, the win is FORCED against any defense — a mathematical proof,
not a heuristic. Soundness rules:

  * if the attacker can complete five now -> proven, play it;
  * if the defender has ANY five point, the proof fails (defender would rather
    complete their own five) — conservative, keeps proofs sound;
  * a created four with >=2 distinct completion points is unstoppable;
  * with exactly one completion point the defender must block there; if that
    block simultaneously completes five for the defender, the branch is refuted.
"""
from __future__ import annotations

import time

from .board import Board, other
from .heuristic import cell_score

MAX_NODES_DEFAULT = 4000


class _Abort(Exception):
    pass


def vcf_search(board: Board, attacker: int, max_depth: int = 14,
               node_budget: int = MAX_NODES_DEFAULT,
               time_limit: float = 2.0) -> list[tuple[int, int]] | None:
    """Return the attacker's forcing moves of a proven win, else None."""
    deadline = time.time() + time_limit
    nodes = [0]

    def rec(depth: int) -> list[tuple[int, int]] | None:
        nodes[0] += 1
        if nodes[0] > node_budget or time.time() > deadline:
            raise _Abort
        me, opp = attacker, other(attacker)

        my5 = board.five_points(me)
        if my5:
            return [my5[0]]
        opp5 = board.five_points(opp)
        if opp5:
            if len(opp5) > 1:
                return None  # defender's unstoppable four beats any forcing line
            if depth <= 0:
                return None
            # attacker may only continue with a blocking four at the defender's
            # unique five point (defender has no counter-five afterwards)
            tries = []
            for m in opp5:
                comps = board.four_completions(m[0], m[1], me)
                if comps:
                    tries.append((m, comps))
            tries.sort(key=lambda t: -len(t[1]))
        else:
            if depth == 0:
                return None
            tries = []
            for m in board.candidates(radius=1):
                comps = board.four_completions(m[0], m[1], me)
                if comps:
                    tries.append((m, comps))
            # strongest forcing moves first: double fours, then aggressive cells
            tries.sort(key=lambda t: (-len(t[1]), -cell_score(board, t[0][0], t[0][1], me)))

        for m, comps in tries:
            board.play(*m)
            if len(comps) >= 2:
                board.undo()
                return [m]
            block = next(iter(comps))
            board.play(*block)
            refuted = board.winner_after(*block) == opp
            sub = None if refuted else rec(depth - 1)
            board.undo()
            board.undo()
            if sub is not None:
                return [m] + sub
        return None

    try:
        return rec(max_depth)
    except _Abort:
        return None
