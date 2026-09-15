"""Interactive terminal Gomoku.

  python3 -m gomoku.play                      # human black vs AI (needs a checkpoint)
  python3 -m gomoku.play --color white        # human white, AI plays black
  python3 -m gomoku.play --ckpt checkpoints/best.pt --sims 256
  python3 -m gomoku.play --two-player         # local 2 players

In-game commands: undo (take back your last move), hint, resign, quit.
Moves: column letter + row number, e.g.  h8  (columns A..O, rows 1..15).
"""
from __future__ import annotations

import argparse
import string
import sys

from .board import Board, BLACK, WHITE, other
from .players import NetPlayer

SYM = {0: ".", 1: "●", 2: "○"}


def render(board: Board) -> str:
    n = board.n
    letters = string.ascii_uppercase[:n]
    head = "    " + " ".join(letters)
    lines = [head]
    for r in range(n):
        row = []
        for c in range(n):
            ch = SYM[int(board.cells[r, c])]
            if board.last == (r, c):
                ch = "◉" if board.cells[r, c] else ch
            row.append(ch)
        lines.append(f"{r + 1:>3} " + " ".join(row))
    lines.append(head)
    return "\n".join(lines)


def parse_move(s: str, n: int):
    s = s.strip().lower()
    if not s:
        return None
    try:
        if " " in s or "," in s:
            parts = s.replace(",", " ").split()
            r, c = int(parts[0]) - 1, int(parts[1]) - 1
        else:
            col = string.ascii_lowercase.index(s[0])
            r = int(s[1:]) - 1
            c = col
        if 0 <= r < n and 0 <= c < n:
            return (r, c)
    except (ValueError, IndexError):
        pass
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/best.pt")
    ap.add_argument("--color", choices=["black", "white"], default="black",
                    help="human's color")
    ap.add_argument("--board", type=int, default=15)
    ap.add_argument("--sims", type=int, default=200, help="AI MCTS simulations")
    ap.add_argument("--vcf-depth", type=int, default=14)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--two-player", action="store_true")
    ap.add_argument("--ai-vs-ai", action="store_true")
    args = ap.parse_args()

    human = BLACK if args.color == "black" else WHITE
    ai = NetPlayer.from_ckpt(args.ckpt, args.board, args.device,
                             sims=args.sims, vcf_depth=args.vcf_depth)
    board = Board(args.board)
    print(f"Gomoku {args.board}x{args.board} — you are "
          f"{'● black' if human == BLACK else '○ white'}; move like  h8; "
          f"commands: undo / hint / resign / quit\n")

    def ai_move():
        print("AI thinking ...", flush=True)
        mv = ai.move(board)
        info = ai.last_info
        kind = info.get("kind", "")
        board.play(*mv)
        note = {"vcf-proof": " [VCF forced win proven — game is over by force]",
                "win-now": " [five!]", "block-five": " [block]"}.get(kind, "")
        print(f"AI plays {string.ascii_lowercase[mv[1]]}{mv[0] + 1}{note}")

    if args.two_player:
        args.ai_vs_ai = False
    while True:
        print(render(board))
        st = board.status()
        if st == -1:
            print("Draw — board full.")
            return
        if st:
            winner = "● Black" if st == BLACK else "○ White"
            print(f"{winner} wins!")
            return
        to_move = "● Black" if board.to_move == BLACK else "○ White"
        print(f"{to_move} to move" + (f" (last: {board.last})" if board.last else ""))

        if args.ai_vs_ai or (not args.two_player and board.to_move != human):
            ai_move()
            continue

        raw = input("your move> ").strip().lower()
        if raw in ("quit", "exit", "q"):
            return
        if raw == "resign":
            print("You resigned. AI wins.")
            return
        if raw == "undo":
            if len(board.history) >= 2 and not args.two_player:
                board.undo(); board.undo()
                print("(took back your last move)")
            elif args.two_player and board.history:
                board.undo()
            continue
        if raw == "hint":
            hb = other(board.to_move)
            tmp = NetPlayer.from_ckpt(args.ckpt, args.board, args.device, sims=64)
            mv = tmp.move(board)
            print(f"hint: {string.ascii_lowercase[mv[1]]}{mv[0] + 1}")
            continue
        mv = parse_move(raw, args.board)
        if mv is None:
            print("  ? format: column+row, e.g. h8")
            continue
        if not board.legal(*mv):
            print("  ? that point is occupied")
            continue
        board.play(*mv)


if __name__ == "__main__":
    main()
