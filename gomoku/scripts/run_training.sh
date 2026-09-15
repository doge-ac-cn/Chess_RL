#!/bin/bash
# Full 15x15 training: BC warm start -> AlphaZero self-play RL.
# Resumes from checkpoints/best.pt when present (keeps iterating).
cd "$(dirname "$0")/.." || exit 1
RESUME=""
if [ -f checkpoints/best.pt ] && [ "$1" != "--fresh" ]; then
  RESUME="--resume checkpoints/best.pt"
fi
exec python3 -m gomoku.train \
  --board 15 --hours 2.0 \
  --bc-games 150 --games-per-iter 32 --sims 64 --lr 3e-4 \
  --workers 24 --buffer-cap 60000 --epochs 2 --batch 256 \
  --gate-games 12 --teacher-eval-games 30 \
  $RESUME \
  --out checkpoints >> train.log 2>&1
