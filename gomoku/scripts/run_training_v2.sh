#!/bin/bash
# V2 model: 6 residual blocks x 64 channels (~562k params), stronger recipe.
# Resumes from checkpoints_v2/best.pt when present (accumulates across nights).
cd "$(dirname "$0")/.." || exit 1
RESUME=""
if [ -f checkpoints_v2/best.pt ] && [ "$1" != "--fresh" ]; then
  RESUME="--resume checkpoints_v2/best.pt"
fi
exec python3 -m gomoku.train \
  --board 15 --hours 3.0 \
  --ch 64 --blocks 6 \
  --bc-games 300 --games-per-iter 24 --sims 96 --lr 2.5e-4 \
  --workers 20 --buffer-cap 80000 --epochs 2 --batch 256 \
  --gate-games 12 --teacher-eval-games 24 \
  $RESUME \
  --out checkpoints_v2 >> train_v2.log 2>&1
