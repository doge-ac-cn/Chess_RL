#!/bin/bash
# V3 model: 6 residual blocks x 96 channels (~1.25M params).
# V2 plateaued vs V1 at equal training (50/50 x2 nights) -> scale up.
cd "$(dirname "$0")/.." || exit 1
RESUME=""
if [ -f checkpoints_v3/best.pt ] && [ "$1" != "--fresh" ]; then
  RESUME="--resume checkpoints_v3/best.pt"
fi
exec python3 -m gomoku.train \
  --board 15 --hours 3.5 \
  --ch 96 --blocks 6 \
  --bc-games 300 --games-per-iter 20 --sims 96 --lr 2.5e-4 \
  --workers 20 --buffer-cap 80000 --epochs 2 --batch 256 \
  --gate-games 12 --teacher-eval-games 24 \
  $RESUME \
  --out checkpoints_v3 >> train_v3.log 2>&1
