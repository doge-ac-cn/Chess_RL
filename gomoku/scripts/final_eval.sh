#!/bin/bash
# Final evaluation suite after training: strength arena + VCF win stats + export.
set -e
cd "$(dirname "$0")/.." || exit 1
CKPT=${1:-checkpoints/best.pt}
SIMS=${2:-128}

echo "================ 1. Arena: final engine vs baselines ================"
python3 -m gomoku.arena --ckpt "$CKPT" --sims "$SIMS" --workers 20 \
  --opponents random heuristic net:checkpoints/early_iter1.pt:128 \
  --games 100 --out checkpoints/final_arena.json

echo
echo "================ 2. Win-type breakdown (as BLACK, 30 games) ========="
python3 -m gomoku.final_stats --ckpt "$CKPT" --opp heuristic --games 20 --sims "$SIMS"

echo
echo "================ 3. Export mobile ONNX models ======================="
python3 -m gomoku.export_onnx --ckpt "$CKPT" --out mobile/models
