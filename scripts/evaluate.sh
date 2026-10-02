#!/usr/bin/env bash
# Run DA360 on Matterport3D, Stanford2D3D, and Metropolis (metrics + 10 viz samples).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

PYTHON="${PYTHON:-python}"
GPU="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES="${GPU}"

run_one() {
  local ckpt="$1"
  local name="$2"
  echo "=== ${name} ==="
  "${PYTHON}" evaluate.py \
    --val_datasets matterport3d stanford2d3d metropolis \
    --model_path "checkpoints/${ckpt}" \
    --batch_size 1 \
    --alignment 1 \
    --model_name "${name}" \
    --save_samples
}

run_one "DA360_small.pth" "DA360_small"
run_one "DA360_base.pth" "DA360_base"
run_one "DA360_large.pth" "DA360_large"

echo "Done. Metrics under ${ROOT}/results/<dataset>/result_<model>.txt"
