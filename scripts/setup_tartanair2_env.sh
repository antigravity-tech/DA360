#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export PYTHONNOUSERSITE=1

if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found." >&2
  exit 1
fi

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"

if [[ "${1:-}" == "--recreate" ]]; then
  conda env remove -n tartanair2 -y 2>/dev/null || true
fi

if conda env list | awk '{print $1}' | grep -qx tartanair2; then
  conda env update -f environment-tartanair2.yaml --prune
else
  conda env create -f environment-tartanair2.yaml
fi

conda activate tartanair2

mkdir -p "$CONDA_PREFIX/etc/conda/activate.d"
printf '%s\n' 'export PYTHONNOUSERSITE=1' >"$CONDA_PREFIX/etc/conda/activate.d/00-pythonnousersite.sh"

# Ignore ~/.config/pip if it sets install.user=true (conflicts with conda env).
pip_install() {
  PIP_CONFIG_FILE=/dev/null python -m pip install "$@"
}
pip_uninstall() {
  PIP_CONFIG_FILE=/dev/null python -m pip uninstall -y "$@" 2>/dev/null || true
}
pip_uninstall opencv-python opencv-python-headless opencv-contrib-python
pip_install -r requirements-tartanair2.txt
pip_install tartanair==1.4.0 --no-deps
# Re-pin after tartanair (nothing should upgrade, but keeps a broken partial env recoverable).
pip_install numpy==1.26.4 "opencv-contrib-python==4.11.0.86" "cupy-cuda12x>=13.6.0,<14"

python - <<'PY'
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2
from scipy.spatial.transform import Rotation
import tartanair
from tartanair.image_resampling.image_sampler import SixPlanarTorch
import cupy
import numpy as np
import torch
assert cv2.__file__ and hasattr(cv2, "INTER_LINEAR")
assert torch.cuda.is_available(), "torch CUDA not available (check driver vs cu118 torch)"
cupy.zeros(1)
torch.from_numpy(np.zeros(1, dtype=np.float32))
Rotation.from_quat([0, 0, 0, 1])
print("tartanair2 environment OK")
PY
