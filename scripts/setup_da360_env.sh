#!/usr/bin/env bash
# Eval / Gradio demo env (name: da360).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

export PYTHONNOUSERSITE=1

if conda env list | awk '{print $1}' | grep -qx da360; then
  conda env update -f environment.yaml --prune
else
  conda env create -f environment.yaml
fi

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate da360

pip install -r requirements.txt

echo "Done. Run: conda activate da360"
