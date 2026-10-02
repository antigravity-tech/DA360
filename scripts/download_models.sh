#!/usr/bin/env bash
# Download DA360 checkpoints from Hugging Face (hljiang/DA360).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

REPO_ID="${DA360_HF_REPO:-hljiang/DA360}"
PYTHON="${PYTHON:-python}"
mkdir -p checkpoints

require_hf_hub() {
  if ! "${PYTHON}" -c "import huggingface_hub" >/dev/null 2>&1; then
    echo "Error: huggingface_hub is not installed in this Python env." >&2
    echo "Run: pip install -r requirements.txt   (or: pip install 'huggingface_hub>=0.19.3,<0.25')" >&2
    exit 1
  fi
}

download_one() {
  local filename="$1"
  local dest="checkpoints/${filename}"
  if [[ -s "${dest}" ]]; then
    echo "skip (exists): ${dest}"
    return 0
  fi
  echo "download -> ${dest}"
  require_hf_hub
  "${PYTHON}" - <<PY
from huggingface_hub import hf_hub_download

hf_hub_download(
    repo_id="${REPO_ID}",
    filename="${filename}",
    local_dir="checkpoints",
)
PY
}

for name in DA360_small.pth DA360_base.pth DA360_large.pth; do
  download_one "${name}"
done

echo "Done. Checkpoints in ${ROOT}/checkpoints/"
