#!/usr/bin/env bash
# Download DA360 benchmark zips from Hugging Face and extract under ./data/.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA="${ROOT}/data"
REPO_ID="${DA360_HF_DATASET:-hljiang/DA360}"
STAGING="${DATA}/.hf_DA360_dataset"
ARCHIVES="${STAGING}/archives"

usage() {
  cat <<EOF
Usage: $(basename "$0") [all|metropolis|matterport3d|stanford2d3d]

Download evaluation data from https://huggingface.co/datasets/${REPO_ID}
and extract to:
  ${DATA}/Metropolis/
  ${DATA}/Matterport3D/
  ${DATA}/Stanford2D3D/
EOF
}

TARGET="${1:-all}"
PYTHON="${PYTHON:-python}"

case "${TARGET}" in
  all|metropolis|matterport3d|stanford2d3d) ;;
  -h|--help) usage; exit 0 ;;
  *) echo "unknown target: ${TARGET}" >&2; usage; exit 1 ;;
esac

require_hf_hub() {
  if ! "${PYTHON}" -c "import huggingface_hub" >/dev/null 2>&1; then
    echo "Error: huggingface_hub is not installed in this Python env." >&2
    echo "Run: pip install -r requirements.txt   (or: pip install 'huggingface_hub>=0.19.3,<0.25')" >&2
    exit 1
  fi
}

download_zip() {
  local rel="$1"
  local dest_dir="$2"
  mkdir -p "${dest_dir}"
  if [[ -s "${dest_dir}/$(basename "${rel}")" ]]; then
    echo "skip download (exists): ${dest_dir}/$(basename "${rel}")"
    return 0
  fi
  echo "download -> ${rel}"
  require_hf_hub
  "${PYTHON}" - <<PY
from huggingface_hub import hf_hub_download

hf_hub_download(
    repo_id="${REPO_ID}",
    filename="${rel}",
    repo_type="dataset",
    local_dir="${STAGING}",
)
PY
}

extract_zip() {
  local name="$1"
  local zip="${ARCHIVES}/${name}.zip"
  local out="${DATA}/${name}"
  if [[ ! -s "${zip}" ]]; then
    echo "Error: missing ${zip}" >&2
    exit 1
  fi
  if [[ -d "${out}" ]] && [[ -n "$(ls -A "${out}" 2>/dev/null || true)" ]]; then
    echo "skip extract (exists): ${out}"
    return 0
  fi
  echo "extract -> ${out}"
  unzip -q -o "${zip}" -d "${DATA}"
}

run_one() {
  local name="$1"
  download_zip "archives/${name}.zip" "${ARCHIVES}"
  extract_zip "${name}"
}

mkdir -p "${DATA}" "${ARCHIVES}"

if [[ "${TARGET}" == "all" || "${TARGET}" == "metropolis" ]]; then
  run_one Metropolis
fi
if [[ "${TARGET}" == "all" || "${TARGET}" == "matterport3d" ]]; then
  run_one Matterport3D
fi
if [[ "${TARGET}" == "all" || "${TARGET}" == "stanford2d3d" ]]; then
  run_one Stanford2D3D
fi

echo "Done. Evaluation data under ${DATA}/"
