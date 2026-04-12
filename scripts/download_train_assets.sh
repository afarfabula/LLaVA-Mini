#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

HF_BIN="${HF_BIN:-huggingface-cli}"
MODEL_REPO="${MODEL_REPO:-ICTNLP/llava-mini-llama-3.1-8b}"
VISION_REPO="${VISION_REPO:-openai/clip-vit-large-patch14-336}"
MODEL_DIR="${MODEL_DIR:-${ROOT}/checkpoints/hf/${MODEL_REPO}}"
VISION_DIR="${VISION_DIR:-${ROOT}/checkpoints/hf/${VISION_REPO}}"
MIN_FREE_GB="${MIN_FREE_GB:-120}"

check_free_space() {
  local target_path="$1"
  local min_free_gb="$2"
  local available_kb
  available_kb="$(df -Pk "${target_path}" | awk 'NR==2 {print $4}')"
  local available_gb=$((available_kb / 1024 / 1024))
  echo "Available space for ${target_path}: ${available_gb} GB"
  if (( available_gb < min_free_gb )); then
    echo "Refusing to download because free space is below ${min_free_gb} GB." >&2
    exit 1
  fi
}

mkdir -p "$(dirname "${MODEL_DIR}")" "$(dirname "${VISION_DIR}")"
check_free_space "$(dirname "${MODEL_DIR}")" "${MIN_FREE_GB}"

echo "Downloading model repo: ${MODEL_REPO}"
"${HF_BIN}" download "${MODEL_REPO}" --local-dir "${MODEL_DIR}"

check_free_space "$(dirname "${VISION_DIR}")" "${MIN_FREE_GB}"
echo "Downloading vision tower repo: ${VISION_REPO}"
"${HF_BIN}" download "${VISION_REPO}" --local-dir "${VISION_DIR}"

echo "Done."
echo "Model dir: ${MODEL_DIR}"
echo "Vision dir: ${VISION_DIR}"
