#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export DATA_PATHS="${DATA_PATHS:-${ROOT}/data/smoke/train.json}"
export IMAGE_FOLDERS="${IMAGE_FOLDERS:-${ROOT}/data/smoke/images}"
export GPUS="${GPUS:-1}"
export SINGLE_PROCESS="${SINGLE_PROCESS:-1}"
export MAX_STEPS="${MAX_STEPS:-1}"
export NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
export MODEL_MAX_LENGTH="${MODEL_MAX_LENGTH:-1024}"
export COMPRESSOR_SIZE="${COMPRESSOR_SIZE:-1}"
export RESOLUTION_RATIO="${RESOLUTION_RATIO:-1}"
export PREFUSION_LAYER_NUM="${PREFUSION_LAYER_NUM:-4}"
export OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/checkpoints/runs/smoke_sft_singlecard_singleproc}"

exec /bin/bash "${ROOT}/scripts/train_solo_sft.sh"
