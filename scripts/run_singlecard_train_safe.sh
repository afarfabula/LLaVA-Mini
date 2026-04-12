#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export PATH="${PATH:-/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin}"
export PATH="/home/tiger/miniconda3/envs/llava-mini-train/bin:${PATH}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

PYTHON_BIN="${PYTHON_BIN:-/home/tiger/miniconda3/envs/llava-mini-train/bin/python}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/checkpoints/runs/singlecard_train_safe}"
DATA_PATHS="${DATA_PATHS:-${ROOT}/data/smoke/train.json}"
IMAGE_FOLDERS="${IMAGE_FOLDERS:-${ROOT}/data/smoke/images}"

mkdir -p "${OUTPUT_DIR}"

exec env \
  PYTHON_BIN="${PYTHON_BIN}" \
  OUTPUT_DIR="${OUTPUT_DIR}" \
  DATA_PATHS="${DATA_PATHS}" \
  IMAGE_FOLDERS="${IMAGE_FOLDERS}" \
  GPUS="${GPUS:-1}" \
  SINGLE_PROCESS="${SINGLE_PROCESS:-1}" \
  GROUP_BY_MODALITY_LENGTH="${GROUP_BY_MODALITY_LENGTH:-False}" \
  GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-False}" \
  BF16="${BF16:-False}" \
  TF32="${TF32:-False}" \
  MAX_STEPS="${MAX_STEPS:-100}" \
  NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}" \
  MODEL_MAX_LENGTH="${MODEL_MAX_LENGTH:-1024}" \
  COMPRESSOR_SIZE="${COMPRESSOR_SIZE:-1}" \
  RESOLUTION_RATIO="${RESOLUTION_RATIO:-1}" \
  PREFUSION_LAYER_NUM="${PREFUSION_LAYER_NUM:-4}" \
  /bin/bash "${ROOT}/scripts/train_solo_sft.sh"
