#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export PATH="${PATH:-/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin}"
export PATH="/home/tiger/miniconda3/envs/llava-mini-train/bin:${PATH}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

PYTHON_BIN="${PYTHON_BIN:-/home/tiger/miniconda3/envs/llava-mini-train/bin/python}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/checkpoints/runs/dualcard_train_gloo}"
DATA_PATHS="${DATA_PATHS:-${ROOT}/data/smoke/train_ddp8.json}"
IMAGE_FOLDERS="${IMAGE_FOLDERS:-${ROOT}/data/smoke/images}"
MASTER_PORT="${MASTER_PORT:-29851}"

mkdir -p "${OUTPUT_DIR}"

exec env \
  PYTHON_BIN="${PYTHON_BIN}" \
  OUTPUT_DIR="${OUTPUT_DIR}" \
  DATA_PATHS="${DATA_PATHS}" \
  IMAGE_FOLDERS="${IMAGE_FOLDERS}" \
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}" \
  GPUS="${GPUS:-2}" \
  SINGLE_PROCESS=0 \
  DDP_BACKEND="${DDP_BACKEND:-gloo}" \
  LLAVAMINI_DDP_BACKEND="${LLAVAMINI_DDP_BACKEND:-gloo}" \
  GROUP_BY_MODALITY_LENGTH="${GROUP_BY_MODALITY_LENGTH:-False}" \
  GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-False}" \
  BF16="${BF16:-False}" \
  TF32="${TF32:-False}" \
  MAX_STEPS="${MAX_STEPS:-4}" \
  NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}" \
  MODEL_MAX_LENGTH="${MODEL_MAX_LENGTH:-1024}" \
  COMPRESSOR_SIZE="${COMPRESSOR_SIZE:-1}" \
  RESOLUTION_RATIO="${RESOLUTION_RATIO:-1}" \
  PREFUSION_LAYER_NUM="${PREFUSION_LAYER_NUM:-4}" \
  MASTER_PORT="${MASTER_PORT}" \
  /bin/bash "${ROOT}/scripts/train_solo_sft.sh"
