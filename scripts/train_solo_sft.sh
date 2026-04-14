#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT}"

export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-warn}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export WANDB_DISABLED="${WANDB_DISABLED:-true}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

GPUS="${GPUS:-1}"
SINGLE_PROCESS="${SINGLE_PROCESS:-0}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"

MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-${ROOT}/checkpoints/hf/ICTNLP/llava-mini-llama-3.1-8b}"
VISION_TOWER="${VISION_TOWER:-${ROOT}/checkpoints/hf/openai/clip-vit-large-patch14-336}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/checkpoints/runs/smoke_sft}"
PROMPT_VERSION="${PROMPT_VERSION:-llava_llama_3_1}"
PRETRAIN_MM_MLP_ADAPTER="${PRETRAIN_MM_MLP_ADAPTER:-}"

: "${DATA_PATHS:?Please set DATA_PATHS to your training json/yaml file}"
: "${IMAGE_FOLDERS:?Please set IMAGE_FOLDERS to your image/video root directory}"

NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
MAX_STEPS="${MAX_STEPS:-10}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-0}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
MODEL_MAX_LENGTH="${MODEL_MAX_LENGTH:-2048}"
COMPRESSOR_SIZE="${COMPRESSOR_SIZE:-1}"
RESOLUTION_RATIO="${RESOLUTION_RATIO:-1}"
PREFUSION_LAYER_NUM="${PREFUSION_LAYER_NUM:-1}"
TEMPORAL_ROUTER_HIDDEN_SIZE="${TEMPORAL_ROUTER_HIDDEN_SIZE:-32}"
PYTHON_BIN="${PYTHON_BIN:-python}"
GROUP_BY_MODALITY_LENGTH="${GROUP_BY_MODALITY_LENGTH:-True}"
BF16="${BF16:-True}"
TF32="${TF32:-True}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-True}"
DDP_BACKEND="${DDP_BACKEND:-}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-1}"

mkdir -p "${OUTPUT_DIR}"

if [[ "${GPUS}" == "1" && -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES=0
fi

TRAIN_ENTRY=(llavamini/train/train.py
  --model_name_or_path "${MODEL_NAME_OR_PATH}"
  --version "${PROMPT_VERSION}"
  --data_paths "${DATA_PATHS}"
  --image_folders "${IMAGE_FOLDERS}"
  --vision_tower "${VISION_TOWER}"
  --mm_projector_type mlp2x_gelu
  --compressor_size "${COMPRESSOR_SIZE}"
  --resolution_ratio "${RESOLUTION_RATIO}"
  --prefusion_layer_num "${PREFUSION_LAYER_NUM}"
  --temporal_router_hidden_size "${TEMPORAL_ROUTER_HIDDEN_SIZE}"
  --mm_vision_select_layer -2
  --mm_vision_select_feature patch
  --mm_use_im_start_end False
  --mm_use_im_patch_token False
  --group_by_modality_length "${GROUP_BY_MODALITY_LENGTH}"
  --tune_mm_mlp_adapter True
  --freeze_backbone True
  --bf16 "${BF16}"
  --tf32 "${TF32}"
  --output_dir "${OUTPUT_DIR}"
  --num_train_epochs "${NUM_TRAIN_EPOCHS}"
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}"
  --per_device_eval_batch_size "${PER_DEVICE_EVAL_BATCH_SIZE}"
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}"
  --evaluation_strategy no
  --save_strategy steps
  --save_steps "${MAX_STEPS}"
  --save_total_limit "${SAVE_TOTAL_LIMIT}"
  --learning_rate "${LEARNING_RATE}"
  --max_grad_norm 0.5
  --weight_decay 0.0
  --warmup_ratio 0.03
  --lr_scheduler_type cosine
  --logging_steps 1
  --model_max_length "${MODEL_MAX_LENGTH}"
  --gradient_checkpointing "${GRADIENT_CHECKPOINTING}"
  --dataloader_num_workers "${DATALOADER_NUM_WORKERS}"
  --lazy_preprocess True
  --report_to none
  --max_steps "${MAX_STEPS}")

if [[ -n "${DDP_BACKEND}" ]]; then
  TRAIN_ENTRY+=(--ddp_backend "${DDP_BACKEND}")
fi


if [[ "${GPUS}" == "1" && "${SINGLE_PROCESS}" == "1" ]]; then
  CMD=("${PYTHON_BIN}" -u "${TRAIN_ENTRY[@]}")
else
  CMD=(
    torchrun
    --nproc_per_node="${GPUS}"
    --nnodes="${NNODES}"
    --node_rank="${NODE_RANK}"
    --master_addr="${MASTER_ADDR}"
    --master_port="${MASTER_PORT}"
    "${TRAIN_ENTRY[@]}"
  )
fi

if [[ -n "${DEEPSPEED_CONFIG}" ]]; then
  CMD+=(--deepspeed "${DEEPSPEED_CONFIG}")
fi

if [[ -n "${PRETRAIN_MM_MLP_ADAPTER}" ]]; then
  CMD+=(--pretrain_mm_mlp_adapter "${PRETRAIN_MM_MLP_ADAPTER}")
fi

printf "Running command:\n%s\n" "${CMD[*]}"
exec "${CMD[@]}"
