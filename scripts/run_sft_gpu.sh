#!/bin/bash
set -euo pipefail

ROOT=/home/tiger/zero_to_n_workspace/LLaVA-Mini
cd /home/tiger/zero_to_n_workspace/LLaVA-Mini

export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export PYTHONPATH=/home/tiger/zero_to_n_workspace/LLaVA-Mini:
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_DEBUG=warn
export OMP_NUM_THREADS=8
export WANDB_DISABLED=true

GPUS=2
NNODES=1
NODE_RANK=0
MASTER_ADDR=127.0.0.1
MASTER_PORT=29500

MODEL_NAME_OR_PATH=/home/tiger/zero_to_n_workspace/LLaVA-Mini/data/mock_model
VISION_TOWER=openai/clip-vit-large-patch14
PRETRAIN_MM_MLP_ADAPTER=
DATA_PATHS=/home/tiger/zero_to_n_workspace/LLaVA-Mini/data/mock_train.json
IMAGE_FOLDERS=/home/tiger/zero_to_n_workspace/LLaVA-Mini/data/mock_images
OUTPUT_DIR=/home/tiger/zero_to_n_workspace/LLaVA-Mini/data/mock_output_h20
PROMPT_VERSION=plain
DEEPSPEED_CONFIG=/home/tiger/zero_to_n_workspace/LLaVA-Mini/scripts/zero2.json
MAX_STEPS=3
PER_DEVICE_TRAIN_BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=1
DATALOADER_NUM_WORKERS=0

mkdir -p "$OUTPUT_DIR"

CMD=(torchrun
  --nproc_per_node="$GPUS"
  --nnodes="$NNODES"
  --node_rank="$NODE_RANK"
  --master_addr="$MASTER_ADDR"
  --master_port="$MASTER_PORT"
  llavamini/train/train.py
  --deepspeed "$DEEPSPEED_CONFIG"
  --model_name_or_path "$MODEL_NAME_OR_PATH"
  --version "$PROMPT_VERSION"
  --data_paths "$DATA_PATHS"
  --image_folders "$IMAGE_FOLDERS"
  --vision_tower "$VISION_TOWER"
  --mm_projector_type linear
  --mm_vision_select_layer -2
  --mm_vision_select_feature patch
  --mm_use_im_start_end False
  --mm_use_im_patch_token False
  --image_aspect_ratio square
  --tune_mm_mlp_adapter True
  --freeze_backbone True
  --bf16 True
  --tf32 True
  --output_dir "$OUTPUT_DIR"
  --num_train_epochs 1
  --per_device_train_batch_size "$PER_DEVICE_TRAIN_BATCH_SIZE"
  --per_device_eval_batch_size 1
  --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS"
  --evaluation_strategy no
  --save_strategy no
  --learning_rate 1e-4
  --weight_decay 0.0
  --warmup_ratio 0.0
  --lr_scheduler_type constant
  --logging_steps 1
  --model_max_length 128
  --gradient_checkpointing False
  --dataloader_num_workers "$DATALOADER_NUM_WORKERS"
  --lazy_preprocess True
  --report_to none
  --save_steps "$MAX_STEPS"
  --max_steps "$MAX_STEPS"
  --compressor_size 2
  --prefusion_layer_num 1
  --temporal_router_hidden_size 32)

if [ -n "$PRETRAIN_MM_MLP_ADAPTER" ]; then
  CMD+=(--pretrain_mm_mlp_adapter "$PRETRAIN_MM_MLP_ADAPTER")
fi

printf "Running command:
%s
" "${CMD[*]}"
exec "${CMD[@]}"
