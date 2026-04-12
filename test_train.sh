#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${SCRIPT_DIR}"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export WANDB_DISABLED=true
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export ROOT

python3 - <<'PY'
import os
import runpy
import sys
root = os.environ["ROOT"]
if root not in sys.path:
    sys.path.insert(0, root)

sys.argv = [
    'train.py',
    '--model_name_or_path', f'{root}/data/mock_model',
    '--version', 'plain',
    '--data_paths', f'{root}/data/mock_train.json',
    '--image_folders', f'{root}/data/mock_images',
    '--vision_tower', 'openai/clip-vit-large-patch14',
    '--mm_projector_type', 'linear',
    '--mm_vision_select_layer', '-2',
    '--mm_vision_select_feature', 'patch',
    '--mm_use_im_start_end', 'False',
    '--mm_use_im_patch_token', 'False',
    '--image_aspect_ratio', 'square',
    '--tune_mm_mlp_adapter', 'True',
    '--freeze_backbone', 'True',
    '--bf16', 'False',
    '--fp16', 'False',
    '--bits', '16',
    '--output_dir', f'{root}/data/mock_output',
    '--num_train_epochs', '3',
    '--per_device_train_batch_size', '1',
    '--gradient_accumulation_steps', '1',
    '--evaluation_strategy', 'no',
    '--save_strategy', 'steps',
    '--learning_rate', '1e-4',
    '--weight_decay', '0.0',
    '--warmup_ratio', '0.0',
    '--lr_scheduler_type', 'constant',
    '--logging_steps', '1',
    '--tf32', 'False',
    '--model_max_length', '128',
    '--gradient_checkpointing', 'False',
    '--dataloader_num_workers', '0',
    '--lazy_preprocess', 'True',
    '--report_to', 'none',
    '--save_steps', '3',
    '--max_steps', '3',
    '--compressor_size', '2',
    '--prefusion_layer_num', '1',
    '--temporal_router_hidden_size', '32',
]
runpy.run_module('llavamini.train.train', run_name='__main__')
PY
