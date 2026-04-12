#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

INSTALL_FLASH_ATTN="${INSTALL_FLASH_ATTN:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SKIP_TORCH_INSTALL="${SKIP_TORCH_INSTALL:-0}"
INSTALL_DEEPSPEED="${INSTALL_DEEPSPEED:-0}"
PIP_INSTALL_ARGS="${PIP_INSTALL_ARGS:-}"

echo "Repo root: ${ROOT}"
echo "Using python: ${PYTHON_BIN}"
echo "Skip torch install: ${SKIP_TORCH_INSTALL}"
echo "Install deepspeed: ${INSTALL_DEEPSPEED}"

cd "${ROOT}"

"${PYTHON_BIN}" - <<'PY'
import sys

major, minor = sys.version_info[:2]
print(f"Python version: {major}.{minor}")
if (major, minor) >= (3, 12):
    raise SystemExit(
        "Python 3.12+ is not recommended for this repo. "
        "Please create and activate a Python 3.10/3.11 conda env first."
    )
PY

"${PYTHON_BIN}" -m pip install ${PIP_INSTALL_ARGS} -U pip setuptools wheel

if [[ "${SKIP_TORCH_INSTALL}" == "1" ]]; then
  "${PYTHON_BIN}" -m pip install ${PIP_INSTALL_ARGS} -e . --no-deps
  "${PYTHON_BIN}" -m pip install ${PIP_INSTALL_ARGS} \
    "numpy<2" \
    "torch==2.1.2" \
    "torchvision==0.16.2" \
    "transformers==4.43.1" \
    "tokenizers==0.19.0" \
    "sentencepiece==0.1.99" \
    shortuuid \
    "accelerate==0.29.0" \
    "peft==0.11.1" \
    bitsandbytes \
    pydantic \
    "markdown2[all]" \
    "scikit-learn==1.2.2" \
    "gradio==5.9.1" \
    "gradio_client==1.5.2" \
    requests \
    "httpx==0.28.1" \
    uvicorn \
    fastapi \
    "einops==0.6.1" \
    "einops-exts==0.0.4" \
    "timm==0.6.13" \
    ninja \
    wandb \
    decord
  if [[ "${INSTALL_DEEPSPEED}" == "1" ]]; then
    "${PYTHON_BIN}" -m pip install ${PIP_INSTALL_ARGS} "deepspeed==0.12.6"
  fi
else
  "${PYTHON_BIN}" -m pip install ${PIP_INSTALL_ARGS} -e .
  if [[ "${INSTALL_DEEPSPEED}" == "1" ]]; then
    "${PYTHON_BIN}" -m pip install ${PIP_INSTALL_ARGS} -e ".[train]"
  fi
  "${PYTHON_BIN}" -m pip install ${PIP_INSTALL_ARGS} decord sentencepiece shortuuid
fi

if [[ "${INSTALL_FLASH_ATTN}" == "1" ]]; then
  "${PYTHON_BIN}" -m pip install ${PIP_INSTALL_ARGS} flash-attn --no-build-isolation
else
  echo "Skipping flash-attn install. Set INSTALL_FLASH_ATTN=1 to enable it."
fi

INSTALL_DEEPSPEED_ENV="${INSTALL_DEEPSPEED}" "${PYTHON_BIN}" - <<'PY'
import importlib
import os

mods = ["torch", "transformers", "peft", "bitsandbytes", "decord"]
if os.environ.get("INSTALL_DEEPSPEED_ENV") == "1":
    mods.append("deepspeed")
for name in mods:
    mod = importlib.import_module(name)
    print(name, getattr(mod, "__version__", "no_version"))
PY

"${PYTHON_BIN}" - <<'PY'
import torch

print("cuda_available", torch.cuda.is_available())
print("device_count", torch.cuda.device_count())
PY
