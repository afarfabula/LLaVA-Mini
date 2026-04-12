#!/bin/bash
set -euo pipefail

TARGET_PATH="${1:-/}"

echo "hostname=$(hostname)"
echo "python=$(command -v python3 || true)"
python3 -V
echo "pip=$(command -v pip3 || true)"
if command -v pip3 >/dev/null 2>&1; then
  pip3 -V
fi

echo "gpu_info:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo "disk_info:"
df -h "${TARGET_PATH}"

echo "python_packages:"
python3 - <<'PY'
import importlib

for name in ["torch", "transformers", "deepspeed", "peft", "bitsandbytes", "decord"]:
    try:
        mod = importlib.import_module(name)
        print(name, getattr(mod, "__version__", "no_version"))
    except Exception as exc:
        print(name, "MISSING", type(exc).__name__)
PY
