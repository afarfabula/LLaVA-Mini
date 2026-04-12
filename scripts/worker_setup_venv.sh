#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

VENV_DIR="${VENV_DIR:-${ROOT}/.runtime/venv}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
REUSE_SYSTEM_SITE_PACKAGES="${REUSE_SYSTEM_SITE_PACKAGES:-1}"
SKIP_TORCH_INSTALL="${SKIP_TORCH_INSTALL:-1}"

mkdir -p "$(dirname "${VENV_DIR}")"

if [[ ! -d "${VENV_DIR}" ]]; then
  if [[ "${REUSE_SYSTEM_SITE_PACKAGES}" == "1" ]]; then
    "${PYTHON_BIN}" -m venv --system-site-packages "${VENV_DIR}"
  else
    "${PYTHON_BIN}" -m venv "${VENV_DIR}"
  fi
fi

source "${VENV_DIR}/bin/activate"

python -V
SKIP_TORCH_INSTALL="${SKIP_TORCH_INSTALL}" bash "${ROOT}/scripts/setup_train_env.sh"
