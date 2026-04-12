#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/checkpoints/runs/smoke_sft_singlecard_singleproc}"
LOG_PATH="${LOG_PATH:-${OUTPUT_DIR}/smoke.log}"

mkdir -p "${OUTPUT_DIR}"
exec /bin/bash "${ROOT}/scripts/run_singlecard_smoke.sh" >"${LOG_PATH}" 2>&1
