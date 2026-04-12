#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/checkpoints/runs/smoke_sft_singlecard_singleproc}"
LOG_PATH="${LOG_PATH:-${OUTPUT_DIR}/smoke.log}"
EXIT_PATH="${EXIT_PATH:-${OUTPUT_DIR}/smoke.exitcode}"

mkdir -p "${OUTPUT_DIR}"
rm -f "${EXIT_PATH}"

if /bin/bash "${ROOT}/scripts/run_singlecard_smoke.sh" >"${LOG_PATH}" 2>&1; then
  echo 0 >"${EXIT_PATH}"
else
  status=$?
  echo "${status}" >"${EXIT_PATH}"
  exit "${status}"
fi
