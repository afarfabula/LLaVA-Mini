#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/checkpoints/runs/smoke_sft_singlecard_singleproc}"
LOG_PATH="${LOG_PATH:-${OUTPUT_DIR}/smoke.log}"
PID_PATH="${PID_PATH:-${OUTPUT_DIR}/smoke.pid}"
EXIT_PATH="${EXIT_PATH:-${OUTPUT_DIR}/smoke.exitcode}"

mkdir -p "${OUTPUT_DIR}"
rm -f "${PID_PATH}"
rm -f "${EXIT_PATH}"

nohup setsid /bin/bash "${ROOT}/scripts/run_singlecard_smoke_capture.sh" >/dev/null 2>&1 &
echo "$!" >"${PID_PATH}"
echo "started_pid=$(cat "${PID_PATH}")"
echo "log_path=${LOG_PATH}"
echo "exit_path=${EXIT_PATH}"
