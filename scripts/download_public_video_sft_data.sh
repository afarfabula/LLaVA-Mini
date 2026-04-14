#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

DATA_ROOT="${DATA_ROOT:-${ROOT}/data/public_video_sft}"
RAW_ROOT="${RAW_ROOT:-${DATA_ROOT}/raw}"
ANNOTATION_ROOT="${ANNOTATION_ROOT:-${DATA_ROOT}/annotations}"
SOURCE_ROOT="${SOURCE_ROOT:-${DATA_ROOT}/sources}"

TARGET_MIX_GB="${TARGET_MIX_GB:-10}"
MIN_FREE_GB="${MIN_FREE_GB:-80}"
MAX_RAW_GB="${MAX_RAW_GB:-40}"

ACTIVITYNET_META_URL="${ACTIVITYNET_META_URL:-https://github.com/activitynet/ActivityNet/raw/refs/heads/master/Evaluation/data/activity_net.v1-3.min.json}"
YOUCOOK2_README_URL="${YOUCOOK2_README_URL:-http://youcook2.eecs.umich.edu/static/YouCookII/youcookii_readme.pdf}"

VIDEOINSTRUCT100K_REPO="${VIDEOINSTRUCT100K_REPO:-}"
MSRVTT_QA_URL="${MSRVTT_QA_URL:-}"
MSVD_QA_URL="${MSVD_QA_URL:-}"
ACTIVITYNET_CAPTION_URL="${ACTIVITYNET_CAPTION_URL:-}"
YOUCOOK2_ANNOTATION_URL="${YOUCOOK2_ANNOTATION_URL:-}"

mkdir -p "${RAW_ROOT}" "${ANNOTATION_ROOT}" "${SOURCE_ROOT}"

available_gb() {
  python3 - <<'PY' "$ROOT"
import os, sys
st = os.statvfs(sys.argv[1])
avail = st.f_bavail * st.f_frsize
print(int(avail / 1024 / 1024 / 1024))
PY
}

used_raw_gb() {
  if [[ ! -d "${RAW_ROOT}" ]]; then
    echo 0
    return
  fi
  python3 - <<'PY' "$RAW_ROOT"
import os, sys
root = sys.argv[1]
total = 0
for dirpath, _, filenames in os.walk(root):
    for name in filenames:
        path = os.path.join(dirpath, name)
        try:
            total += os.path.getsize(path)
        except FileNotFoundError:
            pass
print(int(total / 1024 / 1024 / 1024))
PY
}

assert_space() {
  local avail raw
  avail="$(available_gb)"
  raw="$(used_raw_gb)"
  if (( avail < MIN_FREE_GB )); then
    echo "Refusing to continue: only ${avail}GB free, threshold is ${MIN_FREE_GB}GB." >&2
    exit 1
  fi
  if (( raw > MAX_RAW_GB )); then
    echo "Refusing to continue: raw download cache already uses ${raw}GB, threshold is ${MAX_RAW_GB}GB." >&2
    exit 1
  fi
}

download_if_missing() {
  local url="$1"
  local out="$2"
  if [[ -z "${url}" ]]; then
    return
  fi
  if [[ -f "${out}" ]]; then
    echo "exists ${out}"
    return
  fi
  assert_space
  echo "downloading ${url} -> ${out}"
  curl -L --fail --retry 5 --retry-delay 2 "${url}" -o "${out}"
}

download_hf_dataset() {
  local repo="$1"
  local dest="$2"
  if [[ -z "${repo}" ]]; then
    return
  fi
  assert_space
  mkdir -p "${dest}"
  huggingface-cli download --repo-type dataset "${repo}" --local-dir "${dest}" --resume-download
}

echo "DATA_ROOT=${DATA_ROOT}"
echo "TARGET_MIX_GB=${TARGET_MIX_GB}"
echo "MIN_FREE_GB=${MIN_FREE_GB}"
echo "MAX_RAW_GB=${MAX_RAW_GB}"
echo "available_gb=$(available_gb)"

download_if_missing "${ACTIVITYNET_META_URL}" "${ANNOTATION_ROOT}/activity_net.v1-3.min.json"
download_if_missing "${YOUCOOK2_README_URL}" "${ANNOTATION_ROOT}/youcookii_readme.pdf"
download_if_missing "${ACTIVITYNET_CAPTION_URL}" "${ANNOTATION_ROOT}/activitynet_captions.json"
download_if_missing "${YOUCOOK2_ANNOTATION_URL}" "${ANNOTATION_ROOT}/youcook2_annotations.tar.gz"
download_if_missing "${MSRVTT_QA_URL}" "${ANNOTATION_ROOT}/msrvtt_qa_train.json"
download_if_missing "${MSVD_QA_URL}" "${ANNOTATION_ROOT}/msvd_qa_train.json"
download_hf_dataset "${VIDEOINSTRUCT100K_REPO}" "${SOURCE_ROOT}/VideoInstruct100K"

echo "Done."
echo "Next:"
echo "  1. Unpack any downloaded annotation archives into ${ANNOTATION_ROOT}"
echo "  2. Normalize each dataset into processed JSON files"
echo "  3. Use prepare_video_sft_mix.py to build the 6/3/1GB mixture"
