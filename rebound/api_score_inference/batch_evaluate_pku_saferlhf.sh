#!/usr/bin/env bash
set -euo pipefail

# 配置路径（请按需修改）
BASE_FILE="/mnt/shared-storage-user/zhoujiayi/boyuan/datasets/PKU-SafeRLHF/test_baseline.json"
SOURCE_DIR="/mnt/shared-storage-user/zhoujiayi/boyuan/data_results/rebound/PKU-SafeRLHF"
OUTPUT_ROOT_DIR="/mnt/shared-storage-user/zhoujiayi/boyuan/rm_score_rebound/api_score_results"

# 可配置参数（可通过环境变量覆写）
MODEL="${MODEL:-gpt-4o}"
NUM_WORKERS="${NUM_WORKERS:-50}"
MAX_ITEMS_PER_FILE="${MAX_ITEMS_PER_FILE:-500}"

# API 配置（可选，通过环境变量传入）
export API_KEY=sk-UlMdWKVhtMad7nHrRENi7tnvwbCYNXf5Mk9qiTF1dWJvTLcx

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

EXTRA_ARGS=()
if [[ -n "${MAX_ITEMS_PER_FILE}" ]]; then
  EXTRA_ARGS+=("--max_items_per_file" "${MAX_ITEMS_PER_FILE}")
fi
if [[ -n "${API_BASE:-}" ]]; then
  EXTRA_ARGS+=("--api_base" "${API_BASE}")
fi
if [[ -n "${API_KEY:-}" ]]; then
  EXTRA_ARGS+=("--api_key" "${API_KEY}")
fi

echo "[INFO] Base file: ${BASE_FILE}"
echo "[INFO] Source dir: ${SOURCE_DIR}"
echo "[INFO] Output root: ${OUTPUT_ROOT_DIR}"
echo "[INFO] Model: ${MODEL} | Workers: ${NUM_WORKERS} | Max per file: ${MAX_ITEMS_PER_FILE}"

"${PYTHON_BIN}" "${SCRIPT_DIR}/evaluate_pku_saferlhf.py" \
  --base_file "${BASE_FILE}" \
  --source_dir "${SOURCE_DIR}" \
  --output_dir "${OUTPUT_ROOT_DIR}" \
  --model "${MODEL}" \
  --num_workers "${NUM_WORKERS}" \
  "${EXTRA_ARGS[@]}"

echo "[INFO] Done."


