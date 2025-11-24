#!/bin/bash

# Example usage:
# bash scripts/paloma_eval.sh /path/to/model /path/to/paloma/data /path/to/output

# if [ "$#" -lt 3 ]; then
#     echo "Usage: $0 <model_path> <data_root> <output_dir> [extra_args...]"
#     exit 1
# fi

MODEL_PATH="/mnt/shared-storage-user/zhoujiayi/boyuan/model_results/resist-collapse/1120_test_collapse_eval/safe_llama3.1-8b/slice_collapse_tmp_582"
DATA_ROOT="/mnt/shared-storage-user/zhoujiayi/boyuan/dataset/paloma"
OUTPUT_DIR="/mnt/shared-storage-user/zhoujiayi/boyuan/model_results/resist-collapse/1120_test_collapse_eval/safe_llama3.1-8b/slice_collapse_tmp_582"

export PYTHONPATH="${PYTHONPATH}:$(pwd)"

python /mnt/shared-storage-user/zhoujiayi/boyuan/align-anything-resist/align_anything/evaluation/eval_paloma_offline.py \
    --model_path "${MODEL_PATH}" \
    --data_root "${DATA_ROOT}" \
    --output_dir "${OUTPUT_DIR}" \
    --use_vllm 


# bash scripts/paloma_eval.sh \
#     /mnt/shared-storage-user/zhoujiayi/boyuan/model_results/resist-collapse/1120_test_collapse_eval/safe_llama3.1-8b/slice_collapse_tmp_500 \
#     /path/to/paloma_data \
#     ./eval_results \
#     --limit_per_task 100 \
#     --dtype bf16