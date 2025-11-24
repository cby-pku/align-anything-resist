#!/bin/bash

# Example usage:
# bash scripts/paloma_eval.sh /path/to/model /path/to/paloma/data /path/to/output

# if [ "$#" -lt 3 ]; then
#     echo "Usage: $0 <model_path> <data_root> <output_dir> [extra_args...]"
#     exit 1
# fi

export VLLM_USE_V1=0

SLICE_DIR="/mnt/shared-storage-user/zhoujiayi/boyuan/model_results/resist-collapse/1120_test_collapse_eval/safe_llama3.1-8b"

for SLICE_PATH in "${SLICE_DIR}"/slice_*; do
    if [ -d "$SLICE_PATH" ] || [ -f "$SLICE_PATH" ]; then
        MODEL_PATH="$SLICE_PATH"
        MODEL_SLICE=$(basename "$SLICE_PATH")
        OUTPUT_DIR="${SLICE_DIR}/paloma_collapse/${MODEL_SLICE}"

        export PYTHONPATH="${PYTHONPATH}:$(pwd)"

        python /mnt/shared-storage-user/zhoujiayi/boyuan/align-anything-resist/align_anything/evaluation/eval_paloma_offline.py \
            --model_path "${MODEL_PATH}" \
            --data_root "/mnt/shared-storage-user/zhoujiayi/boyuan/dataset/paloma" \
            --output_dir "${OUTPUT_DIR}" 
            # --use_vllm 
    fi
done
 
