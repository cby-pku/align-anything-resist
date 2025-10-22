#!/bin/bash

# ============================================================================
# 环境变量设置（解决 flex_attention 兼容性问题）
# ============================================================================

# ============================================================================
# 配置参数（可选，使用默认值）
# ============================================================================
TENSOR_PARALLEL_SIZE=4 
# 适配某些qwen 模型 
GPU_MEMORY_UTILIZATION=0.95
SWAP_SPACE=32
TEMPERATURE=0.1
TOP_P=0.95
OUTPUT_ROOT_DIR="/mnt/shared-storage-user/zhoujiayi/boyuan/data_results/rebound/imdb"
INPUT_FILE="/mnt/shared-storage-user/zhoujiayi/boyuan/datasets/preference_datasets/imdb/imdb_test_700.json"


# MODEL_ROOT_DIR="/mnt/shared-storage-user/zhoujiayi/boyuan/model_results/resistance/main_process/new_model"
# 这一行的作用是收集所有LABEL变量对应的模型切片（slice）文件夹路径，存到MODEL_PATHS变量

MODEL_LIST=()
for dir in /mnt/shared-storage-user/zhoujiayi/boyuan/model_results/rebound/imdb/rebound/*/; do
    slice_end_dir="${dir}/slice_end"
    if [ -d "$slice_end_dir" ]; then
        MODEL_LIST+=("$slice_end_dir")
    fi
done



for MODEL_PATH in "${MODEL_LIST[@]}"; do


    MODEL_NAME="$(basename "$(dirname "${MODEL_PATH}")")"
    if [[ "${MODEL_NAME}" == *"pythia"* ]]; then
        export VLLM_ATTENTION_BACKEND=XFORMERS
        export VLLM_USE_V1=0
    fi
    
    # 将 MODEL_NAME 按 '_safe_' 拆分，找前缀
    MAIN_NAME="${MODEL_NAME%%_safe_*}"
    OUTPUT_DIR="${OUTPUT_ROOT_DIR}/${MAIN_NAME}/"

    echo "Input file: ${INPUT_FILE}"
    echo "Model path: ${MODEL_PATH}"
    echo "Model name: ${MODEL_NAME}"
    echo "Output dir: ${OUTPUT_DIR}"
    echo "Output name: ${MODEL_NAME}"

    if [ -f "${OUTPUT_DIR}/${MODEL_NAME}.json" ]; then
        echo "Output file already exists: ${OUTPUT_DIR}/${MODEL_NAME}.json"
        continue
    fi


    python rebound_data_inference.py \
        --input_file "${INPUT_FILE}" \
        --model_path "${MODEL_PATH}" \
        --output_dir "${OUTPUT_DIR}" \
        --model_name "${MODEL_NAME}" \
        --tensor_parallel_size ${TENSOR_PARALLEL_SIZE} \
        --gpu_memory_utilization ${GPU_MEMORY_UTILIZATION} \
        --swap_space ${SWAP_SPACE} \
        --temperature ${TEMPERATURE} \
        --top_p ${TOP_P}
done



