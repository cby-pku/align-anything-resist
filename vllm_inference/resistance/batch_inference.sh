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
OUTPUT_ROOT_DIR="/mnt/shared-storage-user/zhoujiayi/boyuan/data_results/resistance"



# ============================================================================
# 配置参数（必需）
# ============================================================================
LABELS=("PKU-SafeRLHF" "alpaca" "truthfulqa")
MODEL_ROOT_DIR="/mnt/shared-storage-user/zhoujiayi/boyuan/model_results/resistance/main_process/main_process"
# 这一行的作用是收集所有LABEL变量对应的模型切片（slice）文件夹路径，存到MODEL_PATHS变量


for LABEL in "${LABELS[@]}"; do
    # 先用数组而不是字符串来收集 model paths，使其可正确遍历
    readarray -t MODEL_PATHS < <(ls -d ${MODEL_ROOT_DIR}/${LABEL}_*/slice_* 2>/dev/null)
    for MODEL_PATH in "${MODEL_PATHS[@]}"; do

        INPUT_FILE=$(ls "/mnt/shared-storage-user/zhoujiayi/boyuan/datasets/${LABEL}/resist_"*.json | head -n 1)

        MODEL_NAME="$(basename "$(dirname "${MODEL_PATH}")")_$(basename "${MODEL_PATH}")"
        if [[ "${MODEL_NAME}" == *"pythia"* ]]; then
            export VLLM_ATTENTION_BACKEND=XFORMERS
            export VLLM_USE_V1=0
        fi
        
        OUTPUT_DIR="${OUTPUT_ROOT_DIR}/${LABEL}/"

        echo "Input file: ${INPUT_FILE}"
        echo "Model path: ${MODEL_PATH}"
        echo "Model name: ${MODEL_NAME}"
        echo "Output dir: ${OUTPUT_DIR}"
        echo "Output name: ${MODEL_NAME}"

        if [ -f "${OUTPUT_DIR}/${MODEL_NAME}.json" ]; then
            echo "Output file already exists: ${OUTPUT_DIR}/${MODEL_NAME}.json"
            continue
        fi


        python resist_data_inference.py \
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
done



