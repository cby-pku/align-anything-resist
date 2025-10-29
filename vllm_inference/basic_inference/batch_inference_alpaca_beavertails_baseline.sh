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
OUTPUT_ROOT_DIR="/mnt/shared-storage-user/zhoujiayi/boyuan/data_results/rebound/PKU-SafeRLHF"
INPUT_FILE="/mnt/shared-storage-user/zhoujiayi/boyuan/datasets/PKU-SafeRLHF/test.json"


# MODEL_ROOT_DIR="/mnt/shared-storage-user/zhoujiayi/boyuan/model_results/resistance/main_process/new_model"
# 这一行的作用是收集所有LABEL变量对应的模型切片（slice）文件夹路径，存到MODEL_PATHS变量

MODEL_LIST=(
    "/mnt/shared-storage-user/zhoujiayi/jiayi/alpaca-llama-3-1/slice_end/"
)



for MODEL_PATH in "${MODEL_LIST[@]}"; do



    if [[ "${MODEL_NAME}" == *"pythia"* ]]; then
        export VLLM_ATTENTION_BACKEND=XFORMERS
        export VLLM_USE_V1=0
    fi
    
    OUTPUT_DIR="/mnt/shared-storage-user/zhoujiayi/boyuan/datasets/PKU-SafeRLHF/"


    python rebound_data_inference.py \
        --input_file "${INPUT_FILE}" \
        --model_path "${MODEL_PATH}" \
        --output_dir "${OUTPUT_DIR}" \
        --model_name "test_baseline" \
        --tensor_parallel_size ${TENSOR_PARALLEL_SIZE} \
        --gpu_memory_utilization ${GPU_MEMORY_UTILIZATION} \
        --swap_space ${SWAP_SPACE} \
        --temperature ${TEMPERATURE} \
        --top_p ${TOP_P}
done



