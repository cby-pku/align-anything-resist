#!/bin/bash

# ============================================================================
# 配置参数（可选，使用默认值）
# ============================================================================
TENSOR_PARALLEL_SIZE=8
GPU_MEMORY_UTILIZATION=0.95
SWAP_SPACE=32
TEMPERATURE=0.1
TOP_P=0.95
OUTPUT_ROOT_DIR="/mnt/shared-storage-user/zhoujiayi/boyuan/dataset_process/data_results/resistance"



# ============================================================================
# 配置参数（必需）
# ============================================================================
INPUT_FILE="/mnt/shared-storage-user/zhoujiayi/boyuan/datasets/PKU-SafeRLHF/resist_200.json"
MODEL_PATH="/mnt/shared-storage-user/zhoujiayi/boyuan/model_results/resistance/main_process/main_process/PKU-SafeRLHF_qwen1.5-0.5B/slice_208"
MODEL_NAME="PKU-SafeRLHF_qwen1.5-0.5B_slice_208"


# MAX_TOKENS=2048  # 取消注释以设置最大 token 数

# ============================================================================
# 运行推理
# ============================================================================
python resist_data_inference.py \
    --input_file "${INPUT_FILE}" \
    --model_path "${MODEL_PATH}" \
    --output_root_dir "${OUTPUT_ROOT_DIR}" \
    --model_name "${MODEL_NAME}" \
    --tensor_parallel_size ${TENSOR_PARALLEL_SIZE} \
    --gpu_memory_utilization ${GPU_MEMORY_UTILIZATION} \
    --swap_space ${SWAP_SPACE} \
    --temperature ${TEMPERATURE} \
    --top_p ${TOP_P}
    # --max_tokens ${MAX_TOKENS}  # 取消注释以使用
    # --output_file "custom_output.json"  # 取消注释以自定义输出文件名

