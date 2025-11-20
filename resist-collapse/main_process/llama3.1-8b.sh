#!/usr/bin/env bash
#
# Copyright 2025 PKU-Alignment Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


MODEL_NAME_OR_PATH="/mnt/shared-storage-user/zhoujiayi/models/llama/Llama-3.1-8B" # model path
TRAIN_TEMPLATE="Resistance" # sft dataset template
TRAIN_SPLIT="train" # split the sft dataset

# export WANDB_API_KEY='0e77f7c02e33b86269ca2123964b9fefcf9c1a7a'
export WANDB_MODE="offline"  # 关闭在线同步, 避免集群上网络依赖
OUTPUT_ROOT_DIR='/mnt/shared-storage-user/zhoujiayi/boyuan/model_results/resist-collapse/1120_test_collapse_eval'  # 训练输出根目录


TRAIN_DATASETS="/mnt/shared-storage-user/zhoujiayi/boyuan/datasets/PKU-SafeRLHF-new-jiayi/supervised/pos/train.jsonl"

OUTPUT_DIR="${OUTPUT_ROOT_DIR}/safe_llama3.1-8b"
# === PALOMA collapse 评测控制项 ===
ENABLE_COLLAPSE_EVAL=${ENABLE_COLLAPSE_EVAL:-"true"}            # true 时开启 PALOMA 评测
PALOMA_DATA_ROOT=${PALOMA_DATA_ROOT:-""}                        # 指向 HuggingFace 克隆的 paloma 数据根目录
COLLAPSE_NUM_EVALS=${COLLAPSE_NUM_EVALS:-10}                     # 训练全程平均切分运行几次评测, 0 表示不用平均分配
COLLAPSE_STEPS=${COLLAPSE_STEPS:-""}                            # 手动指定评测步数, 逗号分隔, eg "500,1000"
COLLAPSE_LIMIT_PER_TASK=${COLLAPSE_LIMIT_PER_TASK:-""}          # 单任务最多评测多少文档, 为空表示全量
COLLAPSE_MODEL_MAX_LENGTH=${COLLAPSE_MODEL_MAX_LENGTH:-""}      # 强制评测模型的上下文最大长度
COLLAPSE_CONTEXT_WINDOW=${COLLAPSE_CONTEXT_WINDOW:-""}          # 评测时每段输入长度, 不设则按模型默认
COLLAPSE_DTYPE=${COLLAPSE_DTYPE:-""}                            # 评测推理精度, 可填 bf16/fp16/fp32 等

# 组装 collapse 评测的额外参数, 仅在 ENABLE_COLLAPSE_EVAL=true 时追加
EXTRA_ARGS=()
if [[ "${ENABLE_COLLAPSE_EVAL}" == "true" ]]; then
  EXTRA_ARGS+=(--collapse_eval_cfgs.enabled True)
  if [[ -n "${PALOMA_DATA_ROOT}" ]]; then
    EXTRA_ARGS+=(--collapse_eval_cfgs.data_root "${PALOMA_DATA_ROOT}")
  fi
  if (( COLLAPSE_NUM_EVALS > 0 )); then
    EXTRA_ARGS+=(--collapse_eval_cfgs.num_passes "${COLLAPSE_NUM_EVALS}")
  fi
  if [[ -n "${COLLAPSE_STEPS}" ]]; then
    EXTRA_ARGS+=(--collapse_eval_cfgs.steps "${COLLAPSE_STEPS}")
  fi
  if [[ -n "${COLLAPSE_LIMIT_PER_TASK}" ]]; then
    EXTRA_ARGS+=(--collapse_eval_cfgs.limit_per_task "${COLLAPSE_LIMIT_PER_TASK}")
  fi
  if [[ -n "${COLLAPSE_MODEL_MAX_LENGTH}" ]]; then
    EXTRA_ARGS+=(--collapse_eval_cfgs.model_max_length "${COLLAPSE_MODEL_MAX_LENGTH}")
  fi
  if [[ -n "${COLLAPSE_CONTEXT_WINDOW}" ]]; then
    EXTRA_ARGS+=(--collapse_eval_cfgs.context_window "${COLLAPSE_CONTEXT_WINDOW}")
  fi
  if [[ -n "${COLLAPSE_DTYPE}" ]]; then
    EXTRA_ARGS+=(--collapse_eval_cfgs.dtype "${COLLAPSE_DTYPE}")
  fi
fi



# Source the setup script
source ./setup.sh

# Execute deepspeed command
CMD_ARGS=(
    --master_port "${MASTER_PORT}"
    --module align_anything.trainers.text_to_text.sft
    --config_name sft_collapse
    --model_name_or_path "${MODEL_NAME_OR_PATH}"
    --train_template "${TRAIN_TEMPLATE}"
    --train_datasets "${TRAIN_DATASETS}"
    --train_split "${TRAIN_SPLIT}"
    --output_dir "${OUTPUT_DIR}"
    --logger_cfgs.output_dir "${OUTPUT_DIR}"
    --epochs 1
)

CMD_ARGS+=("${EXTRA_ARGS[@]}")

deepspeed "${CMD_ARGS[@]}"

