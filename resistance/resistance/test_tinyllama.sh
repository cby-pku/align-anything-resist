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

TRAIN_TEMPLATE="Resistance"
TRAIN_SPLIT='train'
OUTPUT_ROOT_DIR='/mnt/shared-storage-user/zhoujiayi/boyuan/model_results/resistance/resistance_process'
export WANDB_MODE="offline"
export CUDA_VISIBLE_DEVICES=4,5,6,7 # 控制步数，延长到2倍的步数 
 
# ----- 需要更改 ---- 
# Label 里面的可以循环上
MODEL_ROOT_NAME='alpaca_tinyllama_1T' 
LABEL='alpaca'
INPUT_MODEL_ROOT_PATH='/mnt/shared-storage-user/zhoujiayi/boyuan/model_results/resistance/main_process/main_process/'
TRAIN_DATA_ROOT_PATH='/mnt/shared-storage-user/zhoujiayi/boyuan/data_results/resistance'/${LABEL}
SLICE_LIST=('slice_208' 'slice_416' 'slice_624' 'slice_832' 'slice_1040' 'slice_1248' 'slice_end')
TRAIN_SLICE_LIST=("${SLICE_LIST[@]}")

# ---- 需要更改 ------

for i in "${!SLICE_LIST[@]}"; do
    INPUT_MODEL_SLICE="${SLICE_LIST[$i]}"
    MODEL_NAME_OR_PATH="${INPUT_MODEL_ROOT_PATH}/${MODEL_ROOT_NAME}/${INPUT_MODEL_SLICE}"

    for j in "${!TRAIN_SLICE_LIST[@]}"; do
        TRAIN_SLICE="${TRAIN_SLICE_LIST[$j]}"
        # 跳过相同slice的情况
        if [ "$INPUT_MODEL_SLICE" == "$TRAIN_SLICE" ]; then
            continue
        fi

        # 输出目录中包含slice的位置索引
        OUTPUT_DIR="${OUTPUT_ROOT_DIR}/${MODEL_ROOT_NAME}/main_${i}_resist_${j}"
        TRAIN_DATASETS=${TRAIN_DATA_ROOT_PATH}/${MODEL_ROOT_NAME}_${TRAIN_SLICE}.json

        echo "Evaluating: SLICE = $SLICE (index $i), TRAIN_SLICE = $TRAIN_SLICE (index $j)"
        echo "MODEL_NAME_OR_PATH = $MODEL_NAME_OR_PATH"
        echo "OUTPUT_DIR = $OUTPUT_DIR"
        echo "TRAIN_DATASETS = $TRAIN_DATASETS"


        # Source the setup script
        source ./setup.sh

        # Execute deepspeed command
        deepspeed \
            --master_port ${MASTER_PORT} \
            --module align_anything.trainers.text_to_text.sft \
            --model_name_or_path ${MODEL_NAME_OR_PATH} \
            --train_template ${TRAIN_TEMPLATE} \
            --train_datasets ${TRAIN_DATASETS} \
            --train_split ${TRAIN_SPLIT} \
            --output_dir ${OUTPUT_DIR} \
            --epochs 1 


    done
done

