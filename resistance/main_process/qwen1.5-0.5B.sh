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


MODEL_NAME_OR_PATH="/mnt/shared-storage-user/zhoujiayi/models/qwen/Qwen1.5-0.5B" # model path
TRAIN_TEMPLATE="Resistance" # sft dataset template
TRAIN_SPLIT="train" # split the sft dataset

# export WANDB_API_KEY='0e77f7c02e33b86269ca2123964b9fefcf9c1a7a'
export WANDB_MODE="offline"
OUTPUT_ROOT_DIR='/mnt/shared-storage-user/zhoujiayi/boyuan/model_results/resistance/main_process'

DATASETS=('alpaca' 'PKU-SafeRLHF' 'truthfulqa')
for dataset_label in ${DATASETS[@]}; do
    DATASET_LABEL=$dataset_label

    TRAIN_DATASETS=$(ls /mnt/shared-storage-user/zhoujiayi/boyuan/datasets/${DATASET_LABEL}/train_*.json | head -n 1)

    RESIST_DATASETS=$(ls /mnt/shared-storage-user/zhoujiayi/boyuan/datasets/${DATASET_LABEL}/resist_*.json | head -n 1)
    echo 'Dataset label: '$DATASET_LABEL
    echo 'Train datasets: '$TRAIN_DATASETS
    echo 'Resist datasets: '$RESIST_DATASETS

    OUTPUT_DIR="${OUTPUT_ROOT_DIR}/${DATASET_LABEL}_qwen1.5-0.5B"


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