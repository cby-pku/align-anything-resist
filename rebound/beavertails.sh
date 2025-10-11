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

TRAIN_TEMPLATE="Rebound"
TRAIN_SPLIT='train'
OUTPUT_ROOT_DIR='/mnt/shared-storage-user/zhoujiayi/boyuan/model_results/rebound/beavertails/'
export WANDB_MODE="offline"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 # 控制步数，延长到2倍的步数 


postive_list=("1k" "2k" "5k" "10k" "20k" "30k" "40k" "50k")
negative_list=("100" "200" "500" "1k" "2k" "5k" "10k")

MODEL_LIST=(
    "/mnt/shared-storage-user/zhoujiayi/models/pythia/pythia-1.4b"
    "/mnt/shared-storage-user/zhoujiayi/models/pythia/pythia-1b"
    "/mnt/shared-storage-user/zhoujiayi/models/pythia/pythia-2.8b"
    "/mnt/shared-storage-user/zhoujiayi/models/pythia/pythia-410m"
    "/mnt/shared-storage-user/zhoujiayi/models/pythia/pythia-6.9b"
    "/mnt/shared-storage-user/zhoujiayi/models/pythia/pythia-12b"
    "/mnt/shared-storage-user/zhoujiayi/models/allenai/OLMo-2-0425-1B"
    "/mnt/shared-storage-user/zhoujiayi/models/allenai/OLMo-2-1124-13B"
    "/mnt/shared-storage-user/zhoujiayi/models/allenai/OLMo-2-1124-7B"
)
for MODEL_NAME_OR_PATH in "${MODEL_LIST[@]}"; do
    MODEL_NAME=$(basename "${MODEL_NAME_OR_PATH}")

    # 正向微调 
    for pos_num in "${postive_list[@]}"; do
        TRAIN_DATASETS=/mnt/shared-storage-user/zhoujiayi/boyuan/datasets/preference_datasets/beavertails/safe/safe_${pos_num}.json
        OUTPUT_DIR=${OUTPUT_ROOT_DIR}/main_process/${MODEL_NAME}_safe_${pos_num}
        # Source the setup script
        source ./setup.sh

        # Execute deepspeed command
        deepspeed \
            --master_port ${MASTER_PORT} \
            --module align_anything.trainers.text_to_text.sft \
            --config_name sft_rebound \
            --model_name_or_path ${MODEL_NAME_OR_PATH} \
            --train_template ${TRAIN_TEMPLATE} \
            --train_datasets ${TRAIN_DATASETS} \
            --train_split ${TRAIN_SPLIT} \
            --output_dir ${OUTPUT_DIR} \
            --epochs 1 
    done

    for pos_num in "${postive_list[@]}"; do
        for neg_num in "${negative_list[@]}"; do
            FINAL_TRAIN_DATASETS=/mnt/shared-storage-user/zhoujiayi/boyuan/datasets/preference_datasets/beavertails/unsafe/unsafe_${neg_num}.json
            FINAL_OUTPUT_DIR=${OUTPUT_ROOT_DIR}/rebound/${MODEL_NAME}_safe_${pos_num}_unsafe_${neg_num}
            POSTIVE_MODEL_NAME_OR_PATH=${OUTPUT_ROOT_DIR}/main_process/${MODEL_NAME}_safe_${pos_num}/slice_end
            # Source the setup script
            source ./setup.sh

            # Execute deepspeed command
            deepspeed \
                --master_port ${MASTER_PORT} \
                --module align_anything.trainers.text_to_text.sft \
                --config_name sft_rebound \
                --model_name_or_path ${POSTIVE_MODEL_NAME_OR_PATH} \
                --train_template ${TRAIN_TEMPLATE} \
                --train_datasets ${FINAL_TRAIN_DATASETS} \
                --train_split ${TRAIN_SPLIT} \
                --output_dir ${FINAL_OUTPUT_DIR} \
                --epochs 1 
        done
    done
done