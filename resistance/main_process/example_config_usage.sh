#!/usr/bin/env bash
#
# 示例：如何使用不同的配置文件
#

MODEL_NAME_OR_PATH="/path/to/your/model"
TRAIN_TEMPLATE="Resistance"
TRAIN_SPLIT="train"
TRAIN_DATASETS="/path/to/dataset.json"
OUTPUT_DIR="/path/to/output"

# 方式1：使用 sft_resist.yaml 配置
deepspeed \
    --master_port 29500 \
    --module align_anything.trainers.text_to_text.sft \
    --config_name sft_resist \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --train_template ${TRAIN_TEMPLATE} \
    --train_datasets ${TRAIN_DATASETS} \
    --train_split ${TRAIN_SPLIT} \
    --output_dir ${OUTPUT_DIR} \
    --epochs 1

# 方式2：使用 sft_main_process.yaml 配置
deepspeed \
    --master_port 29500 \
    --module align_anything.trainers.text_to_text.sft \
    --config_name sft_main_process \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --train_template ${TRAIN_TEMPLATE} \
    --train_datasets ${TRAIN_DATASETS} \
    --train_split ${TRAIN_SPLIT} \
    --output_dir ${OUTPUT_DIR} \
    --epochs 1

# 方式3：使用默认的 sft.yaml 配置（不指定 config_name）
deepspeed \
    --master_port 29500 \
    --module align_anything.trainers.text_to_text.sft \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --train_template ${TRAIN_TEMPLATE} \
    --train_datasets ${TRAIN_DATASETS} \
    --train_split ${TRAIN_SPLIT} \
    --output_dir ${OUTPUT_DIR} \
    --epochs 1

