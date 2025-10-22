# ==============================================================================
# 1. 基本配置 (基本不需要修改)
# ==============================================================================
MODEL_NAME_OR_PATH="/mnt/shared-storage-user/zhoujiayi/jiayi/align-anything-main/outputs/exp3-beavertails-qwen2-5-inst-rm/pos/slice_end" # 模型路径
TRAIN_TEMPLATE="PKUSafeRLHF_Eval" # 数据集模板
TRAIN_SPLIT="train" # 数据集拆分

# ==============================================================================
# 2. 核心路径配置 (请确认)
# ==============================================================================
# 数据集所在的母目录
PARENT_DATA_DIR="/mnt/shared-storage-user/zhoujiayi/boyuan/data_results/rebound/PKU-SafeRLHF"
# 总输出目录的根目录
OUTPUT_ROOT_DIR="/mnt/shared-storage-user/zhoujiayi/boyuan/rm_score_rebound/PKU-SafeRLHF" # 如果环境变量未设置，则默认为 ../outputs

# 检查数据集母目录是否存在
if [ ! -d "$PARENT_DATA_DIR" ]; then
    echo "错误: 数据集目录不存在: ${PARENT_DATA_DIR}"
    exit 1
fi

MODEL_NAME_LIST=(
    "OLMo-2-0425-1B"
)

# ==============================================================================
# 3. 环境与执行
# ==============================================================================
# For wandb online logging

for MODEL_NAME in ${MODEL_NAME_LIST[@]}; do
# 遍历母目录下的所有 .json 和 .jsonl 文件
    OUTPUT_DIR="${OUTPUT_ROOT_DIR}/${MODEL_NAME}"
    source ./setup.sh
    for TRAIN_DATASETS in ${PARENT_DATA_DIR}/${MODEL_NAME}/*.json; do
    # 跳过不存在的文件模式（避免 glob 未匹配时的错误）
    [ -f "$TRAIN_DATASETS" ] || continue

    # 执行 deepspeed 命令
    deepspeed \
         --master_port ${MASTER_PORT} \
         --module align_anything.trainers.text_to_text.rm_score \
         --model_name_or_path ${MODEL_NAME_OR_PATH} \
         --eval_template ${TRAIN_TEMPLATE} \
         --eval_datasets "${TRAIN_DATASETS}" \
         --eval_split ${TRAIN_SPLIT} \
         --output_dir "${OUTPUT_DIR}" \
         --eval_size 100 \
         --epochs 1
    done
done