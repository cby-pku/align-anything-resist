# CSV 日志记录功能说明

## 功能概述

现在训练过程会自动保存一份 CSV 格式的指标日志文件，**无论使用 WandB 还是 TensorBoard**。

## 更新内容

### 修改文件
- `align_anything/utils/logger.py`

### 新增功能
1. **自动创建 CSV 文件**：在 `output_dir` 目录下自动创建 `metrics.csv` 文件
2. **实时记录**：每次调用 `logger.log()` 时同步写入 CSV
3. **动态字段**：自动识别和添加新的指标列
4. **立即刷新**：每次写入后立即 flush，确保数据不丢失

## 文件位置

CSV 文件保存位置：
```
{output_dir}/metrics.csv
```

例如：
```
/mnt/shared-storage-user/zhoujiayi/boyuan/model_results/resistance/main_process/alpaca_qwen1.5-0.5B/
├── metrics.csv                  # ← 新增的 CSV 文件
├── events.out.tfevents.*       # TensorBoard 文件
├── arguments.yaml              # 配置备份
└── environ.txt                 # 环境变量备份
```

## CSV 文件格式

### 示例内容

对于 SFT 训练，CSV 文件包含以下列：

```csv
step,train/epoch,train/loss,train/lr,train/step
1,0.001,2.345,0.00002,1
2,0.002,2.234,0.00002,2
3,0.003,2.123,0.00002,3
...
```

对于 DPO 训练，包含更多列：

```csv
step,train/better_sample_reward,train/epoch,train/loss,train/lr,train/reward,train/reward_accuracy,train/reward_margin,train/step,train/worse_sample_reward
1,1.234,0.001,0.456,0.00002,1.123,0.789,0.111,1,0.987
2,1.345,0.002,0.445,0.00002,1.234,0.812,0.123,2,1.098
...
```

### 字段说明

- `step`：全局训练步数
- `train/loss`：训练损失
- `train/lr`：学习率
- `train/epoch`：当前 epoch（浮点数）
- `train/step`：训练步数（与 step 相同）
- 其他字段根据训练方法动态添加

## 使用示例

### 1. 不需要修改配置文件

只需要像往常一样运行训练脚本：

```bash
deepspeed \
    --master_port ${MASTER_PORT} \
    --module align_anything.trainers.text_to_text.sft \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --train_template ${TRAIN_TEMPLATE} \
    --train_datasets ${TRAIN_DATASETS} \
    --output_dir ${OUTPUT_DIR}
```

### 2. 查看 CSV 文件

训练完成后，可以直接用各种工具查看：

#### Python Pandas
```python
import pandas as pd

# 读取 CSV
df = pd.read_csv('/path/to/output_dir/metrics.csv')

# 查看前几行
print(df.head())

# 绘制 loss 曲线
import matplotlib.pyplot as plt
plt.plot(df['step'], df['train/loss'])
plt.xlabel('Step')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.savefig('loss_curve.png')
```

#### Excel 或 Google Sheets
直接导入 CSV 文件即可查看和分析

#### 命令行
```bash
# 查看文件内容
cat metrics.csv

# 查看前 10 行
head -n 10 metrics.csv

# 使用 column 格式化查看
column -t -s, metrics.csv | less
```

## 技术特性

### 1. 智能字段管理
- 自动检测新的指标
- 动态添加新列到 CSV
- 保持已有数据完整性

### 2. 数据安全
- 每次写入后立即 flush
- 避免因程序崩溃导致数据丢失
- 在主进程（rank 0）中写入，避免多进程冲突

### 3. 兼容性
- 同时支持 TensorBoard 和 WandB
- `log_type` 设置为 `none` 时也会记录 CSV（只要指定了 `output_dir`）
- 对原有代码无侵入性修改

## 注意事项

1. **CSV 文件只在主进程创建**：多卡训练时只有 rank 0 进程会写入
2. **需要指定 output_dir**：如果配置中 `output_dir: null`，则不会创建 CSV 文件
3. **实时写入**：文件会在训练过程中实时更新，可以随时查看
4. **字段顺序**：CSV 列按字母顺序排序，方便查找

## 对比其他日志方式

| 特性 | CSV | TensorBoard | WandB |
|------|-----|-------------|-------|
| 易读性 | ✅ 极高 | ⚠️ 需要启动服务 | ⚠️ 需要联网/服务 |
| 处理简单 | ✅ 任何工具 | ⚠️ 需要特殊库 | ⚠️ 需要 API |
| 版本控制 | ✅ Git 友好 | ❌ 二进制文件 | ❌ 云端存储 |
| 实时性 | ✅ 立即可见 | ✅ 实时 | ⚠️ 依赖网络 |
| 功能丰富度 | ⚠️ 仅标量 | ✅ 图表/直方图 | ✅ 功能最全 |
| 离线使用 | ✅ 完全 | ✅ 完全 | ❌ 需要联网 |

## 示例数据分析

```python
import pandas as pd
import matplotlib.pyplot as plt

# 读取数据
df = pd.read_csv('metrics.csv')

# 创建多子图
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Loss 曲线
axes[0, 0].plot(df['step'], df['train/loss'])
axes[0, 0].set_title('Training Loss')
axes[0, 0].set_xlabel('Step')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].grid(True)

# Learning Rate 曲线
axes[0, 1].plot(df['step'], df['train/lr'])
axes[0, 1].set_title('Learning Rate')
axes[0, 1].set_xlabel('Step')
axes[0, 1].set_ylabel('LR')
axes[0, 1].grid(True)

# Epoch 进度
axes[1, 0].plot(df['step'], df['train/epoch'])
axes[1, 0].set_title('Epoch Progress')
axes[1, 0].set_xlabel('Step')
axes[1, 0].set_ylabel('Epoch')
axes[1, 0].grid(True)

# 统计信息
axes[1, 1].text(0.1, 0.9, f"Total Steps: {df['step'].max()}", transform=axes[1, 1].transAxes)
axes[1, 1].text(0.1, 0.8, f"Final Loss: {df['train/loss'].iloc[-1]:.4f}", transform=axes[1, 1].transAxes)
axes[1, 1].text(0.1, 0.7, f"Min Loss: {df['train/loss'].min():.4f}", transform=axes[1, 1].transAxes)
axes[1, 1].text(0.1, 0.6, f"Mean Loss: {df['train/loss'].mean():.4f}", transform=axes[1, 1].transAxes)
axes[1, 1].axis('off')

plt.tight_layout()
plt.savefig('training_summary.png', dpi=300)
print("图表已保存为 training_summary.png")
```

## 问题排查

### Q: CSV 文件没有创建
**A**: 检查以下几点：
1. `output_dir` 是否正确指定（不是 `null`）
2. 是否有写入权限
3. 是否在主进程（rank 0）运行

### Q: CSV 文件中缺少某些列
**A**: 这是正常现象。CSV 会在第一次遇到新指标时动态添加列。缺失的值会显示为空。

### Q: 训练中断后 CSV 数据是否丢失
**A**: 不会。代码在每次写入后都执行 `flush()`，确保数据立即写入磁盘。

### Q: 多卡训练会创建多个 CSV 吗
**A**: 不会。使用 `@rank_zero_only` 装饰器确保只有主进程写入。

## 总结

✅ **自动启用**：无需修改配置  
✅ **实时保存**：训练过程中随时可查看  
✅ **易于分析**：支持所有主流数据分析工具  
✅ **安全可靠**：立即刷新，防止数据丢失  
✅ **兼容性好**：与现有日志系统共存  

现在你可以在训练的同时，实时查看 CSV 文件来监控训练进度，无需依赖 TensorBoard 或 WandB 的 Web 界面！

