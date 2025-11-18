# 模型回答质量评测系统使用说明

## 功能概述

该系统可以基于给定的system_prompt，评测两个回答之间的分数，并保存详细的评测记录。

## 主要功能

1. **自动批量评测**: 支持一次运行完成所有子文件夹中的jsonl文件评测
2. **数据量控制**: 可以指定每个文件的最大评测数量，不需要评测完所有问题
3. **并行处理**: 支持多线程并行评测，提高处理效率
4. **结果保存**: 按原有文件结构保存到新路径，保持目录结构一致
5. **分数提取**: 自动从评测回答中提取分数

## 使用方法

### 基本使用

```bash
# 使用默认配置运行
python evaluate_pku_saferlhf.py

# 快速测试（每个文件仅处理3条数据）
python evaluate_pku_saferlhf.py --quick-test
```

### 自定义参数

```bash
python evaluate_pku_saferlhf.py \
    --base_file "/data/share/projects/jiayi/resist-deploy/alpaca_llama_31_results_conference_test/test.jsonl" \
    --source_dir "/data/share/projects/jiayi/resist-deploy/alpaca_llama_31_v0_results_conference_test_lr_1e_5_b_8" \
    --output_dir "/data/share/projects/jiayi/resist-deploy/alpaca_llama_31_v0_results_conference_test_lr_1e_5_b_8_with_score" \
    --max_items_per_file 100 \
    --num_workers 100 \
    --model "gpt-4o"
```

### 测试模式

```bash
# 测试模式：每个文件仅处理5条数据
python evaluate_pku_saferlhf.py --test_mode
```

## 参数说明

- `--base_file`: 基准文件路径（original_llama_3_1.jsonl）
- `--source_dir`: 源数据目录
- `--output_dir`: 输出目录
- `--max_items_per_file`: 每个文件的最大评测数量（None表示全部）
- `--num_workers`: 并行工作线程数
- `--model`: 评测使用的模型
- `--api_key`: API密钥
- `--api_base`: API基础URL
- `--test_mode`: 测试模式开关

## 输出格式

评测结果保存为JSONL格式，每行一个JSON对象：

```json
{
    "prompt": "system\nYou are a helpful assistant.\nuser\n...",
    "response": "........",
    "score": 9.6875,
    "evaluation_response": "完整的API评测回答",
    "finish_reason": "stop"
}
```

## 系统架构

1. **数据加载**: 读取基准文件和所有子文件夹中的jsonl文件
2. **请求准备**: 为每对回答创建评测prompt
3. **并行评测**: 使用ThreadPoolExecutor并行调用API
4. **分数提取**: 从API回答中提取评分
5. **结果保存**: 按原有目录结构保存到新路径

## 注意事项

1. 确保API密钥和基础URL配置正确
2. 系统会自动创建输出目录结构
3. 支持缓存机制，避免重复调用API
4. 建议先使用测试模式验证配置

## 示例输出

```
INFO:__main__:=== 评测配置 ===
INFO:__main__:基准文件: /path/to/original_llama_3_1.jsonl
INFO:__main__:源数据目录: /path/to/source_directory
INFO:__main__:输出目录: /path/to/output_directory
INFO:__main__:评测模型: gpt-4o
INFO:__main__:并行线程数: 4
INFO:__main__:每文件最大数据量: 100
INFO:__main__:===============

INFO:__main__:加载基准数据: /path/to/original_llama_3_1.jsonl
INFO:__main__:成功加载 6275 条数据从文件
INFO:__main__:找到 104 个待评测文件
INFO:__main__:开始处理文件: pos_1000_neg_100.jsonl
INFO:__main__:成功评测 3 项
INFO:__main__:文件 pos_1000_neg_100.jsonl 平均分数: 6.33
INFO:__main__:所有文件评测完成
```
