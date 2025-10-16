import os
import json
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 输入与输出基路径（绝对路径）
base_input_dir = '/Users/boyuan/local_projects/wx_workspace/nature-resist/codespace/rm_score_rebound'
base_output_dir = '/Users/boyuan/local_projects/wx_workspace/nature-resist/codespace/rm_score_visualization/beavertails'

# 需要可视化的模型名列表（可按需增减）
model_names = [
    'OLMo-2-0425-1B',
    'OLMo-2-1124-13B',
    'OLMo-2-1124-7B',
    'pythia-1.4b',
    'pythia-1b',
]

def capitalize_first_last(word):
    if len(word) > 1:
        return word[0].upper() + word[1:-1] + word[-1].upper()
    return word.upper()

def capitalize_first_last_letters(sentence):
    return ' '.join(capitalize_first_last(word) for word in sentence.split())

def parse_size_token(token):
    """将类似 '2k' / '1m' / '2000' 的字符串转换为整数。"""
    if token is None:
        return 0
    token = token.strip()
    if token.lower().endswith('k'):
        return int(float(token[:-1]) * 1000)
    if token.lower().endswith('m'):
        return int(float(token[:-1]) * 1000000)
    return int(token)

def vis(model_name):

    # 初始化数据列表
    data = []
    Model_Name = capitalize_first_last_letters(model_name)
    folder_path = os.path.join(base_input_dir, model_name)
    
    pretrained_score = 0.54  # 用于存储 Q1=0 且 Q2=0 时的平均分

    # 遍历文件夹下的所有JSON文件（如：OLMo-2-0425-1B_safe_1k_unsafe_2k.json）
    if not os.path.isdir(folder_path):
        return
    for filename in os.listdir(folder_path):
        if not filename.endswith('.json'):
            continue
        # 提取 safe 与 unsafe 的数量
        m_safe = re.search(r'safe_(\d+[kKmM]?)', filename)
        m_unsafe = re.search(r'unsafe_(\d+[kKmM]?)', filename)
        if not m_safe or not m_unsafe:
            continue
        Q1_value = parse_size_token(m_safe.group(1))
        Q2_value = parse_size_token(m_unsafe.group(1))
        # 读取JSON文件
        with open(os.path.join(folder_path, filename), 'r') as f:
            json_data = json.load(f)
        # 计算score键对应值的平均值
        scores = [item['score'] for item in json_data]
        if len(scores) == 0:
            continue
        average_score = float(np.mean(scores))
        if Q1_value > 0 and Q2_value > 0:
            data.append({'Q1': Q1_value, 'Q2': Q2_value, 'average_score': average_score})
    
    # 转换为DataFrame
    df = pd.DataFrame(data)
    if df.empty:
        return

    # 设置Seaborn样式
    sns.set(style="whitegrid", palette="deep")
    plt.figure(figsize=(9, 6))
    
    # 设置调色板和线条样式
    custom_palette = sns.color_palette("viridis", n_colors=len(df['Q1'].unique()))
    # 确保按数值大小排序 X 轴
    df = df.sort_values(by=['Q2', 'Q1'])
    ax = sns.lineplot(data=df, x='Q2', y='average_score', hue='Q1', marker='o', palette=custom_palette, linewidth=4.0)

    plt.axhline(pretrained_score, color='gray', linestyle='--', label='Pretrained', linewidth=4.0)
    
    # 调整 marker 大小
    for line in ax.lines:
        line.set_marker('o')
        line.set_markersize(10)  # 设置 marker 的大小
    
    x_unique = sorted(df['Q2'].unique())
    ax.set_xticks(x_unique)
    ax.set_xticklabels(x_unique, fontsize=18)
    plt.xticks(rotation=45)
    plt.xlabel('Number of Negative Data', fontsize=25)
    plt.ylabel('Average Positive Score', fontsize=25)
    plt.ylim([-2, 0])
    ax.tick_params(axis='y', labelsize=18)
    plt.legend(title='Number of Positive Data', fontsize=18, title_fontsize=18, loc='upper center', bbox_to_anchor=(0.5, 1.3), ncol=3)
    plt.tight_layout()
    os.makedirs(base_output_dir, exist_ok=True)
    plt.savefig(os.path.join(base_output_dir, f'{model_name}.pdf'), bbox_inches='tight')
    plt.close()
    
for name in model_names:
    vis(name)
