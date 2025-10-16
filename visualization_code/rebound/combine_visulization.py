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
# model_names = [
#     'OLMo-2-0425-1B',
#     'OLMo-2-1124-7B',
#     'OLMo-2-1124-13B',
# ]

model_names = [
    'pythia-1.4b',
    'pythia-1b',
]
output_name = 'pythia_combined.pdf'

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

def load_model_df(model_name):
    # 读取单个模型的数据，返回包含 Q1/Q2/average_score 的 DataFrame，并附带 model 列
    data = []
    Model_Name = capitalize_first_last_letters(model_name)
    folder_path = os.path.join(base_input_dir, model_name)
    if not os.path.isdir(folder_path):
        return pd.DataFrame(columns=['Q1', 'Q2', 'average_score', 'model'])
    for filename in os.listdir(folder_path):
        if not filename.endswith('.json'):
            continue
        m_safe = re.search(r'safe_(\d+[kKmM]?)', filename)
        m_unsafe = re.search(r'unsafe_(\d+[kKmM]?)', filename)
        if not m_safe or not m_unsafe:
            continue
        Q1_value = parse_size_token(m_safe.group(1))
        Q2_value = parse_size_token(m_unsafe.group(1))
        with open(os.path.join(folder_path, filename), 'r') as f:
            json_data = json.load(f)
        scores = [item['score'] for item in json_data]
        if len(scores) == 0:
            continue
        average_score = float(np.mean(scores))
        if Q1_value > 0 and Q2_value > 0:
            data.append({'Q1': Q1_value, 'Q2': Q2_value, 'average_score': average_score, 'model': Model_Name})
    return pd.DataFrame(data)


def vis_combined(output_name='olmo_combined.pdf'):
    # 合并多模型在同一张图上：
    # - 每个模型是一簇：用 min-max 阴影带表示不同 Q1 曲线的包络
    # - 其均值折线作为该簇的代表
    sns.set(style="whitegrid", palette="deep")
    plt.figure(figsize=(10, 6))

    all_dfs = []
    for model_name in model_names:
        df_model = load_model_df(model_name)
        if not df_model.empty:
            all_dfs.append(df_model)

    if len(all_dfs) == 0:
        return

    df_all = pd.concat(all_dfs, ignore_index=True)
    ax = plt.gca()

    unique_models = df_all['model'].unique()
    colors = sns.color_palette('tab10', n_colors=len(unique_models))

    x_union = sorted(df_all['Q2'].unique())

    for color, model_label in zip(colors, unique_models):
        sub = df_all[df_all['model'] == model_label].copy()
        band = (
            sub.groupby('Q2')['average_score']
            .agg(['min', 'max', 'mean'])
            .reset_index()
            .sort_values('Q2')
        )
        # 阴影带（包络）
        ax.fill_between(band['Q2'], band['min'], band['max'], color=color, alpha=0.2)
        # 平均折线
        ax.plot(band['Q2'], band['mean'], color=color, linewidth=4.0, label=model_label)

    ax.set_xticks(x_union)
    ax.set_xticklabels(x_union, fontsize=18, rotation=45)
    ax.set_xlabel('Number of Negative Data', fontsize=25)
    ax.set_ylabel('Average Positive Score', fontsize=25)
    ax.set_ylim([-2, -0.5])
    ax.tick_params(axis='y', labelsize=18)
    plt.legend(title='Model', fontsize=16, title_fontsize=18, loc='upper center', bbox_to_anchor=(0.5, 1.25), ncol=3)
    plt.tight_layout()
    os.makedirs(base_output_dir, exist_ok=True)
    plt.savefig(os.path.join(base_output_dir, output_name), bbox_inches='tight')
    plt.close()


vis_combined(output_name=output_name)
