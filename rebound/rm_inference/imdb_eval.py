from transformers import pipeline
import json
import os

input_dir = '/mnt/shared-storage-user/zhoujiayi/boyuan/data_results/rebound/imdb'
output_dir = '/mnt/shared-storage-user/zhoujiayi/boyuan/rm_score_rebound/imdb'

# 文件夹路径
for model_name in os.listdir(input_dir):
    input_folder = os.path.join(input_dir, model_name)
    output_folder = os.path.join(output_dir, model_name)

    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)

    # 初始化情感分析模型
    sentiment_analysis = pipeline("sentiment-analysis", model="/mnt/shared-storage-user/zhoujiayi/models/siebert/sentiment-roberta-large-english", truncation=True, device=4)

    # 遍历文件夹中的所有 JSON 文件
    for filename in os.listdir(input_folder):
        if filename.endswith('.json'):
            input_filename = os.path.join(input_folder, filename)
            output_filename = os.path.join(output_folder, filename)

            # 读取 JSON 文件
            with open(input_filename, 'r') as f:
                data = json.load(f)

            # 提取文本
            input_texts = [item['prompt']+item['response'] for item in data]

            # 执行情感分析
            scores = sentiment_analysis(input_texts)
            new_scores = []
            new_data = []
            for s in scores:
                if s['label'] == 'POSITIVE':
                    new_scores.append(s['score'])
                else:
                    new_scores.append(1 - s['score'])

            # 更新数据并添加分数
            for i in range(len(data)):
                item = data[i].copy()
                item['score'] = new_scores[i]
                new_data.append(item)

            # 将结果写入新的 JSON 文件
            with open(output_filename, 'w', encoding='utf-8') as f:
                json.dump(new_data, f, ensure_ascii=False, indent=4)

    print("当前文件处理完毕。")
print("所有文件处理完毕。")