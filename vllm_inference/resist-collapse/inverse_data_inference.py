import json
from vllm import LLM, SamplingParams
import os
import argparse
from tqdm import tqdm
from datetime import datetime


def parse_args():
    parser = argparse.ArgumentParser(description='使用 VLLM 进行模型推理')
    parser.add_argument('--input_file', type=str, required=True,
                        help='输入的 JSON 文件路径')
    parser.add_argument('--model_path', type=str, required=True,
                        help='模型路径')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='输出结果的根目录')
    parser.add_argument('--model_name', type=str, required=True,
                        help='模型名称')
    parser.add_argument('--tensor_parallel_size', type=int, default=8,
                        help='张量并行大小（默认：8）')
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.95,
                        help='GPU 内存利用率（默认：0.95）')
    parser.add_argument('--swap_space', type=int, default=32,
                        help='交换空间大小（默认：32）')
    parser.add_argument('--temperature', type=float, default=0.1,
                        help='采样温度（默认：0.1）')
    parser.add_argument('--top_p', type=float, default=0.95,
                        help='Top-p 采样参数（默认：0.95）')
    parser.add_argument('--max_tokens', type=int, default=32768)
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("=" * 80)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    # 打印配置信息
    print("\n配置参数:")
    print(f"  输入文件: {args.input_file}")
    print(f"  模型路径: {args.model_path}")
    print(f"  输出目录: {args.output_dir}")
    print(f"  张量并行大小: {args.tensor_parallel_size}")
    print(f"  GPU 内存利用率: {args.gpu_memory_utilization}")
    print(f"  采样温度: {args.temperature}")
    print(f"  Top-p: {args.top_p}")
    if args.max_tokens:
        print(f"  最大 token 数: {args.max_tokens}")
    print()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"✓ 输出目录已创建: {args.output_dir}\n")
    
    
    # 读取输入数据
    print(f"正在读取输入文件: {args.input_file}")
    file_ext = os.path.splitext(args.input_file)[1].lower()
    data = []
    with open(args.input_file, 'r', encoding='utf-8') as f:
        if file_ext == ".jsonl":
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
        elif file_ext == ".json":
            data = json.load(f)
        else:
            raise ValueError("仅支持 .json 或 .jsonl 格式的输入文件")
    print(f"✓ 成功读取 {len(data)} 条数据\n")
    
    # 初始化模型
    print("正在加载模型...")
    llm = LLM(
        model=args.model_path,
        gpu_memory_utilization=args.gpu_memory_utilization,
        swap_space=args.swap_space,
        trust_remote_code=True,
        tensor_parallel_size=args.tensor_parallel_size,
        generation_config="vllm",
    )
    print("✓ 模型加载完成\n")
    
    # 获取 tokenizer（用于应用 chat template 到消息上）
    tokenizer = llm.get_tokenizer()
    has_chat_template = getattr(tokenizer, "chat_template", None) not in (None, "")
    
    print("正在准备 prompts...")
    prompts = []
    # Rebound 的 PKU-SafeRLHF \ IMDB 的模板目前是一样的 
    for item in tqdm(data, desc="提取 prompts"):
        # 构造消息
        messages = [
            {"role": "system", "content": ""},
            {"role": "user", "content": item['prompt']}
        ]
        # # 将消息通过 chat template 转为字符串；如无模板则采用简易回退模板
        # if has_chat_template:
        #     prompt_text = tokenizer.apply_chat_template(
        #         messages,
        #         add_generation_prompt=True,
        #         tokenize=False,
        #     )
        # else:
        #     prompt_text = messages
        prompts.append(messages)
    print(f"✓ 准备了 {len(prompts)} 个 prompts\n")
    
    # 设置采样参数
    sampling_params_dict = {
        'temperature': args.temperature,
        'top_p': args.top_p,
        'repetition_penalty': 1.2,
    }
    if args.max_tokens:
        sampling_params_dict['max_tokens'] = args.max_tokens
    
    sampling_params = SamplingParams(**sampling_params_dict)
    print(f"采样参数: {sampling_params_dict}\n")
    
    with open('/mnt/shared-storage-user/zhoujiayi/boyuan/model_results/chat_template/rebound_template.jinja', "r") as f:
        chat_template = f.read()

    # 生成输出
    print("开始生成...")
    outputs = llm.chat(
        prompts, 
        sampling_params,
        chat_template = chat_template
        )
    print("✓ 生成完成\n")
    
    # 处理输出结果
    print("正在处理输出结果...")
    new_data = []
    for output, item in tqdm(zip(outputs, data), total=len(data), desc="处理结果"):
        prompt = item['prompt']
        generated_text = output.outputs[0].text
        
        item['response'] = generated_text
        new_data.append(
            {
                'prompt': prompt,
                'response': generated_text,
                'finish_reason': output.outputs[0].finish_reason,
                'model_name': args.model_name,   
                'model_path': args.model_path,
                'input_file': args.input_file,
                'output_dir': args.output_dir,
            }
        )
    print(f"✓ 处理了 {len(new_data)} 条结果\n")
    

    output_filename = f"{args.model_name}.json"
    
    output_path = os.path.join(args.output_dir, output_filename)
    print(f"正在保存结果到: {output_path}")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(new_data, f, ensure_ascii=False, indent=4)
    
    print(f"✓ 结果已保存到: {output_path}")
    
    # 打印统计信息
    print("\n" + "=" * 80)
    print("统计信息:")
    print(f"  处理数据条数: {len(new_data)}")
    print(f"  输出文件大小: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")
    print(f"  完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    print("\n✓ 任务完成！")


if __name__ == "__main__":
    main()