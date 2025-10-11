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
    parser.add_argument('--output_root_dir', type=str, required=True,
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
    parser.add_argument('--max_tokens', type=int, default=None,
                        help='最大生成 token 数（可选）')
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
    print(f"  输出目录: {args.output_root_dir}")
    print(f"  张量并行大小: {args.tensor_parallel_size}")
    print(f"  GPU 内存利用率: {args.gpu_memory_utilization}")
    print(f"  采样温度: {args.temperature}")
    print(f"  Top-p: {args.top_p}")
    if args.max_tokens:
        print(f"  最大 token 数: {args.max_tokens}")
    print()
    
    # 创建输出目录
    os.makedirs(args.output_root_dir, exist_ok=True)
    print(f"✓ 输出目录已创建: {args.output_root_dir}\n")
    
    
    # 读取输入数据
    print(f"正在读取输入文件: {args.input_file}")
    with open(args.input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"✓ 成功读取 {len(data)} 条数据\n")
    
    # 初始化模型
    print("正在加载模型...")
    llm = LLM(
        model=args.model_path,
        gpu_memory_utilization=args.gpu_memory_utilization,
        swap_space=args.swap_space,
        trust_remote_code=True,
        tensor_parallel_size=args.tensor_parallel_size,
    )
    print("✓ 模型加载完成\n")
    
    # 准备prompts
    print("正在准备 prompts...")
    prompts = []
    for item in tqdm(data, desc="提取 prompts"):
        prompts.append(item['prompt'])
    print(f"✓ 准备了 {len(prompts)} 个 prompts\n")
    
    # 设置采样参数
    sampling_params_dict = {
        'temperature': args.temperature,
        'top_p': args.top_p
    }
    if args.max_tokens:
        sampling_params_dict['max_tokens'] = args.max_tokens
    
    sampling_params = SamplingParams(**sampling_params_dict)
    print(f"采样参数: {sampling_params_dict}\n")
    
    # 生成输出
    print("开始生成...")
    outputs = llm.generate(prompts, sampling_params)
    print("✓ 生成完成\n")
    
    # 处理输出结果
    print("正在处理输出结果...")
    new_data = []
    for output, item in tqdm(zip(outputs, data), total=len(data), desc="处理结果"):
        prompt = output.prompt
        generated_text = output.outputs[0].text
        
        item['response'] = generated_text
        new_data.append(
            {
                'prompt': prompt,
                'response': generated_text,
                'model_name': args.model_name,   
                'model_path': args.model_path,
                'input_file': args.input_file,
                'output_root_dir': args.output_root_dir,
            }
        )
    print(f"✓ 处理了 {len(new_data)} 条结果\n")
    

    output_filename = f"{args.model_name}.json"
    
    output_path = os.path.join(args.output_root_dir, output_filename)
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