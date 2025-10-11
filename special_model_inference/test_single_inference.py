#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pythia 模型推理测试脚本
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse


def main():
    # 参数配置
    parser = argparse.ArgumentParser(description='Pythia 模型推理测试')
    parser.add_argument(
        '--model_path',
        type=str,
        default='/mnt/shared-storage-user/zhoujiayi/models/pythia/pythia-2.8b',
        help='模型路径'
    )
    parser.add_argument(
        '--prompt',
        type=str,
        default='Hello, I am',
        help='输入提示词'
    )
    parser.add_argument(
        '--max_length',
        type=int,
        default=50,
        help='生成的最大长度'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.7,
        help='生成温度'
    )
    parser.add_argument(
        '--top_p',
        type=float,
        default=0.9,
        help='nucleus sampling 参数'
    )
    parser.add_argument(
        '--top_k',
        type=int,
        default=50,
        help='top-k sampling 参数'
    )
    parser.add_argument(
        '--num_return_sequences',
        type=int,
        default=1,
        help='返回序列数量'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['auto', 'cuda', 'cpu'],
        help='运行设备'
    )
    
    args = parser.parse_args()
    
    # 设备配置
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print(f"使用设备: {device}")
    print(f"加载模型: {args.model_path}")
    
    # 加载模型和分词器
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch.float16 if device == 'cuda' else torch.float32,
            device_map='auto' if device == 'cuda' else None,
            trust_remote_code=True,
        )
        
        if device == 'cpu':
            model = model.to(device)
        
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        
        # 设置 pad_token 如果不存在
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        print("模型加载成功！")
        
    except Exception as e:
        print(f"加载模型失败: {e}")
        return
    
    # 推理
    print(f"\n输入提示词: {args.prompt}")
    print("-" * 80)
    
    try:
        # 准备输入
        inputs = tokenizer(args.prompt, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # 生成文本
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=args.max_length,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                num_return_sequences=args.num_return_sequences,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        # 解码并打印结果
        print("生成结果:\n")
        for idx, output in enumerate(outputs):
            generated_text = tokenizer.decode(output, skip_special_tokens=True)
            print(f"序列 {idx + 1}:")
            print(generated_text)
            print("-" * 80)
        
        # 打印模型信息
        print("\n模型信息:")
        print(f"参数量: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")
        print(f"词表大小: {len(tokenizer)}")
        
    except Exception as e:
        print(f"推理失败: {e}")
        return


if __name__ == "__main__":
    main()