#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
绘制 Resistance 训练过程中的 Loss 曲线
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import numpy as np
from pathlib import Path


def smooth_curve(data, window_size=5):
    """
    使用移动平均法平滑曲线
    
    Args:
        data: 原始数据
        window_size: 窗口大小，必须为奇数
    
    Returns:
        平滑后的数据
    """
    if window_size <= 1:
        return data
    
    # 确保窗口大小为奇数
    if window_size % 2 == 0:
        window_size += 1
    
    # 使用卷积进行移动平均
    kernel = np.ones(window_size) / window_size
    smoothed = np.convolve(data, kernel, mode='same')
    
    # 处理边界效应
    half_window = window_size // 2
    for i in range(half_window):
        smoothed[i] = np.mean(data[:i + half_window + 1])
        smoothed[-(i+1)] = np.mean(data[-(i + half_window + 1):])
    
    return smoothed


def plot_resistance_loss(model_name, input_root_dir, output_root_dir, max_index=6, smooth_window=0):
    """
    绘制 resistance 训练的 loss 曲线
    
    Args:
        model_name: 模型名称
        input_root_dir: 输入根目录
        output_root_dir: 输出根目录
        max_index: 最大索引值（默认为6）
        smooth_window: 平滑窗口大小（0表示不平滑，推荐值5-15）
    """
    # 构建输入和输出目录
    input_dir = os.path.join(input_root_dir, model_name)
    output_dir = os.path.join(output_root_dir, model_name)
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")
    print(f"开始处理模型: {model_name}")
    print("-" * 80)
    
    # 遍历所有的 i, j 组合
    for i in range(max_index + 1):
        for j in range(max_index + 1):
            # 跳过 i == j 的情况
            if i == j:
                continue
            
            # 只处理 i < j 的情况，避免重复
            if i >= j:
                continue
            
            # 构建目录路径
            dir_forward = os.path.join(input_dir, f"main_{i}_resist_{j}")
            dir_inverse = os.path.join(input_dir, f"main_{j}_resist_{i}")
            
            # 构建 CSV 文件路径
            csv_forward = os.path.join(dir_forward, "metrics.csv")
            csv_inverse = os.path.join(dir_inverse, "metrics.csv")
            
            # 检查文件是否存在
            if not os.path.exists(csv_forward):
                print(f"警告: 文件不存在 - {csv_forward}")
                continue
            if not os.path.exists(csv_inverse):
                print(f"警告: 文件不存在 - {csv_inverse}")
                continue
            
            try:
                # 读取 CSV 文件
                df_forward = pd.read_csv(csv_forward)
                df_inverse = pd.read_csv(csv_inverse)
                
                # 检查必需的列是否存在
                if 'step' not in df_forward.columns or 'train/loss' not in df_forward.columns:
                    print(f"警告: {csv_forward} 缺少必需的列")
                    continue
                if 'step' not in df_inverse.columns or 'train/loss' not in df_inverse.columns:
                    print(f"警告: {csv_inverse} 缺少必需的列")
                    continue
                
                # 获取原始数据
                steps_forward = df_forward['step'].values
                loss_forward = df_forward['train/loss'].values
                steps_inverse = df_inverse['step'].values
                loss_inverse = df_inverse['train/loss'].values
                
                # 应用平滑
                if smooth_window > 0:
                    loss_forward_smoothed = smooth_curve(loss_forward, smooth_window)
                    loss_inverse_smoothed = smooth_curve(loss_inverse, smooth_window)
                else:
                    loss_forward_smoothed = loss_forward
                    loss_inverse_smoothed = loss_inverse
                
                # 创建图形
                plt.figure(figsize=(10, 6))
                
                # 如果使用平滑，绘制原始曲线（半透明）
                if smooth_window > 0:
                    plt.plot(
                        steps_forward,
                        loss_forward,
                        color='red',
                        linewidth=0.8,
                        alpha=0.2,
                        linestyle='-'
                    )
                    plt.plot(
                        steps_inverse,
                        loss_inverse,
                        color='blue',
                        linewidth=0.8,
                        alpha=0.2,
                        linestyle='-'
                    )
                
                # 绘制 forward alignment (i < j) - 红色
                plt.plot(
                    steps_forward,
                    loss_forward_smoothed,
                    color='red',
                    linewidth=2,
                    label=f'Forward Alignment (main_{i}_resist_{j})',
                    alpha=0.8
                )
                
                # 绘制 inverse alignment (j > i) - 蓝色
                plt.plot(
                    steps_inverse,
                    loss_inverse_smoothed,
                    color='blue',
                    linewidth=2,
                    label=f'Inverse Alignment (main_{j}_resist_{i})',
                    alpha=0.8
                )
                
                # 设置图形属性
                plt.xlabel('Step', fontsize=12)
                plt.ylabel('Train Loss', fontsize=12)
                plt.title(f'Training Loss Comparison: main_{i} vs main_{j}', fontsize=14, fontweight='bold')
                plt.legend(loc='best', fontsize=10)
                plt.grid(True, alpha=0.3, linestyle='--')
                
                # 保存图形
                output_file = os.path.join(output_dir, f"loss_{i}_{j}.pdf")
                plt.savefig(output_file, format='pdf', bbox_inches='tight', dpi=300)
                plt.close()
                
                print(f"✓ 已生成: loss_{i}_{j}.pdf")
                
            except Exception as e:
                print(f"错误: 处理 loss_{i}_{j}.pdf 时出错 - {e}")
                plt.close()
                continue
    
    print("-" * 80)
    print(f"完成! 所有图表已保存到: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='绘制 Resistance 训练 Loss 曲线')
    parser.add_argument(
        '--model_name',
        type=str,
        default='alpaca_qwen1.5-0.5B',
        help='模型名称'
    )
    parser.add_argument(
        '--input_root_dir',
        type=str,
        default='/mnt/shared-storage-user/zhoujiayi/boyuan/model_results/resistance/resistance_process/',
        help='输入根目录'
    )
    parser.add_argument(
        '--output_root_dir',
        type=str,
        default='/mnt/shared-storage-user/zhoujiayi/boyuan/visualization_results/resistance',
        help='输出根目录'
    )
    parser.add_argument(
        '--max_index',
        type=int,
        default=6,
        help='最大索引值（默认为6，即处理0-6）'
    )
    parser.add_argument(
        '--smooth_window',
        type=int,
        default=10,
        help='平滑窗口大小（0表示不平滑，推荐值5-15，默认为10）'
    )
    
    args = parser.parse_args()
    
    # 执行绘图
    plot_resistance_loss(
        model_name=args.model_name,
        input_root_dir=args.input_root_dir,
        output_root_dir=args.output_root_dir,
        max_index=args.max_index,
        smooth_window=args.smooth_window
    )


if __name__ == "__main__":
    main()