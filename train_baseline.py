#!/usr/bin/env python3
"""
YOLO11m 基线训练脚本
================================================================================
与 train_innovative_v5_safe.py 配套的基线版本，用于对比实验。

特点:
1. 使用标准YOLO损失函数（无创新损失）
2. 使用默认损失权重（box=7.5, cls=0.5, dfl=1.5）
3. 保留相同的数据增强策略
4. 保留相同的安全防护机制

用途:
- 作为创新方案的对照组
- 验证创新损失函数的实际效果
- 建立mAP基准线
================================================================================
"""

import argparse
import sys
import os
from pathlib import Path
import torch
import logging
from ultralytics import YOLO
import gc

# 系统防护：设置环境变量，防止临时文件写入根目录
os.environ['TMPDIR'] = './tmp'
os.makedirs('./tmp', exist_ok=True)


def train_baseline(args):
    """
    基线训练 - 标准YOLO配置
    """
    
    # 检查数据集
    data_yaml = args.data
    if not Path(data_yaml).exists():
        raise FileNotFoundError(f"❌ data.yaml不存在: {data_yaml}")
    
    # 强制指定输出目录
    output_project = Path(args.project).expanduser().resolve()
    output_project.mkdir(parents=True, exist_ok=True)
    
    # 检查磁盘空间
    import shutil
    try:
        _, _, free_space = shutil.disk_usage(output_project)
        free_gb = free_space / (1024**3)
        if free_gb < 5:
            print(f"⚠️  警告: 可用空间不足5GB ({free_gb:.2f}GB)")
        else:
            print(f"✅ 磁盘空间充足: {free_gb:.2f}GB")
    except Exception as e:
        print(f"⚠️  无法检查磁盘空间: {e}")
    
    print("\n" + "=" * 80)
    print("🔵 YOLO11m 基线训练 (对照组)")
    print("=" * 80)
    print(f"数据集: {data_yaml}")
    print(f"设备: {args.device}")
    print(f"输入尺寸: {args.imgsz}")
    print(f"批次大小: {args.batch}")
    print(f"训练轮数: {args.epochs}")
    print(f"输出路径: {output_project}")
    print("=" * 80)
    
    # 加载模型
    print("\n📦 加载模型...")
    
    if args.pretrained and Path(args.pretrained).exists():
        print(f"📥 加载预训练权重: {args.pretrained}")
        model = YOLO(args.pretrained)
    else:
        model = YOLO('yolo11m-seg.pt')
    
    # ========================================================================
    # 基线配置 - 标准YOLO参数
    # ========================================================================
    train_config = {
        # 基础配置
        "data": args.data,
        "epochs": args.epochs,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "device": args.device,
        "workers": args.workers,
        "project": str(output_project),
        "name": args.name,
        "exist_ok": True,
        "pretrained": False if args.pretrained else True,
        
        # 优化器配置 - 标准SGD
        "optimizer": args.optimizer,
        "lr0": args.lr0,
        "lrf": args.lrf,
        "momentum": 0.937,
        "weight_decay": args.weight_decay,
        "warmup_epochs": args.warmup_epochs,
        "warmup_momentum": 0.8,
        "warmup_bias_lr": 0.1,
        
        # 🔵 基线损失权重 - 使用YOLO默认值
        "box": args.box_loss,  # 默认 7.5
        "cls": args.cls_loss,  # 默认 0.5
        "dfl": args.dfl_loss,  # 默认 1.5
        
        # 标签平滑 - 默认关闭
        "label_smoothing": args.label_smoothing,
        
        # 掩码计算
        "overlap_mask": False,
        "mask_ratio": 1,
        
        # 数据增强 - 与V5保持一致便于对比
        "hsv_h": 0.015,
        "hsv_s": args.hsv_s,
        "hsv_v": 0.4,
        "degrees": 10.0,
        "translate": 0.1,
        "scale": 0.5,
        "shear": 0.0,
        "perspective": 0.0,
        "flipud": 0.0,
        "fliplr": 0.5,
        
        # 核心增强 - 与V5保持一致便于对比
        "mosaic": args.mosaic,
        "close_mosaic": args.close_mosaic,
        "copy_paste": args.copy_paste,
        "mixup": args.mixup,
        
        # 训练策略
        "patience": args.patience,
        "freeze": args.freeze,
        
        # 安全配置
        "save": True,
        "save_period": -1,
        "val": True,
        "plots": False,
        "amp": True,
        
        # 验证配置
        "conf": 0.001,
        "iou": 0.5,
        "max_det": 300,
        
        # 数据使用
        "fraction": 1.0,
        
        # 详细输出
        "verbose": True,
    }
    
    print("\n📋 基线配置摘要:")
    print(f"  ├─ 模型: YOLO11m-seg")
    print(f"  ├─ 输入尺寸: {args.imgsz}")
    print(f"  ├─ 优化器: {args.optimizer}")
    print(f"  ├─ 学习率: {args.lr0}")
    print(f"  ├─ 损失权重: box={args.box_loss}, cls={args.cls_loss}, dfl={args.dfl_loss} (标准值)")
    print(f"  ├─ 标签平滑: {args.label_smoothing}")
    print(f"  ├─ CopyPaste: {args.copy_paste}")
    print(f"  └─ 输出路径: {output_project}")
    
    # 开始训练
    print("\n" + "=" * 80)
    print("🔵 开始基线训练...")
    print("=" * 80)
    
    try:
        def cleanup_memory():
            torch.cuda.empty_cache()
            gc.collect()
        
        cleanup_memory()
        
        results = model.train(**train_config)
        
        cleanup_memory()
        
        print("\n" + "=" * 80)
        print("✅ 基线训练完成！")
        print("=" * 80)
        print(f"模型保存在: {output_project}/{args.name}/weights/best.pt")
        print("\n📊 此为基线结果，用于与创新版本对比")
        
        return results
        
    except KeyboardInterrupt:
        print("\n⚠️  训练被用户中断")
        cleanup_memory()
        return None
    except Exception as e:
        print(f"\n❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()
        cleanup_memory()
        return None


def main():
    parser = argparse.ArgumentParser(
        description="YOLO11m 基线训练 (对照组)"
    )
    
    # 基础参数
    parser.add_argument(
        '--data',
        type=str,
        default='container_damage.yaml',
        help='数据集YAML路径（默认: container_damage.yaml）'
    )
    parser.add_argument('--device', type=str, default='0',
                        help='设备ID')
    parser.add_argument('--epochs', type=int, default=100,
                        help='训练轮数')
    parser.add_argument('--batch', type=int, default=16,
                        help='批次大小')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='输入图像尺寸')
    
    # 预训练权重
    parser.add_argument('--pretrained', type=str, default=None,
                        help='预训练权重路径 (可选)')
    
    # 优化器参数 - 使用SGD作为基线
    parser.add_argument('--optimizer', type=str, default='SGD',
                        choices=['SGD', 'AdamW', 'Adam'],
                        help='优化器 (基线默认SGD)')
    parser.add_argument('--lr0', type=float, default=0.01,
                        help='初始学习率 (SGD默认0.01)')
    parser.add_argument('--lrf', type=float, default=0.01,
                        help='最终学习率')
    parser.add_argument('--weight-decay', type=float, default=0.0005,
                        help='权重衰减 (SGD默认0.0005)')
    parser.add_argument('--warmup-epochs', type=int, default=3,
                        help='预热轮数')
    
    # 损失函数权重 - 使用YOLO默认值
    parser.add_argument('--box-loss', type=float, default=7.5,
                        help='边界框损失权重 (YOLO默认)')
    parser.add_argument('--cls-loss', type=float, default=0.5,
                        help='分类损失权重 (YOLO默认)')
    parser.add_argument('--dfl-loss', type=float, default=1.5,
                        help='DFL损失权重 (YOLO默认)')
    
    # 标签平滑 - 基线关闭
    parser.add_argument('--label-smoothing', type=float, default=0.0,
                        help='标签平滑系数 (基线默认0)')
    
    # 数据增强 - 与V5保持一致便于对比
    parser.add_argument('--hsv-s', type=float, default=0.9,
                        help='HSV饱和度增强')
    parser.add_argument('--mosaic', type=float, default=1.0,
                        help='Mosaic增强概率')
    parser.add_argument('--close-mosaic', type=int, default=15,
                        help='提前关闭Mosaic的epoch')
    parser.add_argument('--copy-paste', type=float, default=0.8,
                        help='CopyPaste增强概率 (与V5一致)')
    parser.add_argument('--mixup', type=float, default=0.2,
                        help='Mixup增强概率')
    
    # 训练策略
    parser.add_argument('--patience', type=int, default=0,
                        help='早停耐心值')
    parser.add_argument('--freeze', type=int, default=0,
                        help='冻结Backbone层数')
    
    # 安全配置
    parser.add_argument('--workers', type=int, default=0,
                        help='数据加载进程数 (0=主进程加载，更稳定)')
    parser.add_argument('--project', type=str, default='./runs',
                        help='项目路径')
    
    # 输出路径
    parser.add_argument('--name', type=str, default='y11m_baseline',
                        help='实验名称')
    
    args = parser.parse_args()
    
    # 开始训练
    train_baseline(args)


if __name__ == '__main__':
    main()
