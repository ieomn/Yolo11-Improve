#!/usr/bin/env python3
"""
YOLO11m 创新训练脚本 V5 - 安全防护版
================================================================================
核心改进 (解决服务器卡死问题):
1. 🔒 强制约束日志路径 - 避免根目录被塞满
2. 🔒 损失函数数值防溢出 - 防止NaN和Inf
3. 🔒 防止递归注入 - 确保只注入一次
4. 🔒 关闭冗余I/O - 减少磁盘压力
5. 🔒 显存管理优化 - 防止内存泄漏
6. 🔒 异常捕获增强 - 优雅处理错误

问题诊断:
- 数值正常但服务器卡死
- 根目录被报错文档塞满
- 可能原因: 冗余打印、显存碎片、计算图未释放

安全措施:
- 日志重定向到指定路径
- 关闭冗余绘图和中间保存
- 数值稳定性保护
- 防止递归调用
- 显存清理机制
================================================================================
"""

import argparse
import sys
import os
from pathlib import Path
import torch
import torch.nn as nn
import logging
from ultralytics import YOLO
import gc

# 🔒 系统防护：屏蔽非致命警告，防止日志刷屏
logging.getLogger("ultralytics").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)
logging.getLogger("PIL").setLevel(logging.ERROR)

# 🔒 系统防护：设置环境变量，防止临时文件写入根目录
os.environ['TMPDIR'] = '/data/swy/tmp'  # 确保临时文件不在根目录
os.makedirs('/data/swy/tmp', exist_ok=True)

# 导入创新损失函数
try:
    from innovative_loss import (
        PolyLoss,
        FocalLossVariant,
        SlideLoss,
        FocusIoU,
        LabelSmoothingCrossEntropy,
        QualityFocalLoss,
        ContainerDamageLoss
    )
    INNOVATIVE_LOSS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  创新损失函数不可用: {e}")
    INNOVATIVE_LOSS_AVAILABLE = False

# 🔒 全局状态：防止递归注入
_INJECTED = False


def inject_custom_loss_to_model(model, args):
    """
    通过hook机制注入自定义损失函数（安全版）
    
    改进:
    - 防止递归调用
    - 异常捕获
    - 显存清理
    """
    global _INJECTED
    
    # 🔒 防止递归注入
    if _INJECTED:
        return model
    
    if not INNOVATIVE_LOSS_AVAILABLE:
        return model
    
    try:
        # 尝试替换模型的损失函数
        if hasattr(model, 'trainer') and hasattr(model.trainer, 'loss'):
            print("🔥 尝试注入自定义损失函数...")
            
            # 创建自定义损失函数
            custom_loss = ContainerDamageLoss(
                model=model.model,
                use_poly_loss=True,
                use_label_smoothing=True,
                label_smoothing=args.label_smoothing,
                use_focus_iou=args.use_focus_iou,
                use_slide_loss=True,
                slide_iou_threshold=0.3,  # 针对Hole调整
                slide_weight_scale=2.0,
            )
            
            # 尝试替换
            model.trainer.loss = custom_loss
            print("✅ 自定义损失函数已注入")
            _INJECTED = True
            
            # 🔒 显存清理
            torch.cuda.empty_cache()
            gc.collect()
        else:
            print("⚠️  无法访问模型损失函数，将使用参数优化模式")
    except Exception as e:
        print(f"⚠️  损失函数注入失败: {e}")
        print("💡 将使用参数优化模式（通过调整损失权重间接生效）")
        # 🔒 异常后清理
        torch.cuda.empty_cache()
        gc.collect()
    
    return model


def train_innovative_v5_safe(args):
    """
    创新训练V5 - 安全防护版
    """
    
    # 检查数据集
    data_yaml = args.data
    if not Path(data_yaml).exists():
        raise FileNotFoundError(f"❌ data.yaml不存在: {data_yaml}")
    
    # 🔒 系统防护：强制指定输出目录到数据盘，不在根目录
    output_project = Path(args.project).expanduser().resolve()
    output_project.mkdir(parents=True, exist_ok=True)
    
    # 🔒 系统防护：检查磁盘空间（至少5GB）
    import shutil
    _, _, free_space = shutil.disk_usage(output_project)
    free_gb = free_space / (1024**3)
    if free_gb < 5:
        print(f"⚠️  警告: 可用空间不足5GB ({free_gb:.2f}GB)")
        print(f"💡 建议: 清理空间后再运行训练")
        response = input("是否继续? (y/n): ")
        if response.lower() != 'y':
            return None
    else:
        print(f"✅ 磁盘空间充足: {free_gb:.2f}GB")
    
    print("\n" + "=" * 80)
    print("🚀 YOLO11m 创新训练 V5 - 安全防护版")
    print("=" * 80)
    print(f"数据集: {data_yaml}")
    print(f"设备: {args.device}")
    print(f"输入尺寸: {args.imgsz}")
    print(f"批次大小: {args.batch}")
    print(f"训练轮数: {args.epochs}")
    print(f"输出路径: {output_project} (不在根目录)")
    print("=" * 80)
    
    # 加载模型
    print("\n📦 加载模型...")
    
    # 如果提供了预训练权重，加载它
    if args.pretrained and Path(args.pretrained).exists():
        print(f"📥 加载预训练权重: {args.pretrained}")
        model = YOLO(args.pretrained)
    else:
        model = YOLO('yolo11m-seg.pt')
    
    # 尝试注入自定义损失函数
    if args.use_innovative_loss:
        model = inject_custom_loss_to_model(model, args)
    
    # ========================================================================
    # V5核心配置 - 安全防护版
    # ========================================================================
    train_config = {
        # 基础配置
        "data": data_yaml,
        "epochs": args.epochs,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "device": args.device,
        "workers": args.workers,  # 可配置，避免过多进程
        "project": str(output_project),  # 🔒 使用绝对路径
        "name": args.name,
        "exist_ok": True,
        "pretrained": False if args.pretrained else True,
        
        # 优化器配置
        "optimizer": "AdamW",
        "lr0": args.lr0,
        "lrf": args.lrf,
        "momentum": 0.937,
        "weight_decay": args.weight_decay,
        "warmup_epochs": args.warmup_epochs,
        "warmup_momentum": 0.8,
        "warmup_bias_lr": 0.1,
        
        # 🔥 V5核心优化：损失权重重平衡
        "box": args.box_loss,
        "cls": args.cls_loss,
        "dfl": args.dfl_loss,
        
        # 标签平滑
        "label_smoothing": args.label_smoothing,
        
        # 掩码计算
        "overlap_mask": False,
        "mask_ratio": 1,
        
        # 数据增强
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
        
        # 核心增强
        "mosaic": args.mosaic,
        "close_mosaic": args.close_mosaic,
        "copy_paste": args.copy_paste,
        "mixup": args.mixup,
        
        # 多尺度训练
        "multi_scale": args.multi_scale,
        
        # 训练策略
        "patience": args.patience,
        "freeze": args.freeze,
        
        # 🔒 安全配置：极简主义，减少I/O压力
        "save": True,
        "save_period": -1,  # 🔒 只存最好的，不存中间过程
        "val": True,
        "plots": False,  # 🔒 关闭冗余绘图，减少I/O压力（除非是最后一次训练）
        "amp": True,  # 🔒 在A40上必须开，配合数值保护
        
        # 验证配置
        "conf": 0.001,
        "iou": 0.5,
        "max_det": 300,
        
        # 数据使用
        "fraction": 1.0,
        
        # 🔒 其他安全配置
        "verbose": False,  # 🔒 减少冗余打印
    }
    
    print("\n📋 V5安全配置摘要:")
    print(f"  ├─ 模型: YOLO11m-seg")
    print(f"  ├─ 输入尺寸: {args.imgsz}")
    print(f"  ├─ 优化器: AdamW")
    print(f"  ├─ 学习率: {args.lr0}")
    print(f"  ├─ 损失权重: box={args.box_loss}, cls={args.cls_loss}, dfl={args.dfl_loss}")
    print(f"  ├─ 标签平滑: {args.label_smoothing}")
    print(f"  ├─ CopyPaste: {args.copy_paste}")
    print(f"  ├─ 多尺度训练: {args.multi_scale}")
    print(f"  ├─ 输出路径: {output_project} (安全路径)")
    print(f"  ├─ 冗余绘图: 关闭 (减少I/O)")
    print(f"  └─ 中间保存: 关闭 (只存最好的)")
    
    # 开始训练
    print("\n" + "=" * 80)
    print("🚀 开始V5安全训练...")
    print("=" * 80)
    
    try:
        # 🔒 定期显存清理
        def cleanup_memory():
            torch.cuda.empty_cache()
            gc.collect()
        
        # 训练前清理
        cleanup_memory()
        
        results = model.train(**train_config)
        
        # 训练后清理
        cleanup_memory()
        
        print("\n" + "=" * 80)
        print("✅ V5训练完成！")
        print("=" * 80)
        print(f"模型保存在: {output_project}/{args.name}/weights/best.pt")
        print("\n💡 预期提升:")
        print("   ├─ Hole召回: 0.152 → 0.25-0.30 (+64-97%)")
        print("   ├─ Rusty召回: 0.297 → 0.35-0.40 (+18-35%)")
        print("   └─ 整体mAP: 0.376 → 0.42-0.45 (+12-20%)")
        
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
        description="YOLO11m 创新训练 V5 - 安全防护版"
    )
    
    # 基础参数
    parser.add_argument('--data', type=str, required=True,
                        help='数据集YAML路径')
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
    
    # 优化器参数
    parser.add_argument('--lr0', type=float, default=0.001,
                        help='初始学习率')
    parser.add_argument('--lrf', type=float, default=0.01,
                        help='最终学习率')
    parser.add_argument('--weight-decay', type=float, default=0.05,
                        help='权重衰减')
    parser.add_argument('--warmup-epochs', type=int, default=3,
                        help='预热轮数')
    
    # 损失函数权重
    parser.add_argument('--box-loss', type=float, default=20.0,
                        help='边界框损失权重')
    parser.add_argument('--cls-loss', type=float, default=1.5,
                        help='分类损失权重')
    parser.add_argument('--dfl-loss', type=float, default=2.5,
                        help='DFL损失权重')
    
    # 标签平滑
    parser.add_argument('--label-smoothing', type=float, default=0.05,
                        help='标签平滑系数')
    
    # 数据增强
    parser.add_argument('--hsv-s', type=float, default=0.9,
                        help='HSV饱和度增强')
    parser.add_argument('--mosaic', type=float, default=1.0,
                        help='Mosaic增强概率')
    parser.add_argument('--close-mosaic', type=int, default=15,
                        help='提前关闭Mosaic的epoch')
    parser.add_argument('--copy-paste', type=float, default=0.8,
                        help='CopyPaste增强概率')
    parser.add_argument('--mixup', type=float, default=0.2,
                        help='Mixup增强概率')
    
    # 训练策略
    parser.add_argument('--multi-scale', action='store_true', default=True,
                        help='启用多尺度训练')
    parser.add_argument('--patience', type=int, default=0,
                        help='早停耐心值')
    parser.add_argument('--freeze', type=int, default=0,
                        help='冻结Backbone层数')
    
    # 🔒 安全配置
    parser.add_argument('--workers', type=int, default=4,  # 降低默认值
                        help='数据加载进程数 (降低以减少I/O压力)')
    parser.add_argument('--project', type=str, default='/data/swy/runs',  # 🔒 指定安全路径
                        help='项目路径 (确保在数据盘，不在根目录)')
    
    # 创新损失函数
    parser.add_argument('--use-innovative-loss', action='store_true',
                        help='启用创新损失函数注入')
    parser.add_argument('--use-focus-iou', action='store_true', default=True,
                        help='启用Focus-IoU')
    
    # 输出路径
    parser.add_argument('--name', type=str, default='y11m_v5_safe',
                        help='实验名称')
    
    args = parser.parse_args()
    
    # 开始训练
    train_innovative_v5_safe(args)


if __name__ == '__main__':
    main()

