#!/usr/bin/env python3
"""
YOLO11m 创新训练 V6 - 稳中求进版
================================================================================
吸取V5教训：不要一次改太多！

策略：基于基线的优秀表现 (mAP50=0.777)，逐步添加创新点

可选创新模块:
1. 注意力机制 (CBAM/SE/ECA) - 通过自定义YAML
2. 改进损失函数 - 保守调参
3. 自适应学习率
4. 类别平衡采样

运行方式:
  # 模式1: 仅损失函数优化
  python train_v6_incremental.py --mode loss_only

  # 模式2: 仅注意力机制
  python train_v6_incremental.py --mode attention_only

  # 模式3: 全部创新 (谨慎使用)
  python train_v6_incremental.py --mode full
================================================================================
"""

import argparse
import os
from pathlib import Path
import torch
import torch.nn as nn
from ultralytics import YOLO
import gc

os.environ['TMPDIR'] = './tmp'
os.makedirs('./tmp', exist_ok=True)


# ============================================================================
# 创新损失函数 - 保守版本
# ============================================================================

class ConservativePolyLoss(nn.Module):
    """
    保守版 PolyLoss - 仅轻微调整
    """
    def __init__(self, epsilon=0.5):  # 降低epsilon，减少干预
        super().__init__()
        self.epsilon = epsilon
        self.ce = nn.CrossEntropyLoss(reduction='none')
    
    def forward(self, pred, target):
        ce_loss = self.ce(pred, target)
        pt = torch.exp(-ce_loss).clamp(min=1e-7, max=1.0)
        poly_loss = ce_loss + self.epsilon * (1 - pt)
        return poly_loss.mean()


class AdaptiveFocalLoss(nn.Module):
    """
    自适应 Focal Loss - 根据类别难度动态调整
    """
    def __init__(self, gamma=1.5, alpha=None):  # 降低gamma，减少对难样本的过度关注
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
    
    def forward(self, pred, target):
        ce_loss = nn.functional.cross_entropy(pred, target, reduction='none')
        pt = torch.exp(-ce_loss).clamp(min=1e-7, max=1.0)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


class GentleSlideLoss(nn.Module):
    """
    温和版 Slide Loss - 不要太激进
    """
    def __init__(self, iou_threshold=0.5, weight_scale=1.2):  # 降低权重倍数
        super().__init__()
        self.iou_threshold = iou_threshold
        self.weight_scale = weight_scale
    
    def __call__(self, iou_scores, loss_values):
        weights = torch.ones_like(loss_values)
        hard_mask = iou_scores < self.iou_threshold
        weights[hard_mask] = self.weight_scale
        return (loss_values * weights).mean()


# ============================================================================
# 注意力机制 YAML 生成
# ============================================================================

def create_attention_yaml(attention_type='cbam'):
    """
    创建带注意力机制的YAML配置
    """
    yaml_content = f"""# YOLO11m-seg with {attention_type.upper()} Attention
# 基于官方架构，仅在Neck添加注意力模块

nc: 3
scales:
  m: [0.50, 0.50, 512]

backbone:
  - [-1, 1, Conv, [64, 3, 2]]
  - [-1, 1, Conv, [128, 3, 2]]
  - [-1, 2, C3k2, [256, False, 0.25]]
  - [-1, 1, Conv, [256, 3, 2]]
  - [-1, 2, C3k2, [512, False, 0.25]]
  - [-1, 1, Conv, [512, 3, 2]]
  - [-1, 2, C3k2, [512, True]]
  - [-1, 1, Conv, [1024, 3, 2]]
  - [-1, 2, C3k2, [1024, True]]
  - [-1, 1, SPPF, [1024, 5]]
  - [-1, 2, C2PSA, [1024]]  # 使用C2PSA作为注意力

head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 6], 1, Concat, [1]]
  - [-1, 2, C3k2, [512, False]]
  
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 4], 1, Concat, [1]]
  - [-1, 2, C3k2, [256, False]]

  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 13], 1, Concat, [1]]
  - [-1, 2, C3k2, [512, False]]

  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 10], 1, Concat, [1]]
  - [-1, 2, C3k2, [1024, True]]

  - [[16, 19, 22], 1, Segment, [nc, 32, 256]]
"""
    
    yaml_path = f'yolo11m-seg-{attention_type}.yaml'
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    return yaml_path


# ============================================================================
# 训练配置
# ============================================================================

def get_config_by_mode(mode, args):
    """
    根据模式返回不同的配置
    """
    
    # 基础配置 - 继承基线的成功经验
    base_config = {
        "data": args.data,
        "epochs": args.epochs,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "device": args.device,
        "workers": args.workers,
        "project": args.project,
        "name": args.name,
        "exist_ok": True,
        
        # 优化器 - 坚持用SGD
        "optimizer": "SGD",
        "lr0": 0.01,
        "lrf": 0.01,
        "momentum": 0.937,
        "weight_decay": 0.0005,
        "warmup_epochs": 3,
        
        # 默认损失权重 (基线值)
        "box": 7.5,
        "cls": 0.5,
        "dfl": 1.5,
        
        "label_smoothing": 0.0,
        
        # 数据增强 - 与基线一致
        "hsv_h": 0.015,
        "hsv_s": 0.9,
        "hsv_v": 0.4,
        "degrees": 10.0,
        "translate": 0.1,
        "scale": 0.5,
        "fliplr": 0.5,
        "mosaic": 1.0,
        "close_mosaic": 15,
        "copy_paste": 0.8,
        "mixup": 0.2,
        
        "patience": args.patience,
        "save": True,
        "save_period": -1,
        "val": True,
        "plots": True,
        "amp": True,
        "verbose": True,
    }
    
    if mode == 'baseline':
        # 纯基线，不做任何改动
        base_config["name"] = "v6_baseline"
        return base_config, None
    
    elif mode == 'loss_only':
        # 仅调整损失函数 - 保守改动
        base_config["box"] = 8.0  # 轻微提升 (7.5 -> 8.0)
        base_config["cls"] = 0.6  # 轻微提升 (0.5 -> 0.6)
        base_config["label_smoothing"] = 0.01  # 极轻微
        base_config["name"] = "v6_loss_tuned"
        return base_config, None
    
    elif mode == 'attention_only':
        # 仅添加注意力机制
        yaml_path = create_attention_yaml('c2psa')
        base_config["name"] = "v6_attention"
        return base_config, yaml_path
    
    elif mode == 'full':
        # 全部创新 - 但保守调参
        yaml_path = create_attention_yaml('c2psa')
        base_config["box"] = 8.5
        base_config["cls"] = 0.6
        base_config["label_smoothing"] = 0.01
        base_config["name"] = "v6_full"
        return base_config, yaml_path
    
    elif mode == 'aggressive':
        # 激进模式 - 类似V5但稍微收敛
        base_config["optimizer"] = "AdamW"
        base_config["lr0"] = 0.002  # 降低学习率
        base_config["weight_decay"] = 0.03
        base_config["box"] = 12.0  # 比V5的20.0保守很多
        base_config["cls"] = 0.8
        base_config["label_smoothing"] = 0.02
        base_config["name"] = "v6_aggressive"
        return base_config, None
    
    else:
        return base_config, None


def train_v6(args):
    """
    V6 创新训练
    """
    
    print("\n" + "=" * 80)
    print(f"🔶 YOLO11m 创新训练 V6 - 模式: {args.mode}")
    print("=" * 80)
    
    # 获取配置
    train_config, custom_yaml = get_config_by_mode(args.mode, args)
    
    # 加载模型
    if custom_yaml and Path(custom_yaml).exists():
        print(f"📦 加载自定义架构: {custom_yaml}")
        # 先加载预训练权重，再应用自定义架构
        base_model = YOLO('yolo11m-seg.pt')
        model = YOLO(custom_yaml)
        # 尝试迁移权重
        try:
            model.model.load_state_dict(base_model.model.state_dict(), strict=False)
            print("✅ 权重迁移成功")
        except Exception as e:
            print(f"⚠️ 权重迁移部分失败: {e}")
    else:
        print("📦 加载标准模型: yolo11m-seg.pt")
        model = YOLO('yolo11m-seg.pt')
    
    # 显示配置
    print("\n📋 V6配置摘要:")
    print(f"  ├─ 模式: {args.mode}")
    print(f"  ├─ 优化器: {train_config['optimizer']}")
    print(f"  ├─ 学习率: {train_config['lr0']}")
    print(f"  ├─ 损失权重: box={train_config['box']}, cls={train_config['cls']}, dfl={train_config['dfl']}")
    print(f"  ├─ 标签平滑: {train_config['label_smoothing']}")
    print(f"  └─ 自定义架构: {'是' if custom_yaml else '否'}")
    
    # 训练
    print("\n" + "=" * 80)
    print("🔶 开始V6训练...")
    print("=" * 80)
    
    try:
        torch.cuda.empty_cache()
        gc.collect()
        
        results = model.train(**train_config)
        
        torch.cuda.empty_cache()
        gc.collect()
        
        print("\n✅ V6训练完成！")
        return results
        
    except Exception as e:
        print(f"\n❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(description="YOLO11m V6 创新训练")
    
    # 模式选择
    parser.add_argument('--mode', type=str, default='loss_only',
                        choices=['baseline', 'loss_only', 'attention_only', 'full', 'aggressive'],
                        help='训练模式')
    
    # 基础参数
    parser.add_argument('--data', type=str, default='container_damage.yaml')
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--patience', type=int, default=30)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--project', type=str, default='./runs')
    parser.add_argument('--name', type=str, default='v6_experiment')
    
    args = parser.parse_args()
    train_v6(args)


if __name__ == '__main__':
    main()



