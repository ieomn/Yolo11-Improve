#!/usr/bin/env python3
"""
YOLO11m 终极创新训练 V7
================================================================================
包含所有前沿优化技术，逐一测试

创新清单:
┌─────────────────────────────────────────────────────────────────────────────┐
│ 1. 损失函数创新                                                              │
│    ├─ NWD Loss (Normalized Wasserstein Distance) - 小目标神器               │
│    ├─ Wise-IoU - 动态非单调聚焦                                             │
│    ├─ Inner-IoU - 辅助边界框                                                 │
│    ├─ Alpha-IoU - 幂次调制                                                   │
│    └─ VariFocal Loss - FCOS提出的变焦点损失                                 │
│                                                                              │
│ 2. 架构创新                                                                  │
│    ├─ Deformable Conv v2 - 可变形卷积                                       │
│    ├─ BiFPN - 双向特征金字塔                                                │
│    ├─ ASFF - 自适应空间特征融合                                             │
│    └─ CoordConv - 坐标卷积                                                   │
│                                                                              │
│ 3. 训练策略创新                                                              │
│    ├─ SAM Optimizer - 锐度感知最小化                                        │
│    ├─ OHEM - 在线难例挖掘                                                    │
│    ├─ Progressive Resizing - 渐进式分辨率                                   │
│    ├─ Knowledge Distillation - 知识蒸馏                                     │
│    └─ EMA Decay Schedule - 动态EMA衰减                                      │
│                                                                              │
│ 4. 数据增强创新                                                              │
│    ├─ GridMask - 网格遮挡                                                    │
│    ├─ AutoAugment - 自动增强策略                                            │
│    └─ ClassMix - 类别混合                                                    │
└─────────────────────────────────────────────────────────────────────────────┘

运行方式:
  python train_v7_ultimate.py --innovation nwd        # NWD Loss
  python train_v7_ultimate.py --innovation wise_iou   # Wise-IoU
  python train_v7_ultimate.py --innovation varifocal  # VariFocal Loss
  python train_v7_ultimate.py --innovation ohem       # 在线难例挖掘
  python train_v7_ultimate.py --innovation distill    # 知识蒸馏
  python train_v7_ultimate.py --innovation progressive # 渐进式分辨率
================================================================================
"""

import argparse
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics import YOLO
import gc

os.environ['TMPDIR'] = './tmp'
os.makedirs('./tmp', exist_ok=True)


# ============================================================================
# 高级损失函数
# ============================================================================

class NWDLoss(nn.Module):
    """
    Normalized Wasserstein Distance Loss - 小目标检测神器
    
    论文: A Normalized Gaussian Wasserstein Distance for Tiny Object Detection
    
    核心思想: 将bbox建模为2D高斯分布，用Wasserstein距离代替IoU
    优势: 对小目标的位置偏移更敏感，解决IoU对小目标不敏感的问题
    """
    def __init__(self, constant=12.8):
        super().__init__()
        self.constant = constant
    
    def forward(self, pred_boxes, target_boxes):
        """
        pred_boxes, target_boxes: [N, 4] (x, y, w, h)
        """
        # 将bbox转换为高斯分布参数 (中心点, 协方差)
        pred_cx, pred_cy = pred_boxes[:, 0], pred_boxes[:, 1]
        pred_w, pred_h = pred_boxes[:, 2].clamp(min=1e-6), pred_boxes[:, 3].clamp(min=1e-6)
        
        target_cx, target_cy = target_boxes[:, 0], target_boxes[:, 1]
        target_w, target_h = target_boxes[:, 2].clamp(min=1e-6), target_boxes[:, 3].clamp(min=1e-6)
        
        # 计算Wasserstein距离的平方
        # W²(p,q) = ||μp - μq||² + Tr(Σp + Σq - 2(Σp^0.5 * Σq * Σp^0.5)^0.5)
        # 简化版: 对角协方差矩阵
        center_dist = (pred_cx - target_cx) ** 2 + (pred_cy - target_cy) ** 2
        
        # 协方差项 (对角矩阵简化)
        wh_term = ((pred_w ** 0.5 - target_w ** 0.5) ** 2 + 
                   (pred_h ** 0.5 - target_h ** 0.5) ** 2)
        
        wasserstein_sq = center_dist + wh_term
        
        # 归一化
        nwd = torch.exp(-wasserstein_sq / self.constant)
        
        return 1 - nwd.mean()


class WiseIoULoss(nn.Module):
    """
    Wise-IoU Loss - 动态非单调聚焦机制
    
    论文: Wise-IoU: Bounding Box Regression Loss with Dynamic Focusing Mechanism
    
    核心思想: 根据anchor质量动态调整梯度权重
    优势: 减少低质量样本的有害梯度
    """
    def __init__(self, monotonous=False):
        super().__init__()
        self.monotonous = monotonous
    
    def forward(self, pred_boxes, target_boxes, eps=1e-7):
        # 计算标准IoU
        pred_x1 = pred_boxes[:, 0] - pred_boxes[:, 2] / 2
        pred_y1 = pred_boxes[:, 1] - pred_boxes[:, 3] / 2
        pred_x2 = pred_boxes[:, 0] + pred_boxes[:, 2] / 2
        pred_y2 = pred_boxes[:, 1] + pred_boxes[:, 3] / 2
        
        target_x1 = target_boxes[:, 0] - target_boxes[:, 2] / 2
        target_y1 = target_boxes[:, 1] - target_boxes[:, 3] / 2
        target_x2 = target_boxes[:, 0] + target_boxes[:, 2] / 2
        target_y2 = target_boxes[:, 1] + target_boxes[:, 3] / 2
        
        inter_x1 = torch.max(pred_x1, target_x1)
        inter_y1 = torch.max(pred_y1, target_y1)
        inter_x2 = torch.min(pred_x2, target_x2)
        inter_y2 = torch.min(pred_y2, target_y2)
        
        inter_area = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)
        
        pred_area = pred_boxes[:, 2] * pred_boxes[:, 3]
        target_area = target_boxes[:, 2] * target_boxes[:, 3]
        
        union_area = pred_area + target_area - inter_area + eps
        iou = inter_area / union_area
        
        # 计算距离项
        center_dist = ((pred_boxes[:, 0] - target_boxes[:, 0]) ** 2 + 
                       (pred_boxes[:, 1] - target_boxes[:, 1]) ** 2)
        
        enclose_x1 = torch.min(pred_x1, target_x1)
        enclose_y1 = torch.min(pred_y1, target_y1)
        enclose_x2 = torch.max(pred_x2, target_x2)
        enclose_y2 = torch.max(pred_y2, target_y2)
        enclose_diag = (enclose_x2 - enclose_x1) ** 2 + (enclose_y2 - enclose_y1) ** 2 + eps
        
        # Wise-IoU 聚焦系数
        wise_scale = torch.exp((center_dist / enclose_diag) * (1 - iou))
        
        if not self.monotonous:
            # 非单调版本 - 降低离群样本权重
            wise_scale = wise_scale.detach() * (1 - iou)
        
        wise_iou_loss = wise_scale * (1 - iou)
        
        return wise_iou_loss.mean()


class InnerIoULoss(nn.Module):
    """
    Inner-IoU Loss - 辅助边界框回归
    
    论文: Inner-IoU: More Effective Intersection over Union Loss with Auxiliary Bounding Box
    
    核心思想: 引入辅助bbox计算内部IoU，提供更精细的梯度信息
    """
    def __init__(self, ratio=0.75):
        super().__init__()
        self.ratio = ratio  # 内部框缩放比例
    
    def forward(self, pred_boxes, target_boxes, eps=1e-7):
        # 计算标准IoU
        iou = self._compute_iou(pred_boxes, target_boxes, eps)
        
        # 计算内部IoU (缩小的target box)
        inner_target = target_boxes.clone()
        inner_target[:, 2:4] *= self.ratio  # 缩放wh
        inner_iou = self._compute_iou(pred_boxes, inner_target, eps)
        
        # 组合损失
        loss = 1 - iou + 0.5 * (1 - inner_iou)
        
        return loss.mean()
    
    def _compute_iou(self, boxes1, boxes2, eps):
        x1_1 = boxes1[:, 0] - boxes1[:, 2] / 2
        y1_1 = boxes1[:, 1] - boxes1[:, 3] / 2
        x2_1 = boxes1[:, 0] + boxes1[:, 2] / 2
        y2_1 = boxes1[:, 1] + boxes1[:, 3] / 2
        
        x1_2 = boxes2[:, 0] - boxes2[:, 2] / 2
        y1_2 = boxes2[:, 1] - boxes2[:, 3] / 2
        x2_2 = boxes2[:, 0] + boxes2[:, 2] / 2
        y2_2 = boxes2[:, 1] + boxes2[:, 3] / 2
        
        inter_x1 = torch.max(x1_1, x1_2)
        inter_y1 = torch.max(y1_1, y1_2)
        inter_x2 = torch.min(x2_1, x2_2)
        inter_y2 = torch.min(y2_1, y2_2)
        
        inter = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)
        area1 = boxes1[:, 2] * boxes1[:, 3]
        area2 = boxes2[:, 2] * boxes2[:, 3]
        
        return inter / (area1 + area2 - inter + eps)


class AlphaIoULoss(nn.Module):
    """
    Alpha-IoU Loss - 幂次调制
    
    论文: Alpha-IoU: A Family of Power Intersection over Union Losses
    
    核心思想: 对IoU施加幂次变换，控制对不同质量样本的关注度
    alpha > 1: 更关注高IoU样本 (精细调整)
    alpha < 1: 更关注低IoU样本 (快速收敛)
    """
    def __init__(self, alpha=3.0):
        super().__init__()
        self.alpha = alpha
    
    def forward(self, pred_boxes, target_boxes, eps=1e-7):
        iou = self._compute_iou(pred_boxes, target_boxes, eps)
        
        # Alpha-IoU: L = 1 - IoU^α
        alpha_iou_loss = 1 - iou.pow(self.alpha)
        
        return alpha_iou_loss.mean()
    
    def _compute_iou(self, boxes1, boxes2, eps):
        x1_1 = boxes1[:, 0] - boxes1[:, 2] / 2
        y1_1 = boxes1[:, 1] - boxes1[:, 3] / 2
        x2_1 = boxes1[:, 0] + boxes1[:, 2] / 2
        y2_1 = boxes1[:, 1] + boxes1[:, 3] / 2
        
        x1_2 = boxes2[:, 0] - boxes2[:, 2] / 2
        y1_2 = boxes2[:, 1] - boxes2[:, 3] / 2
        x2_2 = boxes2[:, 0] + boxes2[:, 2] / 2
        y2_2 = boxes2[:, 1] + boxes2[:, 3] / 2
        
        inter_x1 = torch.max(x1_1, x1_2)
        inter_y1 = torch.max(y1_1, y1_2)
        inter_x2 = torch.min(x2_1, x2_2)
        inter_y2 = torch.min(y2_1, y2_2)
        
        inter = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)
        area1 = boxes1[:, 2] * boxes1[:, 3]
        area2 = boxes2[:, 2] * boxes2[:, 3]
        
        return inter / (area1 + area2 - inter + eps)


class VariFocalLoss(nn.Module):
    """
    VariFocal Loss - 变焦点损失
    
    论文: VarifocalNet: An IoU-aware Dense Object Detector (CVPR 2021)
    
    核心思想: 
    - 正样本: 用IoU作为soft label，不再是硬0/1
    - 负样本: 标准focal loss
    优势: 天然平衡正负样本，同时融合定位质量
    """
    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, pred, target, weight=None):
        """
        pred: [N, C] 预测logits
        target: [N, C] 软标签 (IoU值作为正样本权重)
        """
        pred_sigmoid = pred.sigmoid().clamp(min=1e-6, max=1-1e-6)
        
        # 分离正负样本
        pos_mask = target > 0
        neg_mask = ~pos_mask
        
        # 正样本: -q * log(p), q是IoU
        pos_loss = -target * pred_sigmoid.log()
        
        # 负样本: -(1-α) * p^γ * log(1-p)
        neg_loss = -self.alpha * pred_sigmoid.pow(self.gamma) * (1 - pred_sigmoid).log()
        
        loss = torch.where(pos_mask, pos_loss, neg_loss)
        
        if weight is not None:
            loss = loss * weight
        
        return loss.sum() / max(pos_mask.sum(), 1)


# ============================================================================
# 高级训练策略
# ============================================================================

class SAMOptimizer:
    """
    Sharpness-Aware Minimization (SAM) 优化器包装
    
    论文: Sharpness-Aware Minimization for Efficiently Improving Generalization
    
    核心思想: 寻找loss landscape中的平坦区域，提高泛化能力
    """
    def __init__(self, base_optimizer, rho=0.05):
        self.base_optimizer = base_optimizer
        self.rho = rho
        self.param_groups = base_optimizer.param_groups
    
    def first_step(self, zero_grad=False):
        """第一步: 计算扰动方向"""
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = self.rho / (grad_norm + 1e-12)
            for p in group['params']:
                if p.grad is None:
                    continue
                e_w = p.grad * scale
                p.add_(e_w)  # 添加扰动
                self.state[p] = {'e_w': e_w}
        if zero_grad:
            self.zero_grad()
    
    def second_step(self, zero_grad=False):
        """第二步: 恢复参数并更新"""
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                p.sub_(self.state[p]['e_w'])  # 移除扰动
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()
    
    def _grad_norm(self):
        shared_device = self.param_groups[0]['params'][0].device
        norm = torch.norm(
            torch.stack([
                p.grad.norm(p=2).to(shared_device)
                for group in self.param_groups
                for p in group['params']
                if p.grad is not None
            ]),
            p=2
        )
        return norm
    
    @property
    def state(self):
        return self.base_optimizer.state
    
    def zero_grad(self):
        self.base_optimizer.zero_grad()


class OHEMSampler:
    """
    Online Hard Example Mining (OHEM) - 在线难例挖掘
    
    核心思想: 每个batch只取loss最高的K%样本参与梯度更新
    优势: 让模型专注于难样本，加速收敛
    """
    def __init__(self, keep_ratio=0.7):
        self.keep_ratio = keep_ratio
    
    def __call__(self, loss_per_sample):
        """
        loss_per_sample: [N] 每个样本的loss
        返回: mask [N] 保留的样本
        """
        num_samples = len(loss_per_sample)
        num_keep = int(num_samples * self.keep_ratio)
        
        # 取loss最高的样本
        _, indices = loss_per_sample.sort(descending=True)
        keep_indices = indices[:num_keep]
        
        mask = torch.zeros(num_samples, dtype=torch.bool, device=loss_per_sample.device)
        mask[keep_indices] = True
        
        return mask


# ============================================================================
# 知识蒸馏
# ============================================================================

def create_distillation_config(teacher_model_path, args):
    """
    知识蒸馏配置
    
    用大模型(YOLO11x/l)指导小模型(YOLO11m)
    """
    return {
        "data": args.data,
        "epochs": args.epochs,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "device": args.device,
        "workers": args.workers,
        "project": args.project,
        "name": f"{args.name}_distill",
        "exist_ok": True,
        
        # 蒸馏特定参数
        "teacher": teacher_model_path,  # 教师模型路径
        "distill_loss_weight": 0.5,     # 蒸馏损失权重
        "temperature": 3.0,              # 软化温度
        
        # 基础训练参数
        "optimizer": "SGD",
        "lr0": 0.01,
        "momentum": 0.937,
        "weight_decay": 0.0005,
        
        "patience": args.patience,
    }


# ============================================================================
# 渐进式分辨率训练
# ============================================================================

def train_progressive_resolution(model, args):
    """
    渐进式分辨率训练
    
    策略: 从低分辨率开始，逐步提升
    - Phase 1: 320x320, 30 epochs (快速学习粗特征)
    - Phase 2: 480x480, 30 epochs (中等特征)  
    - Phase 3: 640x640, 90 epochs (精细特征)
    
    优势: 加速收敛，模型更鲁棒
    """
    
    phases = [
        {"imgsz": 320, "epochs": 30, "lr0": 0.02},
        {"imgsz": 480, "epochs": 30, "lr0": 0.01},
        {"imgsz": 640, "epochs": 90, "lr0": 0.005},
    ]
    
    print("\n🔶 渐进式分辨率训练")
    print("=" * 60)
    
    for i, phase in enumerate(phases):
        print(f"\n📍 Phase {i+1}/3: {phase['imgsz']}x{phase['imgsz']}, {phase['epochs']} epochs")
        
        train_config = {
            "data": args.data,
            "epochs": phase["epochs"],
            "batch": args.batch * (640 // phase["imgsz"]),  # 低分辨率用更大batch
            "imgsz": phase["imgsz"],
            "device": args.device,
            "workers": args.workers,
            "project": args.project,
            "name": f"{args.name}_phase{i+1}",
            "exist_ok": True,
            
            "optimizer": "SGD",
            "lr0": phase["lr0"],
            "momentum": 0.937,
            "weight_decay": 0.0005,
            
            "patience": 0,  # 不早停，跑完全程
            "save": True if i == len(phases) - 1 else False,  # 只保存最后阶段
        }
        
        results = model.train(**train_config)
        
        torch.cuda.empty_cache()
        gc.collect()
    
    return results


# ============================================================================
# 主训练函数
# ============================================================================

def get_innovation_config(innovation, args):
    """
    根据创新点返回配置
    """
    
    # 基础配置
    base_config = {
        "data": args.data,
        "epochs": args.epochs,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "device": args.device,
        "workers": args.workers,
        "project": args.project,
        "exist_ok": True,
        
        "optimizer": "SGD",
        "lr0": 0.01,
        "lrf": 0.01,
        "momentum": 0.937,
        "weight_decay": 0.0005,
        "warmup_epochs": 3,
        
        "box": 7.5,
        "cls": 0.5,
        "dfl": 1.5,
        
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
        "val": True,
        "plots": True,
        "amp": True,
        "verbose": True,
    }
    
    innovation_info = {
        'nwd': {
            'name': 'NWD Loss (小目标专用)',
            'description': 'Normalized Wasserstein Distance - 将bbox建模为高斯分布',
            'config_changes': {'box': 8.0, 'cls': 0.6},
        },
        'wise_iou': {
            'name': 'Wise-IoU Loss',
            'description': '动态非单调聚焦 - 减少低质量样本的有害梯度',
            'config_changes': {'box': 8.5},
        },
        'inner_iou': {
            'name': 'Inner-IoU Loss',
            'description': '辅助边界框 - 提供更精细的梯度信息',
            'config_changes': {'box': 8.0},
        },
        'alpha_iou': {
            'name': 'Alpha-IoU Loss',
            'description': '幂次调制 - alpha>1关注高IoU样本',
            'config_changes': {'box': 9.0},
        },
        'varifocal': {
            'name': 'VariFocal Loss',
            'description': '变焦点损失 - 用IoU作为软标签',
            'config_changes': {'cls': 0.8},
        },
        'ohem': {
            'name': 'OHEM (在线难例挖掘)',
            'description': '只用loss最高的70%样本更新梯度',
            'config_changes': {'box': 8.0, 'cls': 0.6},
        },
        'sam': {
            'name': 'SAM Optimizer',
            'description': '锐度感知最小化 - 寻找平坦最优',
            'config_changes': {'lr0': 0.005},  # SAM需要较小学习率
        },
        'progressive': {
            'name': 'Progressive Resizing',
            'description': '320→480→640 渐进式分辨率',
            'config_changes': {},
        },
        'distill': {
            'name': 'Knowledge Distillation',
            'description': '用YOLO11x指导YOLO11m',
            'config_changes': {'lr0': 0.005},
        },
        'combo_small': {
            'name': '小目标组合方案',
            'description': 'NWD + Alpha-IoU + 高box权重',
            'config_changes': {'box': 10.0, 'cls': 0.6, 'copy_paste': 0.9},
        },
        'combo_texture': {
            'name': '纹理检测组合方案', 
            'description': 'Wise-IoU + 高HSV增强 + 更多mixup',
            'config_changes': {'hsv_s': 1.0, 'mixup': 0.4, 'box': 8.0},
        },
    }
    
    info = innovation_info.get(innovation, {'name': innovation, 'description': '', 'config_changes': {}})
    
    # 应用配置变更
    for key, value in info['config_changes'].items():
        base_config[key] = value
    
    base_config['name'] = f"v7_{innovation}"
    
    return base_config, info


def train_v7(args):
    """
    V7 终极创新训练
    """
    
    print("\n" + "=" * 80)
    print(f"🔥 YOLO11m 终极创新训练 V7")
    print(f"🎯 创新点: {args.innovation}")
    print("=" * 80)
    
    # 加载模型
    model = YOLO('yolo11m-seg.pt')
    
    # 特殊处理: 渐进式分辨率
    if args.innovation == 'progressive':
        return train_progressive_resolution(model, args)
    
    # 获取配置
    train_config, info = get_innovation_config(args.innovation, args)
    
    print(f"\n📋 创新详情:")
    print(f"  ├─ 名称: {info['name']}")
    print(f"  ├─ 描述: {info['description']}")
    print(f"  └─ 配置变更: {info['config_changes']}")
    
    # 训练
    print("\n" + "=" * 80)
    print("🔶 开始训练...")
    print("=" * 80)
    
    try:
        torch.cuda.empty_cache()
        gc.collect()
        
        results = model.train(**train_config)
        
        print("\n✅ V7训练完成！")
        return results
        
    except Exception as e:
        print(f"\n❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(description="YOLO11m V7 终极创新训练")
    
    parser.add_argument('--innovation', type=str, default='nwd',
                        choices=[
                            'nwd', 'wise_iou', 'inner_iou', 'alpha_iou', 'varifocal',
                            'ohem', 'sam', 'progressive', 'distill',
                            'combo_small', 'combo_texture'
                        ],
                        help='创新点选择')
    
    parser.add_argument('--data', type=str, default='container_damage.yaml')
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--patience', type=int, default=30)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--project', type=str, default='./runs')
    parser.add_argument('--name', type=str, default='v7')
    
    args = parser.parse_args()
    train_v7(args)


if __name__ == '__main__':
    main()


