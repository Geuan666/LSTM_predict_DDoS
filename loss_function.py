#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import numpy as np
import logging
from typing import Optional, Tuple, Union

logger = logging.getLogger(__name__)


class WeightedBCELoss(nn.Module):
    """
    带权二元交叉熵损失函数

    用于解决类别不平衡问题的损失函数，可单独设置正负样本权重
    """

    def __init__(self, pos_weight: float = None, neg_weight: float = None):
        """
        初始化

        Args:
            pos_weight: 正样本权重，None则自动计算
            neg_weight: 负样本权重，None则自动计算
        """
        super().__init__()
        self.pos_weight = pos_weight
        self.neg_weight = neg_weight
        self.epsilon = 1e-12  # 防止log(0)

        logger.info(f"初始化WeightedBCELoss: pos_weight={pos_weight}, neg_weight={neg_weight}")

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        计算损失

        Args:
            predictions: 预测概率，形状 [batch_size, 1]
            targets: 真实标签，形状 [batch_size, 1] 或 [batch_size]

        Returns:
            计算得到的损失值
        """
        # 确保targets的形状和predictions一致
        if targets.dim() != predictions.dim():
            targets = targets.view_as(predictions)

        # 使用实际样本数据动态计算权重（如果未提供）
        if self.pos_weight is None or self.neg_weight is None:
            pos_weight, neg_weight = calculate_class_weights(targets)

            if self.pos_weight is None:
                self.pos_weight = pos_weight
            if self.neg_weight is None:
                self.neg_weight = neg_weight

            logger.debug(f"动态计算权重: pos_weight={self.pos_weight:.4f}, neg_weight={self.neg_weight:.4f}")

        # 裁剪预测值，防止出现0或1
        predictions = torch.clamp(predictions, self.epsilon, 1 - self.epsilon)

        # 计算正样本和负样本的损失
        pos_loss = -targets * torch.log(predictions)
        neg_loss = -(1 - targets) * torch.log(1 - predictions)

        # 应用权重
        weighted_pos_loss = self.pos_weight * pos_loss
        weighted_neg_loss = self.neg_weight * neg_loss

        # 合并损失
        loss = weighted_pos_loss + weighted_neg_loss

        # 计算平均损失
        return torch.mean(loss)


def calculate_class_weights(targets: torch.Tensor) -> Tuple[float, float]:
    """
    根据目标标签计算类别权重

    使用公式: w_pos = N_total / (2 * N_pos) 和 w_neg = N_total / (2 * N_neg)

    Args:
        targets: 目标标签张量

    Returns:
        (pos_weight, neg_weight) 类别权重元组
    """
    # 将张量转为一维
    if targets.dim() > 1:
        targets = targets.view(-1)

    # 计算正负样本数量
    n_samples = targets.size(0)
    n_pos = torch.sum(targets).item()
    n_neg = n_samples - n_pos

    # 防止除零错误
    if n_pos == 0:
        pos_weight = 1.0
        logger.warning("没有正样本，设置pos_weight=1.0")
    else:
        pos_weight = n_samples / (2 * n_pos)

    if n_neg == 0:
        neg_weight = 1.0
        logger.warning("没有负样本，设置neg_weight=1.0")
    else:
        neg_weight = n_samples / (2 * n_neg)

    # 限制权重上限，防止极端不平衡情况
    pos_weight = min(pos_weight, 10.0)
    neg_weight = min(neg_weight, 10.0)

    logger.info(f"样本统计: 总样本={n_samples}, 正样本={n_pos}({n_pos / n_samples:.2%}), "
                f"负样本={n_neg}({n_neg / n_samples:.2%})")
    logger.info(f"计算得到权重: pos_weight={pos_weight:.4f}, neg_weight={neg_weight:.4f}")

    return pos_weight, neg_weight


def create_loss_function(targets: Optional[Union[torch.Tensor, np.ndarray]] = None) -> nn.Module:
    """
    创建损失函数实例

    可以基于训练数据自动计算类别权重

    Args:
        targets: 训练集标签，用于计算类别权重

    Returns:
        损失函数实例
    """
    if targets is not None:
        # 如果输入是numpy数组，转换为torch张量
        if isinstance(targets, np.ndarray):
            targets = torch.FloatTensor(targets)

        # 计算类别权重
        pos_weight, neg_weight = calculate_class_weights(targets)

        # 创建带权损失函数
        loss_fn = WeightedBCELoss(pos_weight=pos_weight, neg_weight=neg_weight)
    else:
        # 创建默认损失函数，将在首次前向传播时计算权重
        loss_fn = WeightedBCELoss()

    return loss_fn


if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # 测试损失函数
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 创建测试数据
    batch_size = 16
    predictions = torch.rand(batch_size, 1).to(device)
    targets = torch.randint(0, 2, (batch_size, 1)).float().to(device)

    # 测试自动计算权重
    loss_fn = create_loss_function(targets)
    loss = loss_fn(predictions, targets)

    logger.info(f"损失值: {loss.item():.4f}")