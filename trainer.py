#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import time
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from typing import Dict, List, Tuple, Optional, Union, Any
from tqdm import tqdm
import json
from pathlib import Path

from lstm_model import DDoSPredictionLSTM
from loss_function import create_loss_function

logger = logging.getLogger(__name__)


class DDoSModelTrainer:
    """
    DDoS预测模型训练器

    负责模型训练、验证、测试及可视化
    """

    def __init__(self,
                 model: nn.Module,
                 loss_fn: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 device: torch.device,
                 scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                 checkpoint_dir: str = 'checkpoints'):
        """
        初始化训练器

        Args:
            model: 模型实例
            loss_fn: 损失函数
            optimizer: 优化器
            device: 设备(CPU/GPU)
            scheduler: 学习率调度器(可选)
            checkpoint_dir: 检查点保存目录
        """
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler
        self.checkpoint_dir = checkpoint_dir

        # 创建检查点目录
        os.makedirs(checkpoint_dir, exist_ok=True)

        # 训练历史记录
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_auc': [],
            'val_precision': [],
            'val_recall': [],
            'val_f1': [],
            'learning_rate': []
        }

        # 早停相关参数
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        self.patience = 5
        self.patience_counter = 0
        self.min_delta = 0.001

        logger.info(f"初始化DDoSModelTrainer: model={type(model).__name__}, "
                    f"loss_fn={type(loss_fn).__name__}, optimizer={type(optimizer).__name__}")

    def train_epoch(self, train_loader: DataLoader, accumulation_steps: int = 4) -> float:
        """
        训练一个epoch

        Args:
            train_loader: 训练数据加载器
            accumulation_steps: 梯度累积步数，默认4

        Returns:
            平均训练损失
        """
        self.model.train()
        total_loss = 0
        batch_count = 0

        # 使用tqdm显示进度条
        pbar = tqdm(train_loader, desc="Training", leave=False)

        # 重置梯度
        self.optimizer.zero_grad()

        for batch_idx, (data, target) in enumerate(pbar):
            # 将数据移到设备
            data, target = data.to(self.device), target.to(self.device)

            # 前向传播
            output = self.model(data)
            loss = self.loss_fn(output, target)

            # 计算梯度
            loss = loss / accumulation_steps  # 缩放损失
            loss.backward()

            # 更新总损失(使用未缩放的损失)
            total_loss += loss.item() * accumulation_steps
            batch_count += 1

            # 更新进度条
            pbar.set_postfix({'loss': f"{total_loss / batch_count:.4f}"})

            # 梯度累积
            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                self.optimizer.zero_grad()

        # 计算平均损失
        avg_loss = total_loss / batch_count
        return avg_loss

    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        验证模型

        Args:
            val_loader: 验证数据加载器

        Returns:
            包含验证损失和指标的字典
        """
        self.model.eval()
        val_loss = 0
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for data, target in val_loader:
                # 将数据移到设备
                data, target = data.to(self.device), target.to(self.device)

                # 前向传播
                output = self.model(data)

                # 计算损失
                loss = self.loss_fn(output, target)
                val_loss += loss.item()

                # 保存预测和目标
                all_predictions.append(output.cpu().numpy())
                all_targets.append(target.cpu().numpy())

        # 计算平均损失
        val_loss /= len(val_loader)

        # 连接所有批次的预测和目标
        all_predictions = np.concatenate(all_predictions).ravel()
        all_targets = np.concatenate(all_targets).ravel()

        # 计算指标
        # 对于二分类，需要设置阈值将概率转换为类别
        threshold = 0.5
        binary_predictions = (all_predictions >= threshold).astype(int)

        try:
            auc = roc_auc_score(all_targets, all_predictions)
        except:
            logger.warning("无法计算AUC，可能只有一个类别")
            auc = 0.0

        precision = precision_score(all_targets, binary_predictions, zero_division=0)
        recall = recall_score(all_targets, binary_predictions, zero_division=0)
        f1 = f1_score(all_targets, binary_predictions, zero_division=0)

        # 返回结果
        metrics = {
            'val_loss': val_loss,
            'val_auc': auc,
            'val_precision': precision,
            'val_recall': recall,
            'val_f1': f1
        }

        return metrics

    def fit(self,
            train_loader: DataLoader,
            val_loader: DataLoader,
            epochs: int = 50,
            accumulation_steps: int = 4,
            early_stopping: bool = True) -> Dict[str, List[float]]:
        """
        训练模型

        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            epochs: 训练轮数
            accumulation_steps: 梯度累积步数
            early_stopping: 是否启用早停

        Returns:
            训练历史记录
        """
        logger.info(f"开始训练: epochs={epochs}, accumulation_steps={accumulation_steps}, "
                    f"early_stopping={early_stopping}")

        start_time = time.time()

        for epoch in range(epochs):
            epoch_start = time.time()

            # 训练一个epoch
            train_loss = self.train_epoch(train_loader, accumulation_steps)

            # 验证
            metrics = self.validate(val_loader)
            val_loss = metrics['val_loss']
            val_auc = metrics['val_auc']
            val_precision = metrics['val_precision']
            val_recall = metrics['val_recall']
            val_f1 = metrics['val_f1']

            # 获取当前学习率
            current_lr = self.optimizer.param_groups[0]['lr']

            # 更新历史记录
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_auc'].append(val_auc)
            self.history['val_precision'].append(val_precision)
            self.history['val_recall'].append(val_recall)
            self.history['val_f1'].append(val_f1)
            self.history['learning_rate'].append(current_lr)

            # 更新学习率
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            # 计算epoch用时
            epoch_time = time.time() - epoch_start

            # 输出训练信息
            logger.info(f"Epoch {epoch + 1}/{epochs} - "
                        f"train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f}, "
                        f"val_auc: {val_auc:.4f}, val_f1: {val_f1:.4f}, "
                        f"lr: {current_lr:.6f}, time: {epoch_time:.1f}s")

            # 检查是否保存最佳模型
            if val_loss < self.best_val_loss - self.min_delta:
                self.best_val_loss = val_loss
                self.best_epoch = epoch
                self.patience_counter = 0

                # 保存最佳模型
                self._save_checkpoint(epoch, val_loss, metrics, is_best=True)
                logger.info(f"保存最佳模型: epoch={epoch + 1}, val_loss={val_loss:.4f}")
            else:
                self.patience_counter += 1

                # 每5个epoch保存一次常规检查点
                if (epoch + 1) % 5 == 0:
                    self._save_checkpoint(epoch, val_loss, metrics, is_best=False)

            # 检查是否早停
            if early_stopping and self.patience_counter >= self.patience:
                logger.info(f"早停触发: {self.patience}个epoch无改善")
                break

        # 训练完成
        total_time = time.time() - start_time
        logger.info(f"训练完成: 总用时={total_time:.1f}s, 最佳epoch={self.best_epoch + 1}, "
                    f"最佳val_loss={self.best_val_loss:.4f}")

        # 可视化训练过程
        self.visualize_training()

        return self.history

    def _save_checkpoint(self,
                         epoch: int,
                         val_loss: float,
                         metrics: Dict[str, float],
                         is_best: bool = False) -> None:
        """
        保存检查点

        Args:
            epoch: 当前epoch
            val_loss: 验证损失
            metrics: 验证指标
            is_best: 是否为最佳模型
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'metrics': metrics,
            'history': self.history
        }

        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        # 保存检查点
        if is_best:
            checkpoint_path = os.path.join(self.checkpoint_dir, 'best_model.pt')
        else:
            checkpoint_path = os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch + 1}.pt')

        torch.save(checkpoint, checkpoint_path)

    def load_checkpoint(self, checkpoint_path: str, load_optimizer: bool = True) -> int:
        """
        加载检查点

        Args:
            checkpoint_path: 检查点路径
            load_optimizer: 是否加载优化器状态

        Returns:
            加载的epoch
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # 加载模型
        self.model.load_state_dict(checkpoint['model_state_dict'])

        # 加载优化器
        if load_optimizer and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # 加载调度器
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        # 加载历史记录
        if 'history' in checkpoint:
            self.history = checkpoint['history']

        # 更新最佳验证损失
        if 'val_loss' in checkpoint:
            self.best_val_loss = checkpoint['val_loss']

        logger.info(f"从 {checkpoint_path} 加载检查点: epoch={checkpoint['epoch'] + 1}")

        return checkpoint['epoch']

    def visualize_training(self, save_dir: str = 'plots') -> None:
        """
        可视化训练过程

        Args:
            save_dir: 图表保存目录
        """
        os.makedirs(save_dir, exist_ok=True)

        # 设置风格
        sns.set(style="whitegrid")
        plt.figure(figsize=(12, 10))

        # 1. 绘制训练/验证损失
        plt.subplot(2, 2, 1)
        epochs = range(1, len(self.history['train_loss']) + 1)
        plt.plot(epochs, self.history['train_loss'], 'b-', label='Training Loss')
        plt.plot(epochs, self.history['val_loss'], 'r-', label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()

        # 2. 绘制验证指标
        plt.subplot(2, 2, 2)
        plt.plot(epochs, self.history['val_auc'], 'g-', label='AUC')
        plt.plot(epochs, self.history['val_precision'], 'm-', label='Precision')
        plt.plot(epochs, self.history['val_recall'], 'c-', label='Recall')
        plt.plot(epochs, self.history['val_f1'], 'y-', label='F1')
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.title('Validation Metrics')
        plt.legend()

        # 3. 绘制学习率变化
        plt.subplot(2, 2, 3)
        plt.plot(epochs, self.history['learning_rate'], 'k-')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Schedule')

        # 保存图表
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'training_history.png'))
        plt.close()

        logger.info(f"训练可视化保存至 {save_dir}/training_history.png")

        # 保存训练历史记录为JSON
        with open(os.path.join(save_dir, 'training_history.json'), 'w') as f:
            json.dump(self.history, f, indent=4)

    def test(self, test_loader: DataLoader) -> Dict[str, float]:
        """
        在测试集上评估模型

        Args:
            test_loader: 测试数据加载器

        Returns:
            测试指标
        """
        return self.validate(test_loader)  # 复用验证函数

    @staticmethod
    def prepare_data_loaders(X_train: np.ndarray, y_train: np.ndarray,
                             X_val: np.ndarray, y_val: np.ndarray,
                             batch_size: int = 64) -> Tuple[DataLoader, DataLoader]:
        """
        准备数据加载器

        Args:
            X_train: 训练特征
            y_train: 训练标签
            X_val: 验证特征
            y_val: 验证标签
            batch_size: 批大小

        Returns:
            训练和验证数据加载器
        """
        # 转换为PyTorch张量
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)
        X_val_tensor = torch.FloatTensor(X_val)
        y_val_tensor = torch.FloatTensor(y_val).unsqueeze(1)

        # 创建数据集
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

        # 创建数据加载器
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        return train_loader, val_loader


def setup_trainer(model: nn.Module,
                  train_dataset: torch.utils.data.Dataset,  # 修改：使用Dataset而不是labels数组
                  device: torch.device,
                  lr: float = 0.001,
                  weight_decay: float = 0.001) -> DDoSModelTrainer:
    """
    设置训练器

    Args:
        model: 模型实例
        train_dataset: 训练数据集，用于计算类别权重
        device: 设备(CPU/GPU)
        lr: 学习率
        weight_decay: 权重衰减(L2正则化)

    Returns:
        配置好的训练器实例
    """
    # 计算类别权重 - 从数据集中获取所有标签
    all_labels = []
    for _, y in train_dataset:
        all_labels.append(y.numpy())
    train_labels = np.concatenate(all_labels)

    # 创建损失函数
    loss_fn = create_loss_function(train_labels)

    # 创建优化器
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # 创建学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-6, verbose=True
    )

    # 创建训练器
    trainer = DDoSModelTrainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=device,
        scheduler=scheduler
    )

    return trainer


if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")

    # 创建模型实例（用于测试）
    input_dim = 128  # 与模型匹配的输入维度
    model = DDoSPredictionLSTM(input_dim=input_dim).to(device)

    # 创建随机训练数据（用于测试）
    num_samples = 1000
    X_train = np.random.randn(num_samples, input_dim)
    y_train = np.random.randint(0, 2, num_samples)
    X_val = np.random.randn(num_samples // 5, input_dim)
    y_val = np.random.randint(0, 2, num_samples // 5)

    # 准备数据加载器
    train_loader, val_loader = DDoSModelTrainer.prepare_data_loaders(
        X_train, y_train, X_val, y_val, batch_size=64
    )

    # 设置训练器
    trainer = setup_trainer(model, y_train, device)

    # 进行训练（简短的）
    trainer.fit(train_loader, val_loader, epochs=2)

    logger.info("测试完成")