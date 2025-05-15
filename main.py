#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import logging
import argparse
import numpy as np
import torch
import json
from datetime import datetime
from typing import Dict, List
from torch.utils.data import DataLoader, random_split
from lstm_model import DDoSPredictionLSTM
from trainer import DDoSModelTrainer, setup_trainer
from utils import setup_logger, compute_node_attack_probability, set_seed
from data_processor import DDoSDataset


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='基于LSTM的SDN网络DDoS攻击预测模型')

    # 基本参数
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'predict'],
                        help='运行模式: train(训练) 或 predict(预测)')
    parser.add_argument('--data_path', type=str, required=True,
                        help='数据文件或目录路径')
    parser.add_argument('--output_dir', type=str, default='output',
                        help='输出目录路径')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')

    # 训练相关参数
    parser.add_argument('--epochs', type=int, default=50,
                        help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='批处理大小')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='学习率')
    parser.add_argument('--weight_decay', type=float, default=0.001,
                        help='权重衰减(L2正则化)')
    parser.add_argument('--early_stopping', action='store_true',
                        help='是否启用早停')
    parser.add_argument('--accumulation_steps', type=int, default=4,
                        help='梯度累积步数')

    # 模型相关参数
    parser.add_argument('--model_path', type=str, default=None,
                        help='模型路径(预测模式必需，训练模式下为权重加载路径)')
    parser.add_argument('--input_dim', type=int, default=128,
                        help='输入特征维度')
    parser.add_argument('--embedding_dim', type=int, default=32,
                        help='嵌入维度')
    parser.add_argument('--hidden_dim', type=int, default=64,
                        help='LSTM隐藏层维度')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='LSTM层数')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='Dropout率')
    parser.add_argument('--attention_heads', type=int, default=4,
                        help='注意力头数')

    # 数据处理相关参数
    parser.add_argument('--threshold_points', type=str, default='4,8,16,32,64,128',
                        help='阈值点列表，以逗号分隔')
    parser.add_argument('--window_size', type=int, default=15,
                        help='滑动窗口大小(秒)')
    parser.add_argument('--step_size', type=int, default=1,
                        help='滑动步长(秒)')
    parser.add_argument('--n_workers', type=int, default=4,
                        help='并行处理的工作进程数')

    # 预测相关参数
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='攻击判定阈值')

    args = parser.parse_args()

    # 转换阈值点字符串为列表
    args.threshold_points = [int(x) for x in args.threshold_points.split(',')]

    # 检查预测模式必需参数
    if args.mode == 'predict' and args.model_path is None:
        parser.error("预测模式需要指定 --model_path")

    return args


def train(args, logger):
    """训练模式"""
    logger.info("启动训练模式")

    # 设置随机种子
    set_seed(args.seed)

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")

    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"train_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_dir = os.path.join(output_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    logger.info(f"输出目录: {output_dir}")

    # 保存训练参数
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)

    # 初始化数据集
    logger.info("初始化数据集")
    try:
        # 设置是否应用PCA：确保训练和推理一致
        apply_pca = True  # 可以从命令行参数中获取这个设置

        train_dataset = DDoSDataset(
            data_path=args.data_path,
            threshold_points=args.threshold_points,
            window_size=args.window_size,
            step_size=args.step_size,
            train=True
        )

        # 保存预处理器
        preprocessor_path = os.path.join(output_dir, 'preprocessor.pkl')
        train_dataset.get_preprocessor().save_preprocessors(preprocessor_path)

        # 记录PCA应用状态到配置文件中，确保后续推理时知道是否使用了PCA
        with open(os.path.join(output_dir, 'pca_config.json'), 'w') as f:
            json.dump({"applied_pca": apply_pca}, f)

        # 划分训练集和验证集
        train_size = int(0.8 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_subset, val_subset = random_split(
            train_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(args.seed)
        )

        logger.info(f"数据集大小: {len(train_dataset)}, 训练集: {len(train_subset)}, 验证集: {len(val_subset)}")

        # 创建数据加载器
        train_loader = DataLoader(
            train_subset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.n_workers,
            pin_memory=torch.cuda.is_available()
        )

        val_loader = DataLoader(
            val_subset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.n_workers,
            pin_memory=torch.cuda.is_available()
        )

        # 计算特征维度
        sample_features, _ = train_dataset[0]
        input_dim = sample_features.shape[0]
        logger.info(f"特征维度: {input_dim}")

        # 将实际特征维度保存到配置中，确保后续加载模型时使用正确的维度
        with open(os.path.join(output_dir, 'feature_dim.json'), 'w') as f:
            json.dump({"input_dim": int(input_dim)}, f)

    except Exception as e:
        logger.error(f"数据集创建失败: {str(e)}")
        return

    # 创建模型
    logger.info("创建模型")
    model = DDoSPredictionLSTM(
        input_dim=input_dim,  # 使用实际特征维度
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        attention_heads=args.attention_heads,
        attention_dim=args.hidden_dim,
        threshold_points=len(args.threshold_points)
    ).to(device)

    # 如果指定了模型路径，则加载权重
    if args.model_path and os.path.exists(args.model_path):
        logger.info(f"从 {args.model_path} 加载模型权重")
        loaded_model = DDoSPredictionLSTM.load(args.model_path, device)
        model.load_state_dict(loaded_model.state_dict())

    # 获取训练标签用于计算类别权重
    all_labels = []
    for _, label in train_subset:
        all_labels.append(label.numpy())
    train_labels = np.concatenate(all_labels)

    # 设置训练器
    logger.info("设置训练器")
    trainer = setup_trainer(
        model=model,
        train_labels=train_labels,  # 传递标签数组而不是数据集
        device=device,
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )

    # 设置检查点目录
    trainer.checkpoint_dir = checkpoint_dir

    # 开始训练
    logger.info("开始训练")
    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        accumulation_steps=args.accumulation_steps,
        early_stopping=args.early_stopping
    )

    # 保存最终模型
    final_model_path = os.path.join(output_dir, 'final_model.pt')
    model.save(final_model_path)
    logger.info(f"最终模型已保存至 {final_model_path}")

    logger.info("训练完成")


# 先不用predict函数
def predict(args, logger):
    """预测模式"""
    logger.info("启动预测模式")
    # 此部分暂时不实现，当前任务仅为训练模型
    logger.info("预测功能尚未完全实现")


def main():
    """主程序入口"""
    # 解析命令行参数
    args = parse_args()

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 设置日志
    log_dir = os.path.join(args.output_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"{args.mode}_{timestamp}.log")

    logger = setup_logger(name="SDN_DDoS_Predictor", log_file=log_file)

    # 记录启动信息
    logger.info(f"SDN网络DDoS攻击预测模型 - 模式: {args.mode}")
    logger.info(f"命令行参数: {vars(args)}")

    try:
        # 根据模式执行对应功能
        if args.mode == 'train':
            train(args, logger)
        elif args.mode == 'predict':
            predict(args, logger)
        else:
            logger.error(f"不支持的模式: {args.mode}")
    except Exception as e:
        logger.exception(f"运行时错误: {str(e)}")
        raise

    logger.info("程序执行完毕")


if __name__ == "__main__":
    main()