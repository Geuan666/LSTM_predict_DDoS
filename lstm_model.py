#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


class PositionalEncoding(nn.Module):
    """
    位置编码模块，用于向输入添加位置信息
    """

    def __init__(self, d_model: int, max_len: int = 10):
        """
        初始化位置编码

        Args:
            d_model: 嵌入维度
            max_len: 最大序列长度
        """
        super().__init__()

        # 创建位置编码矩阵
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        添加位置编码到输入

        Args:
            x: 输入张量，形状 [batch_size, seq_len, d_model]

        Returns:
            添加位置编码后的张量
        """
        seq_len = x.size(1)
        x = x + self.pe[:seq_len, :]
        return x


class MultiHeadAttention(nn.Module):
    """
    多头自注意力机制
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        """
        初始化多头注意力

        Args:
            d_model: 模型维度
            num_heads: 注意力头数
            dropout: Dropout率
        """
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # 线性投影层
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.output_linear = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

        # 使用Xavier初始化
        nn.init.xavier_uniform_(self.q_linear.weight)
        nn.init.xavier_uniform_(self.k_linear.weight)
        nn.init.xavier_uniform_(self.v_linear.weight)
        nn.init.xavier_uniform_(self.output_linear.weight)

    def scaled_dot_product_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                                     mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        计算缩放点积注意力

        Args:
            q: 查询张量 [batch_size, num_heads, seq_len, d_k]
            k: 键张量 [batch_size, num_heads, seq_len, d_k]
            v: 值张量 [batch_size, num_heads, seq_len, d_k]
            mask: 可选掩码张量

        Returns:
            注意力加权后的值张量
        """
        # 计算注意力分数
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)

        # 应用掩码（如果有）
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # 计算注意力权重
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 应用注意力权重
        output = torch.matmul(attn_weights, v)

        return output

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播

        Args:
            q: 查询张量 [batch_size, seq_len, d_model]
            k: 键张量 [batch_size, seq_len, d_model]
            v: 值张量 [batch_size, seq_len, d_model]
            mask: 可选掩码张量

        Returns:
            注意力加权后的输出张量
        """
        batch_size = q.size(0)

        # 线性投影并重塑为多头形式
        q = self.q_linear(q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.k_linear(k).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.v_linear(v).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # 应用缩放点积注意力
        attn_output = self.scaled_dot_product_attention(q, k, v, mask)

        # 重塑回原始维度
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        # 最终线性变换
        output = self.output_linear(attn_output)

        return output


class DDoSPredictionLSTM(nn.Module):
    """
    基于双向LSTM和自注意力机制的DDoS攻击预测模型
    """

    def __init__(self,
                 input_dim: int = 128,  # PCA后维度(32)×4种统计特征
                 embedding_dim: int = 32,
                 hidden_dim: int = 64,
                 num_layers: int = 2,
                 dropout: float = 0.3,
                 attention_heads: int = 4,
                 attention_dim: int = 64,
                 threshold_points: int = 6):
        """
        初始化模型架构

        Args:
            input_dim: 输入特征维度
            embedding_dim: 嵌入维度
            hidden_dim: LSTM隐藏层维度
            num_layers: LSTM层数
            dropout: Dropout率
            attention_heads: 注意力头数
            attention_dim: 注意力层维度
            threshold_points: 阈值点数量（最大序列长度）
        """
        super().__init__()
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.attention_dim = attention_dim
        self.threshold_points = threshold_points

        logger.info(f"初始化DDoSPredictionLSTM: 输入维度={input_dim}, 嵌入维度={embedding_dim}, "
                    f"隐藏层维度={hidden_dim}, LSTM层数={num_layers}")

        # 特征嵌入层
        self.embedding = nn.Sequential(
            nn.Linear(input_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU()
        )

        # 位置编码
        self.positional_encoding = PositionalEncoding(embedding_dim, max_len=max(10, threshold_points))

        # 双向LSTM层
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # 自注意力层
        self.attention = MultiHeadAttention(
            d_model=hidden_dim * 2,  # 双向LSTM输出维度
            num_heads=attention_heads,
            dropout=dropout
        )

        # 特征集成层
        self.feature_integration = nn.Sequential(
            nn.Linear(hidden_dim * 2, attention_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(attention_dim, attention_dim // 2),
            nn.ReLU()
        )

        # 输出层
        self.output_layer = nn.Linear(attention_dim // 2, 1)

        # 使用Xavier初始化
        self._init_weights()

        logger.info("DDoSPredictionLSTM模型初始化完成")

    def _init_weights(self):
        """初始化模型权重"""
        for name, param in self.named_parameters():
            if 'weight' in name and 'lstm' not in name:  # LSTM有自己的初始化
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

        # 为LSTM层特别初始化
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入特征张量，形状 [batch_size, threshold_points, input_dim]

        Returns:
            攻击概率预测，形状 [batch_size, 1]
        """
        batch_size, seq_len, _ = x.shape

        # 应用特征嵌入
        x_reshaped = x.contiguous().view(batch_size * seq_len, -1)
        embedded = self.embedding[0](x_reshaped)
        embedded = embedded.view(batch_size, seq_len, -1)

        # 应用批归一化（在序列维度上）
        embedded = embedded.transpose(1, 2)  # [batch_size, embedding_dim, seq_len]
        embedded = self.embedding[1](embedded)
        embedded = embedded.transpose(1, 2)  # [batch_size, seq_len, embedding_dim]

        # 应用ReLU激活
        embedded = self.embedding[2](embedded)

        # 添加位置编码
        embedded = self.positional_encoding(embedded)

        # 通过LSTM层
        lstm_out, _ = self.lstm(embedded)  # [batch_size, seq_len, hidden_dim*2]

        # 应用自注意力
        attn_out = self.attention(lstm_out, lstm_out, lstm_out)  # [batch_size, seq_len, hidden_dim*2]

        # 全局池化 - 使用平均池化汇总序列
        global_repr = attn_out.mean(dim=1)  # [batch_size, hidden_dim*2]

        # 特征集成
        integrated = self.feature_integration(global_repr)  # [batch_size, attention_dim//2]

        # 输出层
        logits = self.output_layer(integrated)  # [batch_size, 1]

        # 使用Sigmoid激活函数得到概率输出
        prob = torch.sigmoid(logits)

        return prob

    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        预测函数

        Args:
            x: 输入特征张量
            threshold: 决策阈值

        Returns:
            预测概率和二元预测结果
        """
        self.eval()
        with torch.no_grad():
            probs = self.forward(x)
            predictions = (probs >= threshold).float()
        return probs, predictions

    def save(self, path: str):
        """
        保存模型

        Args:
            path: 保存路径
        """
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': {
                'input_dim': self.input_dim,
                'embedding_dim': self.embedding_dim,
                'hidden_dim': self.hidden_dim,
                'attention_dim': self.attention_dim,
                'threshold_points': self.threshold_points
            }
        }, path)
        logger.info(f"模型已保存至 {path}")

    @classmethod
    def load(cls, path: str, device: torch.device = None):
        """
        加载模型

        Args:
            path: 模型路径
            device: 设备(CPU/GPU)

        Returns:
            加载的模型
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        checkpoint = torch.load(path, map_location=device)
        config = checkpoint['config']

        model = cls(
            input_dim=config['input_dim'],
            embedding_dim=config['embedding_dim'],
            hidden_dim=config['hidden_dim'],
            attention_dim=config['attention_dim'],
            threshold_points=config['threshold_points']
        )

        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        logger.info(f"模型已从 {path} 加载")

        return model


if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # 测试模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")

    # 创建模型
    model = DDoSPredictionLSTM().to(device)
    logger.info(f"模型参数总数: {sum(p.numel() for p in model.parameters())}")

    # 测试前向传播
    batch_size = 4
    seq_len = 6  # 阈值点数量
    input_dim = 128  # PCA后维度(32)×4种统计特征

    # 随机生成测试数据
    test_input = torch.randn(batch_size, seq_len, input_dim).to(device)

    # 前向传播
    output = model(test_input)

    logger.info(f"输入形状: {test_input.shape}")
    logger.info(f"输出形状: {output.shape}")
    logger.info(f"输出样例: {output}")