#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import random
import logging
import numpy as np
import torch
import json
from typing import Dict, List, Optional, Union, Any
import math
from datetime import datetime


def setup_logger(name: str, log_file: Optional[str] = None, level: int = logging.INFO) -> logging.Logger:
    """
    设置日志记录器

    Args:
        name: 日志记录器名称
        log_file: 日志文件路径(可选)
        level: 日志级别

    Returns:
        配置好的日志记录器
    """
    # 创建日志记录器
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # 避免重复添加处理器
    if logger.handlers:
        return logger

    # 创建格式化器
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 如果指定了日志文件，创建文件处理器
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def set_seed(seed: int = 42):
    """
    设置随机种子以保证实验可重现

    Args:
        seed: 随机种子
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # 设置确定性算法
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def compute_node_attack_probability(
        flow_probs: np.ndarray,
        flow_rates: np.ndarray,
        service_importance: np.ndarray,
        baseline_threshold: float = 0.5,
        load_ratio: float = 1.0,
        delta: float = 0.15,
        risk_level: str = 'normal'
) -> float:
    """
    计算节点攻击概率

    使用加权流量聚合公式：P(节点被攻击) = 1 - exp(-∑(Wi * P(流i被攻击)))

    Args:
        flow_probs: 各流量被攻击的概率数组
        flow_rates: 各流量的相对速率数组
        service_importance: 服务重要性系数数组
        baseline_threshold: 基线阈值，默认0.5
        load_ratio: 当前负载与平均负载的比值，默认1.0
        delta: 调整因子，默认0.15
        risk_level: 风险级别，可选值: 'low', 'normal', 'high'

    Returns:
        节点被攻击的概率
    """
    # 确保输入数组长度一致
    n_flows = len(flow_probs)
    assert len(flow_rates) == n_flows, "流量速率数组长度必须与概率数组长度一致"
    assert len(service_importance) == n_flows, "服务重要性数组长度必须与概率数组长度一致"

    # 动态调整基线阈值
    # 基线阈值 = 0.5 + δ * (当前网络负载 / 平均网络负载 - 1)
    adjusted_threshold = 0.5 + delta * (load_ratio - 1)

    # 根据风险级别进一步调整阈值
    if risk_level == 'low':
        adjusted_threshold += 0.1
    elif risk_level == 'high':
        adjusted_threshold -= 0.1

    # 确保阈值在有效范围内
    baseline_threshold = max(0.3, min(0.7, adjusted_threshold))

    # 初始化权重数组
    weights = np.zeros(n_flows)

    # 1. 计算流量比重权重 (50%)
    if np.sum(flow_rates) > 0:
        traffic_weights = 0.5 * (flow_rates / np.sum(flow_rates))
    else:
        traffic_weights = np.zeros(n_flows)

    # 2. 计算重要性权重 (30%)
    importance_weights = 0.3 * service_importance

    # 3. 计算异常程度权重 (20%)
    anomaly_weights = np.zeros(n_flows)
    for i in range(n_flows):
        if flow_probs[i] > baseline_threshold:
            anomaly_weights[i] = 0.2 * (flow_probs[i] - baseline_threshold) / (1 - baseline_threshold)

    # 计算综合权重
    weights = traffic_weights + importance_weights + anomaly_weights

    # 计算节点攻击概率
    # P(节点被攻击) = 1 - exp(-∑(Wi * P(流i被攻击)))
    weighted_sum = np.sum(weights * flow_probs)
    node_attack_prob = 1 - math.exp(-weighted_sum)

    return node_attack_prob


def send_to_sdn_controller(
        node_id: str,
        attack_probability: float,
        predictions: np.ndarray,
        flow_ids: List[str],
        controller_url: str,
        api_key: Optional[str] = None
) -> Dict:
    """
    向SDN控制器发送攻击预测结果

    Args:
        node_id: 节点ID
        attack_probability: 节点攻击概率
        predictions: 各流的攻击预测结果
        flow_ids: 流ID列表
        controller_url: SDN控制器URL
        api_key: API密钥(可选)

    Returns:
        控制器响应
    """
    # 构建请求数据
    payload = {
        'node_id': node_id,
        'timestamp': datetime.now().isoformat(),
        'attack_probability': float(attack_probability),
        'flow_predictions': [
            {'flow_id': flow_id, 'is_attack': bool(pred)}
            for flow_id, pred in zip(flow_ids, predictions)
        ]
    }

    # 这里应该实现实际的API调用逻辑
    # 由于这只是接口定义，我们只返回构建的payload
    # 实际实现中，应该使用requests等库向控制器发送HTTP请求

    logger = logging.getLogger("SDN_DDoS_Predictor")
    logger.info(f"向SDN控制器 {controller_url} 发送预测结果: 节点={node_id}, 攻击概率={attack_probability:.4f}")

    # 这里应返回实际的响应，现在仅返回构建的payload用于示例
    return {'status': 'success', 'sent_payload': payload}


def read_sdn_flow_metrics(
        node_id: str,
        controller_url: str,
        api_key: Optional[str] = None
) -> Dict[str, Any]:
    """
    从SDN控制器读取流量指标

    Args:
        node_id: 节点ID
        controller_url: SDN控制器URL
        api_key: API密钥(可选)

    Returns:
        流量指标数据
    """
    # 这里应该实现实际的API调用逻辑
    # 由于这只是接口定义，我们只返回模拟数据

    logger = logging.getLogger("SDN_DDoS_Predictor")
    logger.info(f"从SDN控制器 {controller_url} 读取节点 {node_id} 的流量指标")

    # 返回模拟数据
    return {
        'node_id': node_id,
        'timestamp': datetime.now().isoformat(),
        'current_load': 100.0,  # 当前负载
        'average_load': 80.0,  # 平均负载
        'flows': [
            {'flow_id': f'flow_{i}', 'rate': 10.0, 'importance': 1.0}
            for i in range(10)
        ]
    }


def get_dynamic_baseline_threshold(
        current_load: float,
        avg_load: float,
        delta: float = 0.15,
        risk_level: str = 'normal'
) -> float:
    """
    计算动态基线阈值

    Args:
        current_load: 当前网络负载
        avg_load: 平均网络负载
        delta: 调整因子，默认0.15
        risk_level: 风险级别，可选值: 'low', 'normal', 'high'

    Returns:
        动态基线阈值
    """
    # 基线阈值计算公式
    if avg_load > 0:
        baseline = 0.5 + delta * (current_load / avg_load - 1)
    else:
        baseline = 0.5

    # 根据风险级别调整
    if risk_level == 'low':
        baseline += 0.1
    elif risk_level == 'high':
        baseline -= 0.1

    # 限制阈值范围
    baseline = max(0.3, min(0.7, baseline))

    return baseline


# 如果作为主程序运行，进行简单测试
if __name__ == "__main__":
    # 设置日志
    logger = setup_logger("utils_test")

    # 测试节点攻击概率计算
    n_flows = 5
    flow_probs = np.random.rand(n_flows)
    flow_rates = np.ones(n_flows) / n_flows
    service_importance = np.ones(n_flows)

    node_prob = compute_node_attack_probability(
        flow_probs, flow_rates, service_importance,
        load_ratio=1.2, risk_level='normal'
    )

    logger.info(f"流攻击概率: {flow_probs}")
    logger.info(f"节点攻击概率: {node_prob:.4f}")

    # 测试动态阈值
    threshold = get_dynamic_baseline_threshold(120, 100, risk_level='normal')
    logger.info(f"动态基线阈值: {threshold:.4f}")