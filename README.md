# 基于LSTM的SDN网络DDoS攻击预测模型

## 项目概述

本项目实现了一个先进的基于LSTM和自注意力机制的SDN网络DDoS攻击预测模型，能够预测网络节点（特别是spine1节点）未来15秒内遭受DDoS攻击的概率。该模型在SDN服务链的h5a节点上训练和部署，使用CICDDoS2019数据集进行模型训练和验证。

该预测模型为网络管理员提供先发制人的防御能力，在攻击实际发生前采取预防措施，显著降低DDoS攻击对网络性能和服务可用性的影响。

## 核心特性

- **双向LSTM架构**：使用层次化双向LSTM网络结合自注意力机制，提高对时序数据的建模能力
- **阈值式序列构建**：创新的阈值式序列处理方法，高效处理网络流量数据
- **自适应动态阈值**：根据网络负载和风险等级动态调整决策阈值
- **SDN控制器集成**：与SDN控制器实时交互，实现攻击预警和防御策略部署
- **综合加权概率模型**：考虑流量比重、服务重要性和异常程度的节点攻击概率计算框架

## 技术栈

- Python 3.10
- PyTorch 2.1.0
- CUDA 12.1 (GPU加速)
- pandas, numpy, scikit-learn, matplotlib等核心库

## 项目结构

```
├── README.md                # 项目文档与使用说明
├── requirements.txt         # 项目依赖列表
├── main.py                  # 主程序入口，命令行参数解析
├── data_processor.py        # 数据加载、预处理与特征工程
├── lstm_model.py            # LSTM网络架构与自注意力机制实现
├── loss_function.py         # 自定义损失函数实现
├── trainer.py               # 模型训练流程与优化策略
├── predictor.py             # 模型推理与SDN控制器集成
├── utils.py                 # 工具函数集合
├── train.sh                 # 训练启动脚本与参数设置
└── predict.sh               # 预测部署脚本
```

## 安装指南

### 环境要求

- Python 3.10
- CUDA 12.1 (可选，但推荐用于加速训练)
- Git

### 安装步骤

1. 克隆项目仓库
   ```bash
   git clone https://github.com/your-username/sdn-ddos-predictor.git
   cd sdn-ddos-predictor
   ```

2. 创建虚拟环境（推荐）
   ```bash
   conda create -n sdn_predictor python=3.10
   conda activate sdn_predictor
   ```

3. 安装依赖
   ```bash
   pip install -r requirements.txt
   ```

4. 安装PyTorch（根据CUDA版本选择正确的安装命令）
   ```bash
   # 对于CUDA 12.1
   pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
   ```

## 使用指南

### 数据准备

1. 下载CICDDoS2019数据集
2. 将数据放置于`data/train`目录下用于训练，`data/test`目录下用于测试

### 模型训练

使用提供的训练脚本：

```bash
chmod +x train.sh
./train.sh
```

或者手动执行：

```bash
python main.py --mode train --data_path data/train --output_dir output --epochs 50 --batch_size 64
```

### 攻击预测

使用提供的预测脚本：

```bash
chmod +x predict.sh
./predict.sh
```

或者手动执行：

```bash
python main.py --mode predict --data_path data/test --model_path output/train_xxxxx/final_model.pt --threshold 0.5
```

### 实时预测服务

```python
from predictor import DDoSPredictor

predictor = DDoSPredictor(
    model_path='output/train_xxxxx/final_model.pt',
    threshold=0.5,
    controller_url='http://sdn-controller:8181/onos/v1'
)

# 启动实时预测
predictor.predict_realtime(
    node_id='spine1',
    interval=60,  # 60秒预测间隔
    notify_controller=True
)
```

## 模型架构

### 双向LSTM架构

模型使用层次化双向LSTM结合自注意力机制，能够有效捕获时序特征。主要组件包括：

1. **特征嵌入层**：将原始特征映射到低维空间
2. **位置编码**：为序列添加位置信息
3. **双向LSTM层**：捕获双向时间依赖
4. **自注意力层**：关注重要特征和时间点
5. **特征集成层**：整合学习到的表示
6. **输出层**：预测攻击概率

### 阈值式序列构建

采用基于阈值的序列构建方法处理网络流量数据：

- 阈值序列为：[4, 8, 16, 32, 64, 128]
- 当流量包数量x满足`n≤x<2n`时，使用前n个包的特征代表整个流

### 节点攻击概率计算

节点攻击概率计算公式：
```
P(节点被攻击) = 1 - exp(-∑(Wi * P(流i被攻击)))
```

其中Wi为综合权重，考虑以下三个因素：
1. 流量比重(50%)
2. 服务重要性(30%)
3. 异常程度(20%)

## 评估指标

- **ROC-AUC**：评估模型区分正常流量和攻击流量的能力
- **PR-AUC**：在类别不平衡情况下评估模型性能
- **F1分数**：综合评估精确率和召回率
- **Brier分数**：评估概率预测的准确性

## 贡献指南

欢迎对本项目提供贡献和改进！请遵循以下步骤：

1. Fork仓库
2. 创建功能分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'Add some amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 提交Pull Request

## 许可证

本项目采用 MIT 许可证 - 详情请见 [LICENSE](LICENSE) 文件

## 致谢

- CICDDoS2019数据集提供者
- PyTorch和相关开源项目的贡献者们