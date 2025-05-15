#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import logging
import numpy as np
import pandas as pd
import multiprocessing as mp
from typing import Dict, List, Tuple, Union, Optional, Any
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.decomposition import PCA
from collections import defaultdict
import pickle
import warnings
import torch
from torch.utils.data import Dataset

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

class DataProcessor:
    """
    数据处理类，负责加载、清洗、特征工程及阈值式序列构建
    """

    def __init__(self,
                 data_path: str,
                 threshold_points: List[int] = [ 128, 256, 512],
                 window_size: int = 15,
                 step_size: int = 1,
                 n_workers: int = 4):
        """
        Args:
            data_path: 数据文件路径
            threshold_points: 阈值点列表，默认[4, 8, 16, 32, 64, 128]
            window_size: 滑动窗口大小(秒)
            step_size: 滑动步长(秒)
            n_workers: 并行处理的工作进程数
        """
        self.data_path = data_path
        self.threshold_points = threshold_points
        self.window_size = window_size
        self.step_size = step_size
        self.n_workers = n_workers

        # 存储各阈值点的特征提取器
        self.scalers = {}
        self.encoders = {}
        self.pca_models = {}

        # 特征列表
        base_features = [
            'Protocol',
            'Flow Duration',
            'Total Fwd Packets',
            'Total Backward Packets',
            'Flow Bytes/s',
            'Flow Packets/s',
            'Flow IAT Mean',
            'Fwd Packet Length Max',
            'Fwd Packet Length Min',
            'Fwd Packet Length Mean',
            'Fwd Packet Length Std',
            'Bwd Packet Length Max',
            'Bwd Packet Length Min',
            'Bwd Packet Length Mean',
            'Bwd Packet Length Std',
            'Packet Length Variance',
            'Flow IAT Min',
            'Flow IAT Max',
            'Flow IAT Mean',
            'Flow IAT Std',
            'Fwd IAT Mean',
            'Fwd IAT Std',
            'Fwd IAT Max',
            'Fwd IAT Min',
            'Bwd IAT Mean',
            'Bwd IAT Std',
            'Bwd IAT Max',
            'Bwd IAT Min',
            'Fwd PSH Flags',
            'Bwd PSH Flags',
            'Fwd URG Flags',
            'Bwd URG Flags',
            'Fwd Header Length',
            'Bwd Header Length',
            'Fwd Packets/s',
            'Bwd Packets/s',
            'Init_Win_bytes_forward',
            'Init_Win_bytes_backward',
            'min_seg_size_forward',
            'Subflow Fwd Bytes',
            'Subflow Bwd Bytes',
            'Average Packet Size',
            'Avg Fwd Segment Size',
            'Avg Bwd Segment Size',
            'Active Mean',
            'Active Min',
            'Active Max',
            'Active Std',
            'Idle Mean',
            'Idle Min',
            'Idle Max',
            'Idle Std'
        ]

        # 只使用带前导空格的特征名
        self.selected_features = [' ' + feature for feature in base_features]

        # 需要进行对数转换的特征
        base_log_features = [
            'Flow Bytes/s', 'Flow Packets/s', 'Fwd Packets/s', 'Bwd Packets/s',
            'Flow Duration', 'Packet Length Variance'
        ]

        self.log_transform_features = [' ' + feature for feature in base_log_features]

        self.categorical_features = [' Protocol']

    def load_data(self) -> pd.DataFrame:
        """加载CICDDoS2019数据集"""
        logger.info(f"加载数据集: {self.data_path}")

        # 支持多文件加载
        if os.path.isdir(self.data_path):
            all_files = [os.path.join(self.data_path, f) for f in os.listdir(self.data_path)
                         if f.endswith('.csv') and not f.startswith('.')]

            # 顺序加载文件（不使用多进程）
            df_list = []
            for file in all_files:
                df = self._load_single_file(file)
                if not df.empty:
                    df_list.append(df)

            # 合并所有数据
            if df_list:
                df = pd.concat(df_list, ignore_index=True)
            else:
                df = pd.DataFrame()
        else:
            df = self._load_single_file(self.data_path)

        logger.info(f"数据加载完成，共 {len(df)} 条记录")

        # 注意：不要清理列名中的空格，保留前导空格
        logger.info(f"列名示例: {list(df.columns)[:10]}...")

        return df

    def _load_single_file(self, file_path: str) -> pd.DataFrame:
        """加载单个CSV文件"""
        try:
            logger.info(f"开始读取文件: {os.path.basename(file_path)}")

            # 尝试不同的读取方式
            try:
                # 尝试作为Excel文件读取
                logger.info(f"尝试以Excel格式读取: {os.path.basename(file_path)}")
                try:
                    df = pd.read_excel(file_path, engine='openpyxl')
                    logger.info(f"成功以Excel格式读取文件")
                    return df
                except Exception as e:
                    logger.info(f"Excel读取失败，尝试CSV格式: {str(e)}")
                    pass

                # 尝试读取表头确定可用列
                header_df = pd.read_csv(file_path, nrows=5)

                # 找出可用列 - 使用带前导空格的列名
                available_cols = [col for col in header_df.columns if col in self.selected_features]

                # 添加标签列
                for label_col in [' Label', ' label', 'Label', 'label']:
                    if label_col in header_df.columns:
                        available_cols.append(label_col)
                        break

            except:
                # 如果默认引擎失败，尝试Python引擎
                logger.info(f"默认CSV引擎失败，尝试Python引擎")
                header_df = pd.read_csv(file_path, nrows=5, engine='python')

                # 找出可用列 - 使用带前导空格的列名
                available_cols = [col for col in header_df.columns if col in self.selected_features]

                # 添加标签列
                for label_col in [' Label', ' label', 'Label', 'label']:
                    if label_col in header_df.columns:
                        available_cols.append(label_col)
                        break

            logger.info(f"找到 {len(available_cols)} 个可用列: {available_cols[:10]}...")

            # 对超大文件进行采样
            sample_rate = 1.0  # 默认不采样
            if os.path.basename(file_path) in ['LDAP.csv', 'UDP.csv']:
                sample_rate = 0.1  # 只处理10%的数据
                logger.info(f"对大文件进行采样处理: {os.path.basename(file_path)}，采样率: {sample_rate}")

            # 分块读取文件
            chunk_list = []
            total_rows = 0

            # 尝试使用C引擎，出错则切换到Python引擎
            try:
                # 尝试使用C引擎，每次小批量读取
                chunks = pd.read_csv(
                    file_path,
                    usecols=available_cols,
                    chunksize=5000,  # 更小的块大小
                    on_bad_lines='skip'
                )
            except Exception as e:
                logger.warning(f"C引擎读取失败: {str(e)}，切换到Python引擎")
                # 切换到Python引擎
                chunks = pd.read_csv(
                    file_path,
                    usecols=available_cols,
                    engine='python',
                    chunksize=5000,
                    on_bad_lines='skip'
                )

            # 处理数据块
            for chunk in chunks:
                # 不清理列名中的空格，保留前导空格

                # 采样处理
                if sample_rate < 1.0:
                    chunk = chunk.sample(frac=sample_rate)

                # 处理标签
                for label_col in [' Label', ' label', 'Label', 'label']:
                    if label_col in chunk.columns:
                        chunk[label_col] = chunk[label_col].apply(
                            lambda x: 0 if str(x).lower() in ['benign', 'normal'] else 1
                        )
                        # 确保标签列名统一为' Label'（带前导空格）
                        if label_col != ' Label':
                            chunk.rename(columns={label_col: ' Label'}, inplace=True)
                        break

                chunk_list.append(chunk)
                total_rows += len(chunk)

                # 显示进度
                if total_rows % 100000 == 0:
                    logger.info(f"已读取 {total_rows} 行数据从 {os.path.basename(file_path)}")

            # 合并所有块
            if chunk_list:
                df = pd.concat(chunk_list, ignore_index=True)
                logger.info(f"文件 {os.path.basename(file_path)} 读取完成，总计 {len(df)} 行")
                return df
            else:
                logger.warning(f"文件 {file_path} 没有读取到有效数据")
                return pd.DataFrame()

        except Exception as e:
            logger.error(f"加载文件 {file_path} 时出错: {str(e)}")
            return pd.DataFrame()

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        数据清洗

        Args:
            df: 原始DataFrame

        Returns:
            清洗后的DataFrame
        """
        logger.info("开始数据清洗")

        # 1. 移除重复记录
        df_clean = df.drop_duplicates()
        logger.info(f"移除重复记录后剩余 {len(df_clean)} 条记录")

        # 2. 处理缺失值
        # 对数值特征使用中位数填充
        numeric_cols = df_clean.select_dtypes(include=['number']).columns
        df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].median())

        # 对分类特征使用众数填充
        cat_cols = df_clean.select_dtypes(include=['object']).columns
        for col in cat_cols:
            if col != ' Label':  # 不处理标签列 - 使用带前导空格的标签列名
                df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])

        # 3. 异常值处理（使用IQR方法）
        for col in numeric_cols:
            if col != ' Label':  # 不处理标签列 - 使用带前导空格的标签列名
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                # 将异常值限制在边界范围内
                df_clean[col] = df_clean[col].clip(lower_bound, upper_bound)

        logger.info("数据清洗完成")
        return df_clean

    def preprocess_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        特征预处理

        Args:
            df: 清洗后的DataFrame
            fit: 是否训练新的转换器

        Returns:
            预处理后的DataFrame
        """
        logger.info("开始特征预处理")

        df_processed = df.copy()

        # 1. 对长尾分布特征进行对数转换: X' = log(1 + X)
        for col in self.log_transform_features:
            if col in df_processed.columns:
                # 确保值为正
                min_val = df_processed[col].min()
                if min_val < 0:
                    df_processed[col] = df_processed[col] - min_val + 1
                df_processed[col] = np.log1p(df_processed[col])

        # 2. 处理类别特征（独热编码）
        for col in self.categorical_features:
            if col in df_processed.columns:
                if fit:
                    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                    encoded_data = encoder.fit_transform(df_processed[[col]])
                    self.encoders[col] = encoder
                else:
                    encoder = self.encoders.get(col)
                    if encoder is None:
                        logger.warning(f"找不到特征 {col} 的编码器，跳过处理")
                        continue
                    encoded_data = encoder.transform(df_processed[[col]])

                # 创建编码后的列
                encoded_cols = [f"{col}_{cat}" for cat in encoder.categories_[0]]
                encoded_df = pd.DataFrame(encoded_data, columns=encoded_cols, index=df_processed.index)

                # 删除原始类别列并添加编码后的列
                df_processed = df_processed.drop(col, axis=1)
                df_processed = pd.concat([df_processed, encoded_df], axis=1)

        # 3. 标准化数值特征
        numeric_cols = df_processed.select_dtypes(include=['number']).columns.tolist()

        # 排除标签列
        if ' Label' in numeric_cols:
            numeric_cols.remove(' Label')

        if numeric_cols:
            if fit:
                scaler = RobustScaler()
                df_processed[numeric_cols] = scaler.fit_transform(df_processed[numeric_cols])
                self.scalers['global'] = scaler
            else:
                scaler = self.scalers.get('global')
                if scaler is not None:
                    df_processed[numeric_cols] = scaler.transform(df_processed[numeric_cols])

        # 4. 特征降维 (根据需要应用PCA)
        # 只对预处理后的数值特征应用PCA
        if len(numeric_cols) > 10 and fit:  # 仅在特征数量足够多的情况下应用PCA
            pca = PCA(n_components=32)  # 保留32个主成分
            pca_result = pca.fit_transform(df_processed[numeric_cols])

            # 保存PCA模型
            self.pca_models['global'] = pca

            # 创建PCA特征DataFrame
            pca_df = pd.DataFrame(
                pca_result,
                columns=[f'PCA_{i}' for i in range(pca_result.shape[1])],
                index=df_processed.index
            )

            # 替换原始特征
            df_processed = df_processed.drop(numeric_cols, axis=1)
            df_processed = pd.concat([df_processed, pca_df], axis=1)

        elif len(numeric_cols) > 10 and not fit:
            pca = self.pca_models.get('global')
            if pca is not None:
                pca_result = pca.transform(df_processed[numeric_cols])

                # 创建PCA特征DataFrame
                pca_df = pd.DataFrame(
                    pca_result,
                    columns=[f'PCA_{i}' for i in range(pca_result.shape[1])],
                    index=df_processed.index
                )

                # 替换原始特征
                df_processed = df_processed.drop(numeric_cols, axis=1)
                df_processed = pd.concat([df_processed, pca_df], axis=1)

        logger.info("特征预处理完成")
        return df_processed

    def construct_threshold_sequences(self, df: pd.DataFrame) -> Dict[int, np.ndarray]:
        """
        实现阈值式序列构建

        使用规则: 当流量包数量x满足 n≤x<2n 时，使用前n个包的特征

        Args:
            df: 预处理后的DataFrame

        Returns:
            各阈值点的特征序列字典
        """
        logger.info("开始构建阈值式序列")

        # 确保数据按时间戳排序
        if ' Timestamp' in df.columns:
            df = df.sort_values(by=' Timestamp')
        elif 'Timestamp' in df.columns:
            df = df.sort_values(by='Timestamp')

        # 存储各阈值点的特征序列
        threshold_sequences = {}

        # 获取总包数列（前向+后向）- 使用带前导空格的列名
        fwd_packets_cols = [' Total Fwd Packets', ' Fwd Packets', ' Forward Packets']
        bwd_packets_cols = [' Total Backward Packets', ' Bwd Packets', ' Backward Packets']

        fwd_col = None
        for col in fwd_packets_cols:
            if col in df.columns:
                fwd_col = col
                break

        bwd_col = None
        for col in bwd_packets_cols:
            if col in df.columns:
                bwd_col = col
                break

        if fwd_col is not None and bwd_col is not None:
            logger.info(f"找到包数量列: {fwd_col} 和 {bwd_col}")
            df['TotalPackets'] = df[fwd_col] + df[bwd_col]
        else:
            # 找不到包数量列，使用备选方案
            logger.warning(f"找不到标准包数量列，尝试备选方案")

            # 尝试使用Flow Bytes/s作为替代指标
            if ' Flow Bytes/s' in df.columns:
                logger.info("使用Flow Bytes/s作为替代指标")
                df['TotalPackets'] = df[' Flow Bytes/s'] / 100  # 假设的转换因子
            elif ' Flow Packets/s' in df.columns and ' Flow Duration' in df.columns:
                logger.info("使用Flow Packets/s作为替代指标")
                df['TotalPackets'] = df[' Flow Packets/s'] * df[' Flow Duration'] / 1000000  # 估计总包数
            else:
                # 最后的方案：简单地使用均匀分割
                logger.warning("无法找到合适的包数量列，使用均匀分割数据")
                total_rows = len(df)
                df['TotalPackets'] = np.arange(1, total_rows + 1)  # 简单递增序列

        # 获取特征列和标签列
        feature_cols = df.columns.tolist()
        if ' Label' in feature_cols:
            feature_cols.remove(' Label')
        elif 'Label' in feature_cols:
            feature_cols.remove('Label')
        if ' Timestamp' in feature_cols:
            feature_cols.remove(' Timestamp')
        elif 'Timestamp' in feature_cols:
            feature_cols.remove('Timestamp')
        if 'TotalPackets' in feature_cols:
            feature_cols.remove('TotalPackets')

        # 标签列，如果存在
        label_col = ' Label' if ' Label' in df.columns else 'Label' if 'Label' in df.columns else None

        # 处理各阈值点 - 修改为串行处理而不是使用multiprocessing
        for threshold in self.threshold_points:
            result = self._process_threshold_point(df, threshold, feature_cols, label_col)
            threshold_sequences[threshold] = result

        logger.info("阈值式序列构建完成")
        return threshold_sequences

    def _process_threshold_point(
            self,
            df: pd.DataFrame,
            threshold: int,
            feature_cols: List[str],
            label_col: Optional[str]
    ) -> np.ndarray:
        """
        处理单个阈值点

        Args:
            df: DataFrame
            threshold: 阈值点
            feature_cols: 特征列
            label_col: 标签列

        Returns:
            阈值点的特征和标签
        """
        # 筛选满足条件的数据
        if threshold == self.threshold_points[-1]:  # 最大阈值点
            valid_df = df[df['TotalPackets'] >= threshold]
        else:
            next_threshold = self.threshold_points[self.threshold_points.index(threshold) + 1]
            valid_df = df[(df['TotalPackets'] >= threshold) & (df['TotalPackets'] < next_threshold)]

        if valid_df.empty:
            logger.warning(f"阈值点 {threshold} 没有匹配的数据")
            return np.array([])

        # 提取特征
        X = valid_df[feature_cols].values

        # 提取标签（如果有）
        y = None
        if label_col is not None:
            y = valid_df[label_col].values

        # 组合特征和标签
        if y is not None:
            logger.info(f"阈值点 {threshold}: 提取了 {len(X)} 条记录")
            return np.hstack((X, y.reshape(-1, 1)))
        else:
            logger.info(f"阈值点 {threshold}: 提取了 {len(X)} 条记录（无标签）")
            return X

    def create_sliding_windows(self,
                               threshold_sequences: Dict[int, np.ndarray]
                               ) -> Tuple[np.ndarray, np.ndarray]:
        """
        创建滑动窗口用于时序预测

        Args:
            threshold_sequences: 各阈值点的特征序列

        Returns:
            X_windows: 窗口特征数据
            y_windows: 窗口标签数据
        """
        logger.info(f"创建滑动窗口: 窗口大小={self.window_size}秒, 步长={self.step_size}秒")

        all_window_data = []
        all_window_labels = []

        # 对每个阈值点的序列进行处理 - 不再使用并行处理
        for threshold, data in threshold_sequences.items():
            if data.size == 0:
                logger.warning(f"阈值点 {threshold} 没有数据，跳过")
                continue

            # 划分特征和标签
            if data.shape[1] > 1:  # 有标签列
                X = data[:, :-1]
                y = data[:, -1]
            else:
                logger.warning(f"阈值点 {threshold} 的数据没有标签列")
                continue

            # 创建窗口
            x_windows, y_windows = self._create_windows_chunk(X, y, self.window_size, self.step_size, 0)

            if x_windows is not None and y_windows is not None:
                all_window_data.append(x_windows)
                all_window_labels.append(y_windows)
                logger.info(f"阈值点 {threshold}: 创建了 {len(x_windows)} 个窗口")

        # 合并所有阈值点的窗口数据
        if all_window_data and all_window_labels:
            X_windows = np.vstack(all_window_data)
            y_windows = np.concatenate(all_window_labels)

            logger.info(f"滑动窗口创建完成: {len(X_windows)} 个窗口")
            return X_windows, y_windows
        else:
            logger.warning("未能创建任何窗口")
            return np.array([]), np.array([])

    def _create_windows_chunk(self,
                              X_chunk: np.ndarray,
                              y_chunk: np.ndarray,
                              window_size: int,
                              step_size: int,
                              chunk_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        为数据块创建滑动窗口

        Args:
            X_chunk: 特征数据块
            y_chunk: 标签数据块
            window_size: 窗口大小
            step_size: 滑动步长
            chunk_idx: 数据块索引

        Returns:
            x_windows: 窗口特征
            y_windows: 窗口标签（预测目标）
        """
        if len(X_chunk) <= window_size:
            logger.warning(f"数据块 {chunk_idx} 长度 {len(X_chunk)} 小于窗口大小 {window_size}")
            return None, None

        x_windows = []
        y_windows = []

        # 创建滑动窗口
        for i in range(0, len(X_chunk) - window_size, step_size):
            # 窗口特征: 当前窗口的数据
            window = X_chunk[i:i + window_size]

            # 预测目标: 窗口后15秒是否发生攻击
            # 用窗口后续数据的最大标签值代表未来状态
            future_idx = min(i + window_size + 15, len(y_chunk) - 1)
            target = np.max(y_chunk[i + window_size:future_idx + 1]) if i + window_size <= future_idx else 0

            # 计算窗口特征统计量 (均值、标准差、最大值、最小值)
            window_mean = np.mean(window, axis=0)
            window_std = np.std(window, axis=0)
            window_max = np.max(window, axis=0)
            window_min = np.min(window, axis=0)

            # 组合窗口特征
            window_features = np.concatenate([window_mean, window_std, window_max, window_min])

            x_windows.append(window_features)
            y_windows.append(target)

        if x_windows and y_windows:
            return np.array(x_windows), np.array(y_windows)
        else:
            logger.warning(f"数据块 {chunk_idx} 未能创建窗口")
            return None, None

    def process_data_pipeline(self, train: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        完整数据处理流水线

        Args:
            train: 是否为训练模式

        Returns:
            X: 处理后的特征数据
            y: 标签数据
        """
        # 1. 加载数据
        df = self.load_data()

        # 2. 数据清洗
        df_clean = self.clean_data(df)

        # 3. 特征预处理
        df_processed = self.preprocess_features(df_clean, fit=train)

        # 4. 构建阈值式序列
        threshold_sequences = self.construct_threshold_sequences(df_processed)

        # 5. 创建滑动窗口
        X, y = self.create_sliding_windows(threshold_sequences)

        return X, y

    def save_preprocessors(self, save_path: str):
        """
        保存预处理器

        Args:
            save_path: 保存路径
        """
        preprocessors = {
            'scalers': self.scalers,
            'encoders': self.encoders,
            'pca_models': self.pca_models
        }

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as f:
            pickle.dump(preprocessors, f)

        logger.info(f"预处理器已保存至 {save_path}")

    def load_preprocessors(self, load_path: str):
        """
        加载预处理器

        Args:
            load_path: 加载路径
        """
        with open(load_path, 'rb') as f:
            preprocessors = pickle.load(f)

        self.scalers = preprocessors.get('scalers', {})
        self.encoders = preprocessors.get('encoders', {})
        self.pca_models = preprocessors.get('pca_models', {})

        logger.info(f"预处理器已从 {load_path} 加载")


class DDoSDataset(Dataset):
    """
    DDoS攻击预测的PyTorch数据集类
    """

    def __init__(self,
                 data_path: str,
                 threshold_points: List[int] = [ 128, 256, 512],
                 window_size: int = 15,
                 step_size: int = 1,
                 preprocessor_path: Optional[str] = None,
                 train: bool = True,
                 transform: Optional[Any] = None):
        """
        初始化数据集

        Args:
            data_path: 数据文件路径
            threshold_points: 阈值点列表
            window_size: 窗口大小
            step_size: 滑动步长
            preprocessor_path: 预处理器路径（用于加载已保存的预处理器）
            train: 是否为训练模式
            transform: 数据转换函数
        """
        self.transform = transform

        # 初始化处理器（单进程模式，数据集内部处理不需要并行）
        self.processor = DataProcessor(
            data_path=data_path,
            threshold_points=threshold_points,
            window_size=window_size,
            step_size=step_size,
            n_workers=1  # 单进程处理
        )

        # 如果有预处理器路径且不是训练模式，加载预处理器
        if preprocessor_path and not train:
            self.processor.load_preprocessors(preprocessor_path)

        # 处理数据
        self.features, self.labels = self.processor.process_data_pipeline(train=train)

        # 转换为PyTorch张量
        if len(self.features) > 0 and len(self.labels) > 0:
            self.features = torch.FloatTensor(self.features)
            self.labels = torch.FloatTensor(self.labels).unsqueeze(1)
        else:
            raise ValueError("处理数据失败，未能生成有效的特征和标签")

    def __len__(self):
        """返回数据集长度"""
        return len(self.features)

    def __getitem__(self, idx):
        """获取单个样本"""
        x = self.features[idx]
        y = self.labels[idx]

        if self.transform:
            x = self.transform(x)

        return x, y

    def get_preprocessor(self):
        """获取预处理器"""
        return self.processor


# 测试用例
if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # 初始化数据处理器
    data_path = "C:\\Users\\17380\\Downloads\\CSV-\\03-11"
    processor = DataProcessor(
        data_path=data_path,
        threshold_points=[ 128, 256, 512],
        window_size=15,
        step_size=1,
        n_workers=4
    )

    # 执行数据处理流水线
    X, y = processor.process_data_pipeline(train=True)

    # 打印结果
    print(f"特征数据形状: {X.shape}")
    if y is not None:
        print(f"标签数据形状: {y.shape}")
        print(f"攻击样本比例: {np.mean(y):.2%}")

    # 测试PyTorch数据集
    try:
        dataset = DDoSDataset(
            data_path=data_path,
            threshold_points=[ 128, 256, 512],
            window_size=15,
            step_size=1,
            train=True
        )
        print(f"数据集大小: {len(dataset)}")
        x, y = dataset[0]
        print(f"样本特征形状: {x.shape}")
        print(f"样本标签形状: {y.shape}")
    except Exception as e:
        print(f"数据集测试失败: {str(e)}")