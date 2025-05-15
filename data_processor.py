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
                 threshold_points: List[int] = [64, 128, 256, 512, 1024],
                 window_size: int = 15,
                 step_size: int = 1,
                 n_workers: int = 4):
        """
        Args:
            data_path: 数据文件路径
            threshold_points: 阈值点列表，默认[64, 128, 256, 512, 1024]
            window_size: 滑动窗口大小(秒)
            step_size: 滑动步长(秒)
            n_workers: 并行处理的工作进程数
        """
        self.data_path = data_path
        self.threshold_points = threshold_points
        self.window_size = window_size
        self.step_size = step_size
        self.n_workers = n_workers
        self.column_map = None  # 用于存储列名映射

        # 存储各阈值点的特征提取器
        self.scalers = {}
        self.encoders = {}
        self.pca_models = {}

        # 不带空格版本的特征列表
        self.base_features = [
            'Protocol',
            'FlowDuration',
            'TotalFwdPackets',
            'TotalBackwardPackets',
            'FlowBytes/s',
            'FlowPackets/s',
            'FwdPacketLengthMax',
            'FwdPacketLengthMin',
            'FwdPacketLengthMean',
            'FwdPacketLengthStd',
            'BwdPacketLengthMax',
            'BwdPacketLengthMin',
            'BwdPacketLengthMean',
            'BwdPacketLengthStd',
            'PacketLengthVariance',
            'FlowIATMin',
            'FlowIATMax',
            'FlowIATMean',
            'FlowIATStd',
            'FwdIATMean',
            'FwdIATStd',
            'FwdIATMax',
            'FwdIATMin',
            'BwdIATMean',
            'BwdIATStd',
            'BwdIATMax',
            'BwdIATMin',
            'FwdPSHFlags',
            'BwdPSHFlags',
            'FwdURGFlags',
            'BwdURGFlags',
            'FwdHeaderLength',
            'BwdHeaderLength',
            'FwdPackets/s',
            'BwdPackets/s',
            'Init_Win_bytes_forward',
            'Init_Win_bytes_backward',
            'min_seg_size_forward',
            'SubflowFwdBytes',
            'SubflowBwdBytes',
            'AveragePacketSize',
            'AvgFwdSegmentSize',
            'AvgBwdSegmentSize',
            'ActiveMean',
            'ActiveMin',
            'ActiveMax',
            'ActiveStd',
            'IdleMean',
            'IdleMin',
            'IdleMax',
            'IdleStd',
            'Timestamp',
        ]

        # 需要进行对数转换的特征
        self.log_transform_features_base = [
            'FlowBytes/s', 'FlowPackets/s', 'FwdPackets/s', 'BwdPackets/s',
            'FlowDuration', 'PacketLengthVariance'
        ]

        # 类别特征
        self.categorical_features_base = ['Protocol']

        # 用于查找包数量列的不同可能名称
        self.fwd_packets_cols_base = ['TotalFwdPackets', 'FwdPackets', 'ForwardPackets']
        self.bwd_packets_cols_base = ['TotalBackwardPackets', 'BwdPackets', 'BackwardPackets']

    def normalize_column_names(self, df: pd.DataFrame) -> Dict[str, str]:
        """
        创建标准化的列名映射
        将所有列名无空格版本作为键，原始列名作为值

        Args:
            df: 输入的DataFrame

        Returns:
            无空格列名到原始列名的映射字典
        """
        column_map = {}
        for col in df.columns:
            # 移除所有空格后的列名作为键
            normalized_key = col.replace(" ", "")
            column_map[normalized_key] = col

        # 打印映射信息
        logger.info(f"创建了列名映射，共 {len(column_map)} 个列")
        logger.debug(f"列名映射示例: {list(column_map.items())[:5]}...")

        return column_map

    def get_actual_column_name(self, normalized_name: str) -> Optional[str]:
        """
        根据标准化名称获取数据集中的实际列名

        Args:
            normalized_name

        Returns:
            实际列名，如果不存在则返回None
        """
        if self.column_map is None:
            logger.warning("列名映射尚未初始化")
            return None

        return self.column_map.get(normalized_name)

    def _load_single_file(self, file_path: str) -> pd.DataFrame:
        """加载单个 CSV 文件，只读取base_features中的列和标签列"""
        try:
            logger.info(f"开始读取文件: {os.path.basename(file_path)}")

            # 先读取文件头，获取列名
            try:
                header_df = pd.read_csv(file_path, nrows=0)
            except Exception as e:
                logger.warning(f"默认引擎读取头部失败: {e}，切换 Python 引擎")
                header_df = pd.read_csv(file_path, nrows=0, engine='python')

            # 先创建临时列名映射，用于识别需要的列
            temp_column_map = {}
            for col in header_df.columns:
                normalized_key = col.replace(" ", "")
                temp_column_map[normalized_key] = col

            # 确定要读取的列名
            usecols = []

            # 添加特征列
            for base_col in self.base_features:
                if base_col in temp_column_map:
                    usecols.append(temp_column_map[base_col])

            # 添加标签列
            label_col = None
            for label_name in ['Label', 'label']:
                if label_name in temp_column_map:
                    label_col = temp_column_map[label_name]
                    usecols.append(label_col)
                    break

            if not label_col:
                # 尝试其他方式找标签列
                for col in header_df.columns:
                    if 'label' in col.lower():
                        usecols.append(col)
                        logger.info(f"使用替代标签列: {col}")
                        break

            if not usecols:
                logger.error(f"无法识别需要读取的列")
                return pd.DataFrame()

            logger.info(f"将读取 {len(usecols)} 列: {len(usecols) - 1} 个特征列和 1 个标签列")

            # 设置采样率
            sample_rate = 0.05 if os.path.basename(file_path) in ['LDAP.csv', 'UDP.csv', 'Syn.csv', 'UDPLag.csv', 'MSSQL.csv', 'NetBIOS.csv', 'Portmap.csv'] else 1.0
            if sample_rate < 1.0:
                logger.info(f"对大文件采样 {sample_rate}")

            # 分块读取，只读需要的列
            chunks = None
            try:
                chunks = pd.read_csv(file_path, chunksize=1000, usecols=usecols, on_bad_lines='skip')
            except Exception as e:
                logger.warning(f"C 引擎读取失败: {e}，切换 Python 引擎")
                chunks = pd.read_csv(file_path, engine='python', chunksize=1000, usecols=usecols, on_bad_lines='skip')

            chunk_list = []
            for chunk in chunks:
                if sample_rate < 1.0:
                    chunk = chunk.sample(frac=sample_rate)
                chunk_list.append(chunk)

            if not chunk_list:
                return pd.DataFrame()

            df = pd.concat(chunk_list, ignore_index=False)
            logger.info(f"文件 {os.path.basename(file_path)} 读取完成，shape={df.shape}")

            # 确认列数
            expected_col_count = len(self.base_features) + 1
            if df.shape[1] != expected_col_count:
                logger.warning(f"列数与预期不符: 得到 {df.shape[1]}，预期 {expected_col_count}")
                logger.info(f"当前 DataFrame 列名列表: {df.columns.tolist()}")
            return df

        except Exception as e:
            logger.error(f"加载文件 {file_path} 时出错: {e}")
            return pd.DataFrame()

    def load_data(self) -> pd.DataFrame:
        """加载CICDDoS2019数据集，保证按行拼接并只读取所需的列"""
        logger.info(f"加载数据集: {self.data_path}")

        df_list = []
        if os.path.isdir(self.data_path):
            all_files = [
                os.path.join(self.data_path, f)
                for f in os.listdir(self.data_path)
                if f.endswith('.csv') and not f.startswith('.')
            ]
            for file in all_files:
                df_part = self._load_single_file(file)
                if not df_part.empty:
                    # 打印每块的 shape，方便调试
                    logger.debug(f"{os.path.basename(file)} shape = {df_part.shape}")
                    df_list.append(df_part)
        else:
            df_list.append(self._load_single_file(self.data_path))

        if not df_list:
            return pd.DataFrame()

        # 确保按行拼接（axis=0）
        df = pd.concat(df_list, ignore_index=True)

        # 拼完再检查：行数一定要大于列数，否则就是搞反了
        if df.shape[0] < df.shape[1]:
            logger.error(
                f"拼接后行数 ({df.shape[0]}) 少于列数 ({df.shape[1]})，请检查单个文件是否读成了转置矩阵"
            )
            raise ValueError("数据行列方向错误，已停止执行")

        # 创建列名映射
        self.column_map = self.normalize_column_names(df)

        logger.info(f"数据加载完成，最终shape: {df.shape}")
        return df

    def dropna_in_chunks(self, df, chunk_size=1_000_00):
        chunks = []
        for i in range(0, len(df), chunk_size):
            chunk = df.iloc[i:i + chunk_size].dropna()
            chunks.append(chunk)
        return pd.concat(chunks, ignore_index=True)

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
        logger.info(f"移除重复记录后剩余 {len(df_clean)} 条记录,df_clean.shape={df_clean.shape}")
        after_dedup = len(df_clean)

        # 获取标签列名
        label_col = self.get_actual_column_name('Label')

        # 2. 处理缺失值
        df_clean = self.dropna_in_chunks(df_clean)
        logger.info(f"删除缺失值后剩余 {len(df_clean)} 条记录 (删除了 {after_dedup - len(df_clean)} 条)")

        # 对分类特征使用众数填充
        numeric_cols = df_clean.select_dtypes(include=['number']).columns
        cat_cols = df_clean.select_dtypes(include=['object']).columns
        for col in cat_cols:
            if col != label_col:  # 不处理标签列
                df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])

        # 3. 异常值处理（使用IQR方法）
        for col in numeric_cols:
            if col != label_col:  # 不处理标签列
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
        for base_col in self.log_transform_features_base:
            actual_col = self.get_actual_column_name(base_col)
            if actual_col and actual_col in df_processed.columns:
                min_val = df_processed[actual_col].min()
                if min_val < 0:
                    df_processed[actual_col] = df_processed[actual_col] - min_val + 1
                df_processed[actual_col] = np.log1p(df_processed[actual_col])

        # 2. 处理类别特征（独热编码）
        for base_col in self.categorical_features_base:
            actual_col = self.get_actual_column_name(base_col)
            if actual_col and actual_col in df_processed.columns:

                # --------- ⛑ 防止重复 one-hot 编码 ⛑ ---------
                already_encoded = any(
                    col.startswith(f"{base_col}_") for col in df_processed.columns
                )
                if already_encoded:
                    logger.info(f"检测到特征 {base_col} 已经完成独热编码，跳过")
                    continue
                # -----------------------------------------

                if fit:
                    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                    encoded_data = encoder.fit_transform(df_processed[[actual_col]])
                    self.encoders[base_col] = encoder
                else:
                    encoder = self.encoders.get(base_col)
                    if encoder is None:
                        logger.warning(f"找不到特征 {base_col} 的编码器，跳过处理")
                        continue
                    encoded_data = encoder.transform(df_processed[[actual_col]])

                encoded_cols = [f"{base_col}_{cat}" for cat in encoder.categories_[0]]
                encoded_df = pd.DataFrame(encoded_data, columns=encoded_cols, index=df_processed.index)

                df_processed = df_processed.drop(actual_col, axis=1)
                df_processed = pd.concat([df_processed, encoded_df], axis=1)

        # 3. 标准化数值特征
        numeric_cols = df_processed.select_dtypes(include=['number']).columns.tolist()
        label_col = self.get_actual_column_name('Label')
        if label_col and label_col in numeric_cols:
            numeric_cols.remove(label_col)

        if numeric_cols:
            if fit:
                scaler = RobustScaler()
                df_processed[numeric_cols] = scaler.fit_transform(df_processed[numeric_cols])
                self.scalers['global'] = scaler
            else:
                scaler = self.scalers.get('global')
                if scaler is not None:
                    df_processed[numeric_cols] = scaler.transform(df_processed[numeric_cols])

        logger.info("特征预处理完成")
        logger.info(f"预处理后特征列数量: {df_processed.shape[1]}")
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
        timestamp_col = self.get_actual_column_name('Timestamp')
        if timestamp_col and timestamp_col in df.columns:
            df = df.sort_values(by=timestamp_col)
            print(f"successful sort time stamp!")

        # 存储各阈值点的特征序列
        threshold_sequences = {}

        # 查找前向包数列和后向包数列
        fwd_col = None
        for base_col in self.fwd_packets_cols_base:
            actual_col = self.get_actual_column_name(base_col)
            if actual_col and actual_col in df.columns:
                fwd_col = actual_col
                print(f"fwd_col: {fwd_col}")
                break

        bwd_col = None
        for base_col in self.bwd_packets_cols_base:
            actual_col = self.get_actual_column_name(base_col)
            if actual_col and actual_col in df.columns:
                bwd_col = actual_col
                print(f"bwd_col: {bwd_col}")
                break

        if fwd_col is not None and bwd_col is not None:
            logger.info(f"找到包数量列: {fwd_col} 和 {bwd_col}")
            df['TotalPackets'] = df[fwd_col] + df[bwd_col]
        else:
            # 找不到包数量列，使用备选方案
            logger.warning(f"找不到标准包数量列，尝试备选方案")

            # 尝试使用Flow Bytes/s作为替代指标
            flow_bytes_col = self.get_actual_column_name('FlowBytes/s')
            flow_duration_col = self.get_actual_column_name('FlowDuration')
            flow_packets_col = self.get_actual_column_name('FlowPackets/s')

            if flow_bytes_col and flow_bytes_col in df.columns:
                logger.info("使用Flow Bytes/s作为替代指标")
                df['TotalPackets'] = df[flow_bytes_col] / 100  # 假设的转换因子
            elif flow_packets_col and flow_duration_col and flow_packets_col in df.columns and flow_duration_col in df.columns:
                logger.info("使用Flow Packets/s作为替代指标")
                df['TotalPackets'] = df[flow_packets_col] * df[flow_duration_col] / 1000000  # 估计总包数
            else:
                # 最后的方案：简单地使用均匀分割
                logger.warning("无法找到合适的包数量列，使用均匀分割数据")
                total_rows = len(df)
                df['TotalPackets'] = np.arange(1, total_rows + 1)  # 简单递增序列

        # 获取特征列和标签列
        feature_cols = df.columns.tolist()
        label_col = self.get_actual_column_name('Label')
        if label_col and label_col in feature_cols:
            feature_cols.remove(label_col)
        if timestamp_col and timestamp_col in feature_cols:
            feature_cols.remove(timestamp_col)
        if 'TotalPackets' in feature_cols:
            feature_cols.remove('TotalPackets')

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
        if label_col is not None and label_col in valid_df.columns:
            y = valid_df[label_col].values

        # 组合特征和标签
        if y is not None:
            logger.info(f"阈值点 {threshold}: 提取了 {len(X)} 条记录数据,shape: {X.shape}")
            return np.hstack((X, y.reshape(-1, 1)))
        else:
            logger.info(f"阈值点 {threshold}: 提取了 {len(X)} 条记录（无标签）,shape: {X.shape}")
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
                 threshold_points: List[int] = [64, 128, 256, 512, 1024],
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
        threshold_points=[64, 128, 256, 512, 1024],
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
            threshold_points=[64, 128, 256, 512, 1024],
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