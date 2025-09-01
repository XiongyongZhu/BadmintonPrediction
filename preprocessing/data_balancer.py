# -*- coding: utf-8 -*-
"""
Created on Mon Aug 18 11:47:15 2025
增强数据平衡处理
@author: Dragon
"""

from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks
import numpy as np
import logging

logger = logging.getLogger(__name__)

class DataBalancer:
    def __init__(self, sampler_type='smotetomek', sampling_strategy='auto', random_state=42):
        """
        初始化数据平衡器
        
        参数:
            sampler_type: 采样器类型 ('smotetomek', 'smote', 'tomek')
            sampling_strategy: 采样策略。 
                'auto': 目标是将所有类别采样到最多类的数量（过采样）或最少类的数量（欠采样）。
                'majority': 只重采样多数类（用于欠采样）。
                'not minority': 重采样除少数类以外的所有类（用于欠采样）。
                float: 指定少数类与多数类的目标比例。例如，0.5 意味着少数类的样本数将达到多数类的50%。
                dict: 指定每个类别期望的样本数。例如 {0: 300, 1: 300}。
            random_state: 随机种子
        """
        self.sampler_type = sampler_type
        self.sampling_strategy = sampling_strategy
        self.random_state = random_state
        self.sampler = None
        
        # 根据类型初始化采样器
        if sampler_type == 'smotetomek':
            self.sampler = SMOTETomek(
                sampling_strategy=sampling_strategy,
                random_state=random_state
            )
        elif sampler_type == 'smote':
            self.sampler = SMOTE(
                sampling_strategy=sampling_strategy,
                random_state=random_state
            )
        elif sampler_type == 'tomek':
            self.sampler = TomekLinks(sampling_strategy=sampling_strategy)
        else:
            raise ValueError(f"不支持的采样器类型: {sampler_type}。可选: 'smotetomek', 'smote', 'tomek'")
    
    def balance(self, X, y):
        """处理类别不平衡"""
        try:
            X_res, y_res = self.sampler.fit_resample(X, y)
            original_ratio = np.sum(y == 1) / np.sum(y == 0)
            new_ratio = np.sum(y_res == 1) / np.sum(y_res == 0)
            logger.info(f"数据平衡完成: 原始形状 {X.shape} (比例 {original_ratio:.2f}), 平衡后形状 {X_res.shape} (比例 {new_ratio:.2f})")
            return X_res, y_res
        except Exception as e:
            logger.error(f"数据平衡失败: {str(e)}")
            raise
        
    def calculate_imbalance_ratio(self, y):
        """计算类别不平衡比例 (负样本数 / 正样本数)，用于 scale_pos_weight"""
        y = np.array(y)
        count_negative = np.sum(y == 0)
        count_positive = np.sum(y == 1)
    
        # 避免除零错误
        if count_positive == 0:
            return 1.0
        imbalance_ratio = count_negative / count_positive
        logger.info(f"计算类别不平衡比例: {count_negative} / {count_positive} = {imbalance_ratio:.4f}")
        return imbalance_ratio

    def get_class_distribution(self, y):
        """获取类别分布信息，用于辅助设置 sampling_strategy"""
        unique, counts = np.unique(y, return_counts=True)
        return dict(zip(unique, counts))

