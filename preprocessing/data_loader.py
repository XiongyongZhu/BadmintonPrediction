# -*- coding: utf-8 -*-
"""
Created on Tue Aug 26 14:32:41 2025

@author: dragon
"""

import pandas as pd
import logging

logger = logging.getLogger(__name__)

class BadmintonDataLoader:
    def load_data(self, file_path='data/raw/badminton_dataset.csv'):
        """加载数据，并确保日期列存在和排序"""
        try:
            data = pd.read_csv(file_path)
            # 确保日期列存在
            if 'date' not in data.columns:
                logger.warning("数据中没有日期列，将使用默认索引排序")
            else:
                # 将日期转换为datetime类型
                data['date'] = pd.to_datetime(data['date'])
                # 按日期排序
                data.sort_values('date', inplace=True)
            return data
        except Exception as e:
            logger.error(f"数据加载失败: {e}")
            # 返回空DataFrame
            return pd.DataFrame()
