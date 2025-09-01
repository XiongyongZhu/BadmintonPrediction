# -*- coding: utf-8 -*-
"""
Created on Fri Aug 15 11:01:07 2025
# 模型参数配置
@author: dragon
"""

class ModelParams:
    # SVM参数
    SVM_PARAMS = {
        'C': [0.1, 1, 10, 100],
        'gamma': [0.001, 0.01, 0.1, 1],
        'kernel': ['rbf', 'linear']
    }
    
    # 特征选择参数
    FEATURE_SELECTION = {
        'k_features': 10,
        'score_func': 'f_classif'
    }
    
    # 数据预处理参数
    PREPROCESSING = {
        'test_size': 0.3,
        'random_state': 42,
        'sampling_strategy': 'auto'
    }
