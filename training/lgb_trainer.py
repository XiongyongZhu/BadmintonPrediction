# -*- coding: utf-8 -*-
"""
Created on Tue Aug 26 10:31:44 2025
LightGBM预测模型
@author: dragon
"""

import lightgbm as lgb
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import logging

class LightGBMTrainer:
    def __init__(self):
        self.best_params = None
        self.best_model = None
        self.cv_results = None
    
    def train(self, X_train, y_train):
        """
        使用网格搜索训练LightGBM模型并找到最佳参数
        
        参数:
        X_train -- 训练特征
        y_train -- 训练标签
        
        返回:
        最佳模型
        """
        try:
            # 基本模型
            base_model = lgb.LGBMClassifier(
                objective='binary',
                random_state=42,
                n_jobs=-1,
                verbose=-1  # 减少输出噪音
            )
            
            # 参数网格
            param_grid = {
                'learning_rate': [0.1, 0.2, 0.4],
                'max_depth': [3, 5, 7],
                'num_leaves': [15, 31, 63],
                'min_child_samples': [20, 50, 200],
                'subsample': [0.8, 1.0],
                'colsample_bytree': [0.6, 0.8, 1.0],
                'reg_alpha': [0, 0.1, 1],
                'reg_lambda': [0, 0.1, 1]
            }
            
            # 交叉验证策略
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            
            # 网格搜索
            grid_search = GridSearchCV(
                estimator=base_model,
                param_grid=param_grid,
                cv=cv,
                scoring='f1',
                n_jobs=-1,
                verbose=1
            )
            
            # 执行网格搜索
            grid_search.fit(X_train, y_train)
            
            # 保存结果和最佳模型
            self.best_params = grid_search.best_params_
            self.best_model = grid_search.best_estimator_
            self.cv_results = {
                'best_score': grid_search.best_score_,
                'all_results': grid_search.cv_results_
            }
            
            logging.info(f"LightGBM训练完成，最佳参数: {self.best_params}")
            logging.info(f"交叉验证最佳准确率: {self.cv_results['best_score']:.4f}")
            
            return self.best_model
        
        except Exception as e:
            logging.error(f"LightGBM训练失败: {str(e)}")
            raise
    
    def get_best_params(self):
        """返回最佳参数"""
        return self.best_params
    
    def get_best_model(self):
        """返回最佳模型"""
        return self.best_model
    
    def get_cv_results(self):
        """返回交叉验证结果"""
        return self.cv_results